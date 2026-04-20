"""这个模块负责组织并运行项目的实验套件，输出 chunking、检索升级和 reranker 微调等对比结果。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd

from legal_contract_rag.config import AppConfig
from legal_contract_rag.answering.generation import AnswerGenerator
from legal_contract_rag.corpus.chunking import Chunker, ChunkingPolicy
from legal_contract_rag.corpus.loaders import load_document_records
from legal_contract_rag.evaluation.answer_metrics import compute_answer_metrics
from legal_contract_rag.evaluation.benchmark import map_examples_to_positive_chunks
from legal_contract_rag.evaluation.retrieval_metrics import compute_retrieval_metrics
from legal_contract_rag.pipelines.rag import RAGPipeline
from legal_contract_rag.retrieval.bge import BGEReranker
from legal_contract_rag.types import EvalExample
from legal_contract_rag.utils import read_jsonl, write_json


def load_eval_examples(path: str) -> list[EvalExample]:
    return [EvalExample(**row) for row in read_jsonl(path)]


def run_experiment_suite(
    config: AppConfig,
    *,
    suite_names: Iterable[str] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    documents = load_document_records(config.paths.input_documents)
    examples = load_eval_examples(config.paths.eval_examples) if config.paths.eval_examples else []
    generator = AnswerGenerator(config.generation) if config.evaluation.include_answer_eval else None
    suites = {
        "chunking_ablation": [
            {"chunking_policy": "fixed", "retrieval_mode": "hybrid_rrf", "rerank_mode": "base_reranker", "fine_tuned": False},
            {"chunking_policy": "fixed_overlap", "retrieval_mode": "hybrid_rrf", "rerank_mode": "base_reranker", "fine_tuned": False},
            {
                "chunking_policy": "structure_aware_v2",
                "retrieval_mode": "hybrid_rrf",
                "rerank_mode": "base_reranker",
                "fine_tuned": False,
            },
        ],
        "retrieval_upgrade_line": [
            {
                "chunking_policy": "structure_aware_v2",
                "retrieval_mode": "dense_only",
                "rerank_mode": "disabled",
                "fine_tuned": False,
            },
            {
                "chunking_policy": "structure_aware_v2",
                "retrieval_mode": "hybrid_rrf",
                "rerank_mode": "disabled",
                "fine_tuned": False,
            },
            {
                "chunking_policy": "structure_aware_v2",
                "retrieval_mode": "hybrid_rrf",
                "rerank_mode": "base_reranker",
                "fine_tuned": False,
            },
        ],
    }
    if config.retrieval.fine_tuned_reranker_model:
        suites["rerank_finetune_ablation"] = [
            {
                "chunking_policy": "structure_aware_v2",
                "retrieval_mode": "hybrid_rrf",
                "rerank_mode": "base_reranker",
                "fine_tuned": False,
            },
            {
                "chunking_policy": "structure_aware_v2",
                "retrieval_mode": "hybrid_rrf",
                "rerank_mode": "finetuned_reranker",
                "fine_tuned": True,
            },
        ]
    if suite_names is not None:
        requested = set(suite_names)
        suites = {name: rows for name, rows in suites.items() if name in requested}

    results: dict[str, list[dict]] = {}
    for suite_name, experiments in suites.items():
        rows: list[dict] = []
        for experiment in experiments:
            experiment_config = _config_with_model_overrides(config, experiment["fine_tuned"])
            chunks = _build_chunks(documents, experiment_config, experiment["chunking_policy"])
            mapped_examples = map_examples_to_positive_chunks(examples, chunks)
            pipeline = RAGPipeline(
                config=experiment_config,
                reranker=_build_reranker(experiment_config, experiment["rerank_mode"]),
            )
            pipeline.index(chunks)
            recall_predictions = {
                example.query_id: pipeline.retrieve_recall_candidates(
                    example.query,
                    retrieval_mode=experiment["retrieval_mode"],
                )
                for example in mapped_examples
            }
            predictions = {
                example.query_id: pipeline.retrieve(
                    example.query,
                    retrieval_mode=experiment["retrieval_mode"],
                    rerank_mode=experiment["rerank_mode"],
                )
                for example in mapped_examples
            }
            retrieval_metrics = compute_retrieval_metrics(
                mapped_examples,
                predictions,
                recall_predictions=recall_predictions,
                recall_k=experiment_config.retrieval.top_k_recall,
                rank_k=experiment_config.retrieval.top_k_rerank,
            )
            answer_metrics: dict[str, float] = {}
            retry_triggered_rate = 0.0
            rewrite_modes: list[str] = []
            if mapped_examples:
                run_states = [
                    pipeline.run(
                        example.query,
                        retrieval_mode=experiment["retrieval_mode"],
                        rerank_mode=experiment["rerank_mode"],
                    )
                    for example in mapped_examples
                ]
                retry_triggered_rate = sum(1.0 for state in run_states if state.get("retry_triggered")) / len(run_states)
                rewrite_modes = [state.get("rewrite_mode", "strict") for state in run_states]
                if config.evaluation.include_answer_eval and generator is not None:
                    answer_predictions = {
                        example.query_id: run_states[index].get("answer", "")
                        for index, example in enumerate(mapped_examples)
                    }
                    answer_metrics = compute_answer_metrics(mapped_examples, answer_predictions)
            rows.append(
                {
                    "dataset_name": "real_legal_contracts",
                    "split": "eval",
                    "example_count": len(mapped_examples),
                    "chunking_policy": experiment["chunking_policy"],
                    "retrieval_mode": experiment["retrieval_mode"],
                    "rerank_mode": experiment["rerank_mode"],
                    "retriever_model": experiment_config.retrieval.retriever_model,
                    "reranker_model": _select_reranker_model(experiment_config, experiment["fine_tuned"])
                    if experiment["rerank_mode"] != "disabled"
                    else None,
                    "retry_triggered": retry_triggered_rate,
                    "rewrite_mode": _dominant_value(rewrite_modes, default="strict"),
                    "fine_tuned": experiment["fine_tuned"],
                    **retrieval_metrics,
                    **answer_metrics,
                }
            )
        results[suite_name] = rows
        if save_dir is not None:
            save_experiment_suite({suite_name: rows}, save_dir)
    return results


def save_experiment_suite(results: dict[str, list[dict]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        write_json(output_dir / f"{name}.json", rows)
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)


def _build_chunks(documents, config: AppConfig, policy_name: str) -> list:
    chunker = Chunker(
        ChunkingPolicy(
            name=policy_name,
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
            max_size=config.chunking.max_size,
        )
    )
    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))
    return chunks


def _build_reranker(config: AppConfig, rerank_mode: str) -> BGEReranker | None:
    if rerank_mode == "disabled":
        return None
    return BGEReranker(model_name=config.retrieval.reranker_model)


def _config_with_model_overrides(config: AppConfig, fine_tuned: bool) -> AppConfig:
    updated = deepcopy(config)
    if fine_tuned and config.retrieval.fine_tuned_retriever_model:
        updated.retrieval.retriever_model = config.retrieval.fine_tuned_retriever_model
    if fine_tuned and config.retrieval.fine_tuned_reranker_model:
        updated.retrieval.reranker_model = config.retrieval.fine_tuned_reranker_model
    return updated


def _select_reranker_model(config: AppConfig, fine_tuned: bool) -> str:
    if fine_tuned and config.retrieval.fine_tuned_reranker_model:
        return config.retrieval.fine_tuned_reranker_model
    return config.retrieval.reranker_model


def _dominant_value(values: list[str], *, default: str) -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]
