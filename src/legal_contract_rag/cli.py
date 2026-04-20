"""这个模块提供项目的命令行入口函数，连接语料准备、训练样本构建、实验评测和 FlagEmbedding 微调相关流程，供 pyproject.toml 中的 CLI 命令调用。"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from legal_contract_rag.config import load_config
from legal_contract_rag.corpus.chunking import Chunker, ChunkingPolicy
from legal_contract_rag.corpus.loaders import load_document_records
from legal_contract_rag.evaluation.benchmark import map_examples_to_positive_chunks
from legal_contract_rag.evaluation.experiments import run_experiment_suite, save_experiment_suite
from legal_contract_rag.pipelines.rag import RAGPipeline
from legal_contract_rag.training.builders import build_reranker_pairs, build_retriever_triplets
from legal_contract_rag.training.flagembedding import (
    FlagEmbeddingFinetuneConfig,
    build_embedder_command,
    build_embedder_train_rows,
    build_reranker_command,
    build_reranker_train_rows,
    save_flagembedding_train_rows,
)
from legal_contract_rag.types import EvalExample
from legal_contract_rag.utils import read_jsonl, write_jsonl


def prepare_corpus_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    documents = load_document_records(config.paths.input_documents)
    output_dir = Path(config.paths.output_dir) / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "documents.jsonl", [document.to_dict() for document in documents])
    for policy_name in ("fixed", "fixed_overlap", "structure_aware_v2"):
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
        write_jsonl(output_dir / f"chunks.{policy_name}.jsonl", [chunk.to_dict() for chunk in chunks])


def build_training_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    documents = load_document_records(config.paths.input_documents)
    chunker = Chunker(
        ChunkingPolicy(
            name=config.chunking.policy,
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
            max_size=config.chunking.max_size,
        )
    )
    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))
    examples = [EvalExample(**row) for row in read_jsonl(config.paths.eval_examples)]
    examples = map_examples_to_positive_chunks(examples, chunks)
    pipeline = RAGPipeline(config=config)
    triplets = build_retriever_triplets(examples, chunks, pipeline)
    rerank_pairs = build_reranker_pairs(examples, chunks, pipeline)
    output_dir = Path(config.paths.output_dir) / "training"
    write_jsonl(output_dir / "retriever_triplets.jsonl", [row.to_dict() for row in triplets])
    write_jsonl(output_dir / "reranker_pairs.jsonl", [row.to_dict() for row in rerank_pairs])


def run_eval_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", action="append", dest="suites")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config.paths.output_dir) / "eval"
    results = run_experiment_suite(config, suite_names=args.suites, save_dir=output_dir)
    save_experiment_suite(results, output_dir)


def prepare_flagembedding_data_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    documents = load_document_records(config.paths.input_documents)
    chunker = Chunker(
        ChunkingPolicy(
            name=config.chunking.policy,
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap,
            max_size=config.chunking.max_size,
        )
    )
    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    examples = [EvalExample(**row) for row in read_jsonl(config.paths.eval_examples)]
    examples = map_examples_to_positive_chunks(examples, chunks)
    pipeline = RAGPipeline(config=config)
    pipeline.index(chunks)
    negatives_by_query = {
        example.query_id: [
            candidate.chunk_id
            for candidate in pipeline.hybrid_retriever.retrieve(
                example.query,
                filters=None,
                retrieval_mode="dense_only",
                top_k=50,
            )
        ]
        for example in examples
    }
    output_dir = Path(config.paths.output_dir) / "flagembedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder_rows = build_embedder_train_rows(examples, chunk_by_id, negatives_by_query)
    reranker_rows = build_reranker_train_rows(examples, chunk_by_id, negatives_by_query)
    save_flagembedding_train_rows(embedder_rows, output_dir / "embedder_train.jsonl")
    save_flagembedding_train_rows(reranker_rows, output_dir / "reranker_train.jsonl")


def run_flagembedding_embedder_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-m3")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--train-group-size", type=int, default=8)
    parser.add_argument("--query-max-len", type=int, default=256)
    parser.add_argument("--passage-max-len", type=int, default=512)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    cfg = FlagEmbeddingFinetuneConfig(
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        train_group_size=args.train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        use_fp16=not args.no_fp16,
        extra_args=extra,
    )
    command = build_embedder_command(cfg)
    if args.dry_run:
        print(" ".join(command))
        return
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")
    subprocess.run(command, check=True, env=env)


def run_flagembedding_reranker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--train-group-size", type=int, default=8)
    parser.add_argument("--query-max-len", type=int, default=256)
    parser.add_argument("--passage-max-len", type=int, default=512)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    cfg = FlagEmbeddingFinetuneConfig(
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        train_group_size=args.train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        use_fp16=not args.no_fp16,
        extra_args=extra,
    )
    command = build_reranker_command(cfg)
    if args.dry_run:
        print(" ".join(command))
        return
    env = os.environ.copy()
    env.setdefault("USE_LIBUV", "0")
    subprocess.run(command, check=True, env=env)
