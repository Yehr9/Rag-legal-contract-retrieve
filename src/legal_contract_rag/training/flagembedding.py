"""这个模块负责生成 FlagEmbedding 训练样本格式，并拼装微调命令，供 CLI 和训练准备脚本调用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

from legal_contract_rag.types import ChunkRecord, EvalExample
from legal_contract_rag.utils import write_jsonl


@dataclass(slots=True)
class FlagEmbeddingFinetuneConfig:
    train_data_path: str
    output_dir: str
    base_model: str
    learning_rate: float = 1e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    train_group_size: int = 8
    query_max_len: int = 256
    passage_max_len: int = 512
    use_fp16: bool = True
    extra_args: list[str] | None = None


def build_embedder_train_rows(
    examples: Iterable[EvalExample],
    chunk_by_id: dict[str, ChunkRecord],
    candidate_negatives_by_query: dict[str, list[str]],
) -> list[dict]:
    rows: list[dict] = []
    for example in examples:
        positives = [chunk_by_id[chunk_id].content for chunk_id in example.positive_chunk_ids if chunk_id in chunk_by_id]
        negatives = [
            chunk_by_id[chunk_id].content
            for chunk_id in candidate_negatives_by_query.get(example.query_id, [])
            if chunk_id in chunk_by_id and chunk_id not in example.positive_chunk_ids
        ]
        if positives and negatives:
            rows.append({"query": example.query, "pos": positives, "neg": negatives})
    return rows


def build_reranker_train_rows(
    examples: Iterable[EvalExample],
    chunk_by_id: dict[str, ChunkRecord],
    candidate_negatives_by_query: dict[str, list[str]],
) -> list[dict]:
    rows: list[dict] = []
    for example in examples:
        positives = [chunk_by_id[chunk_id].content for chunk_id in example.positive_chunk_ids if chunk_id in chunk_by_id]
        negatives = [
            chunk_by_id[chunk_id].content
            for chunk_id in candidate_negatives_by_query.get(example.query_id, [])
            if chunk_id in chunk_by_id and chunk_id not in example.positive_chunk_ids
        ]
        if positives and negatives:
            rows.append({"query": example.query, "pos": positives, "neg": negatives})
    return rows


def save_flagembedding_train_rows(rows: list[dict], output_path: str | Path) -> None:
    write_jsonl(output_path, rows)


def build_embedder_command(config: FlagEmbeddingFinetuneConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--standalone",
        "--rdzv-conf",
        "use_libuv=0",
        "-m",
        "FlagEmbedding.finetune.embedder.encoder_only.base",
        "--model_name_or_path",
        config.base_model,
        "--train_data",
        config.train_data_path,
        "--cache_path",
        str(Path(config.output_dir) / "cache"),
        "--train_group_size",
        str(config.train_group_size),
        "--query_max_len",
        str(config.query_max_len),
        "--passage_max_len",
        str(config.passage_max_len),
        "--learning_rate",
        str(config.learning_rate),
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.per_device_train_batch_size),
        "--output_dir",
        config.output_dir,
        "--save_strategy",
        "epoch",
        "--logging_steps",
        "10",
        "--overwrite_output_dir",
        "--do_train",
    ]
    if config.use_fp16:
        command.append("--fp16")
    if config.extra_args:
        command.extend(config.extra_args)
    return command


def build_reranker_command(config: FlagEmbeddingFinetuneConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--standalone",
        "--rdzv-conf",
        "use_libuv=0",
        "-m",
        "FlagEmbedding.finetune.reranker.encoder_only.base",
        "--model_name_or_path",
        config.base_model,
        "--train_data",
        config.train_data_path,
        "--cache_path",
        str(Path(config.output_dir) / "cache"),
        "--train_group_size",
        str(config.train_group_size),
        "--query_max_len",
        str(config.query_max_len),
        "--passage_max_len",
        str(config.passage_max_len),
        "--learning_rate",
        str(config.learning_rate),
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.per_device_train_batch_size),
        "--output_dir",
        config.output_dir,
        "--save_strategy",
        "epoch",
        "--logging_steps",
        "10",
        "--overwrite_output_dir",
        "--do_train",
    ]
    if config.use_fp16:
        command.append("--fp16")
    if config.extra_args:
        command.extend(config.extra_args)
    return command
