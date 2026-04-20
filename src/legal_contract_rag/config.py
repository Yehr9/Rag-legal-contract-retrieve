"""这个模块定义项目的配置结构与 YAML 加载逻辑，被 CLI、检索主流程、实验流程和数据处理流程共同使用。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ChunkingConfig:
    policy: str = "structure_aware_v2"
    chunk_size: int = 512
    overlap: int = 100
    max_size: int = 1536


@dataclass(slots=True)
class RetrievalConfig:
    retriever_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    fine_tuned_retriever_model: str | None = None
    fine_tuned_reranker_model: str | None = None
    retrieval_mode: str = "hybrid_rrf"
    use_elasticsearch: bool = True
    es_url: str = "http://localhost:9200"
    es_index_name: str = "legal_contract_rag"
    es_user: str | None = None
    es_password: str | None = None
    dense_top_k: int = 20
    sparse_top_k: int = 20
    top_k_recall: int = 20
    top_k_rerank: int = 5
    rerank_threshold: float = 0.0


@dataclass(slots=True)
class GenerationConfig:
    model_name: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    context_chunks: int = 5
    fallback_sentence_count: int = 2


@dataclass(slots=True)
class EvaluationConfig:
    include_answer_eval: bool = False


@dataclass(slots=True)
class GraphConfig:
    max_retry: int = 1


@dataclass(slots=True)
class QueryAnalysisConfig:
    enable_variants: bool = True
    max_variants: int = 3


@dataclass(slots=True)
class RoutingConfig:
    min_rerank_score: float = 0.2
    min_must_term_coverage: float = 0.4
    min_returned_docs: int = 2
    min_top1_margin_over_top5_mean: float = 0.02


@dataclass(slots=True)
class PathsConfig:
    input_documents: list[str] = field(default_factory=list)
    eval_examples: str | None = None
    output_dir: str = "artifacts"


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    query_analysis: QueryAnalysisConfig = field(default_factory=QueryAnalysisConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    chunking_raw = raw.get("chunking", {})
    if chunking_raw.get("policy") == "structure_aware":
        chunking_raw = {**chunking_raw, "policy": "structure_aware_v2"}
    return AppConfig(
        paths=PathsConfig(**raw.get("paths", {})),
        chunking=ChunkingConfig(**chunking_raw),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        graph=GraphConfig(**raw.get("graph", {})),
        query_analysis=QueryAnalysisConfig(**raw.get("query_analysis", {})),
        routing=RoutingConfig(**raw.get("routing", {})),
        extra=raw.get("extra", {}),
    )
