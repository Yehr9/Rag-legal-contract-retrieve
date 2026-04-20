from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    source: str
    source_url: str | None
    agreement_type: str | None
    title: str
    raw_text: str
    cleaned_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    content: str
    token_count: int
    section_path: list[str] = field(default_factory=list)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    source_span: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    bi_score: float
    rerank_score: float | None = None
    rank_before: int | None = None
    rank_after: int | None = None
    retrieval_mode: str | None = None
    source: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalExample:
    query_id: str
    query: str
    answer: str | None
    positive_chunk_ids: list[str]
    positive_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrieverTriplet:
    query: str
    positive_chunk_id: str
    negative_chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RerankerPair:
    query: str
    chunk_id: str
    chunk_text: str
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryAnalysisResult:
    normalized_query: str
    query_variants: list[str]
    must_terms: list[str]
    filters: dict[str, Any]
    intent: str
    expected_answer_type: str
    rewrite_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedCandidate:
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_mode: str = "dense_only"
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rank: int | None = None

    def to_hit(self) -> RetrievalHit:
        base_score = self.fusion_score
        if base_score is None:
            base_score = self.dense_score if self.dense_score is not None else self.sparse_score or 0.0
        return RetrievalHit(
            chunk_id=self.chunk_id,
            bi_score=float(base_score),
            rank_before=self.rank,
            retrieval_mode=self.retrieval_mode,
            dense_score=self.dense_score,
            sparse_score=self.sparse_score,
            fusion_score=self.fusion_score,
            source="retrieval",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RerankDecision:
    action: str
    reason: str
    confidence_signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphTrace:
    node_name: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
