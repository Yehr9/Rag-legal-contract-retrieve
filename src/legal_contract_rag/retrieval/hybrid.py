"""这个模块实现 hybrid retrieval，负责融合 dense 与 sparse 召回，并在可用时接入 Elasticsearch 检索后端。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from legal_contract_rag.config import RetrievalConfig
from legal_contract_rag.retrieval.base import TfidfRetriever
from legal_contract_rag.retrieval.bge import BGERetriever
from legal_contract_rag.types import ChunkRecord, RetrievedCandidate, RetrievalHit


@dataclass(slots=True)
class HybridRetriever:
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    dense_retriever: BGERetriever | None = None
    sparse_retriever: TfidfRetriever | None = None
    _chunks_by_id: dict[str, ChunkRecord] = field(init=False, default_factory=dict)
    _es_retrievers: dict[str, Any] = field(init=False, default_factory=dict)
    _es_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.dense_retriever = self.dense_retriever or BGERetriever(model_name=self.config.retriever_model)
        self.sparse_retriever = self.sparse_retriever or TfidfRetriever()

    def index(self, chunks: list[ChunkRecord]) -> None:
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.dense_retriever.index(chunks)
        self.sparse_retriever.index(chunks)
        self._es_active = False
        self._es_retrievers = {}
        if self.config.use_elasticsearch:
            self._try_index_elasticsearch(chunks)

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        retrieval_mode: str = "hybrid_rrf",
        top_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        top_k = top_k or self.config.top_k_recall
        if self._es_active:
            try:
                return self._retrieve_with_elasticsearch(query, filters=filters, retrieval_mode=retrieval_mode, top_k=top_k)
            except Exception:
                self._es_active = False
        if retrieval_mode == "dense_only":
            dense_hits = self.dense_retriever.retrieve(query, top_k=top_k)
            return self._hits_to_candidates(dense_hits, retrieval_mode="dense_only", filters=filters)
        return self._fallback_hybrid(query, filters=filters, top_k=top_k)

    def _fallback_hybrid(
        self,
        query: str,
        *,
        filters: dict[str, str] | None,
        top_k: int,
    ) -> list[RetrievedCandidate]:
        dense_hits = self.dense_retriever.retrieve(query, top_k=self.config.dense_top_k)
        sparse_hits = self.sparse_retriever.retrieve(query, top_k=self.config.sparse_top_k)
        dense_rank = {hit.chunk_id: hit.rank_before or rank for rank, hit in enumerate(dense_hits, start=1)}
        sparse_rank = {hit.chunk_id: hit.rank_before or rank for rank, hit in enumerate(sparse_hits, start=1)}
        dense_score = {hit.chunk_id: hit.bi_score for hit in dense_hits}
        sparse_score = {hit.chunk_id: hit.bi_score for hit in sparse_hits}
        combined_ids = list(dict.fromkeys([hit.chunk_id for hit in dense_hits] + [hit.chunk_id for hit in sparse_hits]))
        scored = []
        for chunk_id in combined_ids:
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (60 + dense_rank[chunk_id])
            if chunk_id in sparse_rank:
                score += 1.0 / (60 + sparse_rank[chunk_id])
            scored.append((score, chunk_id))
        scored.sort(reverse=True)
        candidates: list[RetrievedCandidate] = []
        for rank, (fusion_score, chunk_id) in enumerate(scored, start=1):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None or not self._passes_filters(chunk, filters):
                continue
            candidates.append(
                RetrievedCandidate(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    metadata=dict(chunk.metadata),
                    retrieval_mode="hybrid_rrf",
                    dense_score=dense_score.get(chunk_id),
                    sparse_score=sparse_score.get(chunk_id),
                    fusion_score=float(fusion_score),
                    rank=rank,
                )
            )
            if len(candidates) >= top_k:
                break
        return candidates

    def _hits_to_candidates(
        self,
        hits: list[RetrievalHit],
        *,
        retrieval_mode: str,
        filters: dict[str, str] | None,
    ) -> list[RetrievedCandidate]:
        candidates: list[RetrievedCandidate] = []
        for rank, hit in enumerate(hits, start=1):
            chunk = self._chunks_by_id.get(hit.chunk_id)
            if chunk is None or not self._passes_filters(chunk, filters):
                continue
            candidates.append(
                RetrievedCandidate(
                    chunk_id=hit.chunk_id,
                    content=chunk.content,
                    metadata=dict(chunk.metadata),
                    retrieval_mode=retrieval_mode,
                    dense_score=hit.bi_score if retrieval_mode == "dense_only" else hit.dense_score,
                    sparse_score=hit.sparse_score,
                    fusion_score=hit.fusion_score,
                    rank=rank,
                )
            )
        return candidates

    def _passes_filters(self, chunk: ChunkRecord, filters: dict[str, str] | None) -> bool:
        if not filters:
            return True
        chunk_text = chunk.content.lower()
        section_text = " > ".join(chunk.section_path).lower()
        metadata = chunk.metadata
        for key, value in filters.items():
            lowered = value.lower()
            if key == "agreement_type" and str(metadata.get("agreement_type", "")).lower() != lowered:
                return False
            if key == "clause_type" and str(metadata.get("clause_type", "")).lower() != lowered:
                return False
            if key == "section_path" and lowered not in section_text:
                return False
            if key == "jurisdiction" and lowered not in chunk_text and lowered not in section_text:
                return False
        return True

    def _try_index_elasticsearch(self, chunks: list[ChunkRecord]) -> None:
        try:
            from langchain_core.documents import Document
            from langchain_elasticsearch import ElasticsearchStore
        except Exception:
            return

        embeddings = _SentenceTransformerEmbeddings(self.config.retriever_model)
        store_kwargs = {
            "es_url": self.config.es_url,
            "index_name": self.config.es_index_name,
            "embedding": embeddings,
        }
        if self.config.es_user and self.config.es_password:
            store_kwargs["es_user"] = self.config.es_user
            store_kwargs["es_password"] = self.config.es_password

        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "agreement_type": chunk.metadata.get("agreement_type"),
                    "clause_type": chunk.metadata.get("clause_type"),
                    "section_path": " > ".join(chunk.section_path),
                    "source_dataset": chunk.metadata.get("source_dataset"),
                    "parent_doc_id": chunk.metadata.get("parent_doc_id"),
                },
            )
            for chunk in chunks
        ]

        try:
            hybrid_strategy = ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True)
            dense_strategy = ElasticsearchStore.ApproxRetrievalStrategy(hybrid=False)
        except Exception:
            return

        dense_store = ElasticsearchStore(strategy=dense_strategy, **store_kwargs)
        hybrid_store = ElasticsearchStore(strategy=hybrid_strategy, **store_kwargs)
        dense_store.add_documents(documents, ids=[chunk.chunk_id for chunk in chunks])
        self._es_retrievers["dense_only"] = dense_store.as_retriever(
            search_kwargs={"k": self.config.top_k_recall}
        )
        self._es_retrievers["hybrid_rrf"] = hybrid_store.as_retriever(
            search_kwargs={"k": self.config.top_k_recall}
        )
        self._es_active = True

    def _retrieve_with_elasticsearch(
        self,
        query: str,
        *,
        filters: dict[str, str] | None,
        retrieval_mode: str,
        top_k: int,
    ) -> list[RetrievedCandidate]:
        retriever = self._es_retrievers[retrieval_mode]
        search_kwargs = {"k": top_k}
        es_filter = _filters_to_es_query(filters)
        if es_filter:
            search_kwargs["filter"] = es_filter
        retriever.search_kwargs = search_kwargs
        docs = retriever.invoke(query)
        candidates: list[RetrievedCandidate] = []
        for rank, doc in enumerate(docs, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            candidates.append(
                RetrievedCandidate(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    metadata=dict(doc.metadata),
                    retrieval_mode=retrieval_mode,
                    rank=rank,
                    fusion_score=float(max(0.0, 1.0 - (rank - 1) / max(1, top_k))),
                )
            )
        return candidates


class _SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, local_files_only=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def _filters_to_es_query(filters: dict[str, str] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    must_clauses: list[dict[str, Any]] = []
    for key, value in filters.items():
        if key == "agreement_type":
            must_clauses.append({"term": {"metadata.agreement_type.keyword": value}})
        elif key == "clause_type":
            must_clauses.append({"term": {"metadata.clause_type.keyword": value}})
        elif key == "section_path":
            must_clauses.append({"match_phrase": {"metadata.section_path": value}})
        elif key == "jurisdiction":
            must_clauses.append({"match_phrase": {"text": value}})
    if not must_clauses:
        return None
    return {"bool": {"must": must_clauses}}
