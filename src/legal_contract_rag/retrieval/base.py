"""这个模块定义检索与重排的基础接口，并提供 TF-IDF 检索器和词法重排器等轻量基线实现，供主流程与测试回退使用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from legal_contract_rag.types import ChunkRecord, RetrievalHit


class BaseRetriever(Protocol):
    model_name: str

    def index(self, chunks: list[ChunkRecord]) -> None: ...

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalHit]: ...


class BaseReranker(Protocol):
    model_name: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass(slots=True)
class InMemoryVectorStore:
    retriever: BaseRetriever
    chunks: list[ChunkRecord] | None = None

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.retriever.index(chunks)

    def similarity_search(self, query: str, k: int = 20) -> list[ChunkRecord]:
        if not self.chunks:
            return []
        hits = self.retriever.retrieve(query, top_k=k)
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [by_id[hit.chunk_id] for hit in hits if hit.chunk_id in by_id]


class TfidfRetriever:
    def __init__(self) -> None:
        self.model_name = "tfidf-baseline"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.chunk_ids: list[str] = []
        self.matrix = None

    def index(self, chunks: list[ChunkRecord]) -> None:
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform([chunk.content for chunk in chunks])

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalHit]:
        if self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(chunk_id=self.chunk_ids[index], bi_score=float(scores[index]), rank_before=rank + 1)
            for rank, index in enumerate(indices)
        ]


class LexicalReranker:
    def __init__(self) -> None:
        self.model_name = "lexical-reranker-baseline"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)

    def score(self, query: str, documents: list[str]) -> list[float]:
        matrix = self.vectorizer.fit_transform([query, *documents])
        scores = cosine_similarity(matrix[0], matrix[1:]).flatten()
        return [float(score) for score in scores]
