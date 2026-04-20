"""这个模块封装 BGE 检索器和 BGE 重排器，在检索主线和实验流程中提供 dense recall 与 rerank 能力。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from legal_contract_rag.retrieval.base import LexicalReranker, TfidfRetriever
from legal_contract_rag.types import ChunkRecord, RetrievalHit


@dataclass(slots=True)
class BGERetriever:
    model_name: str = "BAAI/bge-m3"
    device: str | None = None
    _fallback: bool = field(init=False, default=False)
    _encoder: object | None = field(init=False, default=None)
    _chunk_ids: list[str] = field(init=False, default_factory=list)
    _matrix: object | None = field(init=False, default=None)
    _tfidf: TfidfRetriever | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            self._encoder.max_seq_length = 512
        except Exception:
            self._fallback = True
            self._tfidf = TfidfRetriever()

    def index(self, chunks: list[ChunkRecord]) -> None:
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        if self._fallback:
            self._tfidf.index(chunks)
            return
        batch_size = 24 if self.device == "cpu" else 32
        self._matrix = self._encoder.encode(
            [chunk.content for chunk in chunks],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalHit]:
        if self._fallback:
            return self._tfidf.retrieve(query, top_k=top_k)
        query_vector = self._encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = cosine_similarity(query_vector, self._matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(chunk_id=self._chunk_ids[index], bi_score=float(scores[index]), rank_before=rank + 1)
            for rank, index in enumerate(indices)
        ]


@dataclass(slots=True)
class BGEReranker:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str | None = None
    _fallback: bool = field(init=False, default=False)
    _tokenizer: object | None = field(init=False, default=None)
    _model: object | None = field(init=False, default=None)
    _lexical: LexicalReranker | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=True,
            ).to(self.device)
            self._model.eval()
        except Exception:
            self._fallback = True
            self._lexical = LexicalReranker()

    def score(self, query: str, documents: list[str]) -> list[float]:
        if self._fallback:
            return self._lexical.score(query, documents)
        pairs = [[query, document] for document in documents]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        return [float(score) for score in scores]
