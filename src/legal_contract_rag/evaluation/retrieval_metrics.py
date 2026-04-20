"""这个模块实现检索评测指标计算，供实验流程统计 Recall、Precision、NDCG 和 citation hit rate。"""

from __future__ import annotations

import math
from typing import Iterable

from legal_contract_rag.types import EvalExample, RetrievalHit


def precision_at_k(retrieved_ids: list[str], positive_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in positive_ids) / k


def recall_at_k(retrieved_ids: list[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    return sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in positive_ids) / len(positive_ids)


def ndcg_at_k(retrieved_ids: list[str], positive_ids: set[str], k: int) -> float:
    dcg = 0.0
    for index, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in positive_ids:
            dcg += 1.0 / math.log(index + 1, 2)
    ideal_count = min(len(positive_ids), k)
    idcg = sum(1.0 / math.log(index + 1, 2) for index in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def compute_retrieval_metrics(
    examples: Iterable[EvalExample],
    predictions: dict[str, list[RetrievalHit]],
    *,
    recall_predictions: dict[str, list[RetrievalHit]] | None = None,
    recall_k: int = 20,
    rank_k: int = 5,
) -> dict[str, float]:
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    recall20_scores: list[float] = []
    evidence_hits: list[float] = []

    for example in examples:
        retrieved_ids = [hit.chunk_id for hit in predictions.get(example.query_id, [])]
        recall_retrieved_ids = [hit.chunk_id for hit in (recall_predictions or predictions).get(example.query_id, [])]
        positive_ids = set(example.positive_chunk_ids)
        precision_scores.append(precision_at_k(retrieved_ids, positive_ids, rank_k))
        recall_scores.append(recall_at_k(retrieved_ids, positive_ids, rank_k))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, positive_ids, rank_k))
        recall20_scores.append(recall_at_k(recall_retrieved_ids, positive_ids, recall_k))
        evidence_hits.append(1.0 if any(chunk_id in positive_ids for chunk_id in retrieved_ids[:rank_k]) else 0.0)

    return {
        f"Precision@{rank_k}": _mean(precision_scores),
        f"Recall@{rank_k}": _mean(recall_scores),
        f"NDCG@{rank_k}": _mean(ndcg_scores),
        f"Recall@{recall_k}": _mean(recall20_scores),
        "citation_hit_rate": _mean(evidence_hits),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
