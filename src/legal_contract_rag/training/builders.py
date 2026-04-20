"""这个模块负责根据评测样本和召回结果构建 retriever triplet 与 reranker pair，服务于训练数据生成流程。"""

from __future__ import annotations

from typing import Iterable

from legal_contract_rag.pipelines.rag import RAGPipeline
from legal_contract_rag.types import ChunkRecord, EvalExample, RerankerPair, RetrieverTriplet


def build_retriever_triplets(
    examples: Iterable[EvalExample],
    chunks: list[ChunkRecord],
    pipeline: RAGPipeline,
    hard_negative_k: int = 50,
) -> list[RetrieverTriplet]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    pipeline.index(chunks)
    triplets: list[RetrieverTriplet] = []
    for example in examples:
        positives = [chunk_id for chunk_id in example.positive_chunk_ids if chunk_id in chunk_by_id]
        if not positives:
            continue
        hits = [
            candidate.to_hit()
            for candidate in pipeline.hybrid_retriever.retrieve(
                example.query,
                filters=None,
                retrieval_mode="dense_only",
                top_k=hard_negative_k,
            )
        ]
        hard_negatives = [hit.chunk_id for hit in hits if hit.chunk_id not in positives]
        if not hard_negatives:
            continue
        for positive in positives:
            triplets.append(
                RetrieverTriplet(
                    query=example.query,
                    positive_chunk_id=positive,
                    negative_chunk_id=hard_negatives[0],
                    metadata={"query_id": example.query_id},
                )
            )
    return triplets


def build_reranker_pairs(
    examples: Iterable[EvalExample],
    chunks: list[ChunkRecord],
    pipeline: RAGPipeline,
    hard_negative_k: int = 50,
) -> list[RerankerPair]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    pipeline.index(chunks)
    pairs: list[RerankerPair] = []
    for example in examples:
        positives = set(example.positive_chunk_ids)
        for positive in positives:
            chunk = chunk_by_id.get(positive)
            if chunk:
                pairs.append(
                    RerankerPair(
                        query=example.query,
                        chunk_id=chunk.chunk_id,
                        chunk_text=chunk.content,
                        label=1,
                        metadata={"query_id": example.query_id},
                    )
                )
        hits = [
            candidate.to_hit()
            for candidate in pipeline.hybrid_retriever.retrieve(
                example.query,
                filters=None,
                retrieval_mode="dense_only",
                top_k=hard_negative_k,
            )
        ]
        for hit in hits:
            if hit.chunk_id in positives:
                continue
            chunk = chunk_by_id.get(hit.chunk_id)
            if not chunk:
                continue
            pairs.append(
                RerankerPair(
                    query=example.query,
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.content,
                    label=0,
                    metadata={"query_id": example.query_id, "negative_type": "hard"},
                )
            )
            break
    return pairs
