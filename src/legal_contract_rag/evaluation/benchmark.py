"""这个模块负责把评测样本映射到正样本 chunk，用于检索评测、训练样本构建和实验流程中的监督对齐。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from legal_contract_rag.types import ChunkRecord, EvalExample


WHITESPACE_RE = re.compile(r"\s+")


def map_examples_to_positive_chunks(
    examples: Iterable[EvalExample],
    chunks: list[ChunkRecord],
    *,
    min_overlap_recall: float = 0.5,
) -> list[EvalExample]:
    chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)

    mapped_examples: list[EvalExample] = []
    for example in examples:
        if example.positive_chunk_ids:
            mapped_examples.append(example)
            continue

        evidence_texts = _extract_evidence_texts(example)
        positive_chunk_ids = sorted(
            {
                chunk_id
                for doc_id in example.positive_doc_ids
                for chunk_id in _match_evidence_to_doc_chunks(
                    evidence_texts,
                    chunks_by_doc.get(doc_id, []),
                    min_overlap_recall=min_overlap_recall,
                )
            }
        )
        if not positive_chunk_ids:
            continue

        metadata = dict(example.metadata)
        metadata["mapped_positive_chunk_count"] = len(positive_chunk_ids)
        mapped_examples.append(replace(example, positive_chunk_ids=positive_chunk_ids, metadata=metadata))
    return mapped_examples


def _extract_evidence_texts(example: EvalExample) -> list[str]:
    evidence_texts = [
        text.strip()
        for text in example.metadata.get("evidence_texts", [])
        if isinstance(text, str) and _is_actionable_evidence(text)
    ]
    if evidence_texts:
        return evidence_texts
    if example.answer and _is_actionable_evidence(example.answer):
        return [example.answer.strip()]
    return []


def _match_evidence_to_doc_chunks(
    evidence_texts: list[str],
    doc_chunks: list[ChunkRecord],
    *,
    min_overlap_recall: float,
) -> list[str]:
    if not evidence_texts or not doc_chunks:
        return []

    matched_chunk_ids: set[str] = set()
    normalized_chunks = {
        chunk.chunk_id: _normalize_text(chunk.content)
        for chunk in doc_chunks
    }
    for evidence_text in evidence_texts:
        normalized_evidence = _normalize_text(evidence_text)
        if not normalized_evidence:
            continue

        exact_matches = [
            chunk.chunk_id
            for chunk in doc_chunks
            if normalized_evidence in normalized_chunks[chunk.chunk_id]
        ]
        if exact_matches:
            matched_chunk_ids.update(exact_matches)
            continue

        anchor_matches = []
        for chunk in doc_chunks:
            anchor_score = _anchor_overlap_score(normalized_evidence, normalized_chunks[chunk.chunk_id])
            if anchor_score > 0:
                anchor_matches.append((anchor_score, chunk.chunk_id))
        if anchor_matches:
            anchor_matches.sort(reverse=True)
            best_score = anchor_matches[0][0]
            for score, chunk_id in anchor_matches:
                if score == best_score:
                    matched_chunk_ids.add(chunk_id)
            continue

        scored_matches = []
        for chunk in doc_chunks:
            score = _token_overlap_recall(normalized_evidence, normalized_chunks[chunk.chunk_id])
            if score > 0:
                scored_matches.append((score, chunk.chunk_id))
        if not scored_matches:
            continue

        scored_matches.sort(reverse=True)
        best_score = scored_matches[0][0]
        keep_count = 0
        for score, chunk_id in scored_matches:
            if score < min_overlap_recall and keep_count > 0:
                break
            if score < best_score * 0.9 and keep_count > 0:
                break
            matched_chunk_ids.add(chunk_id)
            keep_count += 1
            if keep_count >= 3:
                break
    return list(matched_chunk_ids)


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def _is_actionable_evidence(text: str) -> bool:
    normalized = _normalize_text(text)
    return len(normalized) >= 40 and len(normalized.split()) >= 6


def _token_overlap_recall(evidence_text: str, chunk_text: str) -> float:
    evidence_tokens = evidence_text.split()
    chunk_tokens = chunk_text.split()
    if not evidence_tokens or not chunk_tokens:
        return 0.0
    evidence_counts = Counter(evidence_tokens)
    chunk_counts = Counter(chunk_tokens)
    overlap = sum(min(count, chunk_counts[token]) for token, count in evidence_counts.items())
    return overlap / len(evidence_tokens)


def _anchor_overlap_score(evidence_text: str, chunk_text: str) -> int:
    evidence_tokens = evidence_text.split()
    if len(evidence_tokens) < 8:
        return 0
    anchor_size = min(16, max(8, len(evidence_tokens) // 6))
    step = max(4, anchor_size // 2)
    score = 0
    for start in range(0, len(evidence_tokens) - anchor_size + 1, step):
        anchor = " ".join(evidence_tokens[start:start + anchor_size])
        if anchor in chunk_text:
            score += 1
    return score
