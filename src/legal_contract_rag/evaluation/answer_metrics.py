"""这个模块实现答案级评测指标计算，供开启答案评估时统计 EM 和 F1。"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from legal_contract_rag.types import EvalExample


def compute_answer_metrics(
    examples: Iterable[EvalExample],
    predictions: dict[str, str],
) -> dict[str, float]:
    exact_match_scores: list[float] = []
    f1_scores: list[float] = []
    for example in examples:
        if not example.answer:
            continue
        gold = _normalize_answer(example.answer)
        pred = _normalize_answer(predictions.get(example.query_id, ""))
        exact_match_scores.append(1.0 if pred == gold else 0.0)
        f1_scores.append(_f1(pred, gold))
    return {"EM": _mean(exact_match_scores), "F1": _mean(f1_scores)}


def _normalize_answer(text: str) -> str:
    return " ".join(text.lower().split())


def _f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
