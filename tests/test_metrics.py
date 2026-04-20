from legal_contract_rag.evaluation.retrieval_metrics import compute_retrieval_metrics
from legal_contract_rag.types import EvalExample, RetrievalHit


def test_retrieval_metrics_capture_hit() -> None:
    examples = [EvalExample(query_id="q1", query="term", answer="three years", positive_chunk_ids=["c1"])]
    predictions = {
        "q1": [
            RetrievalHit(chunk_id="c1", bi_score=0.9, rank_before=1),
            RetrievalHit(chunk_id="c2", bi_score=0.4, rank_before=2),
        ]
    }
    metrics = compute_retrieval_metrics(examples, predictions, recall_k=20, rank_k=5)
    assert metrics["Precision@5"] > 0
    assert metrics["Recall@5"] == 1.0
    assert metrics["citation_hit_rate"] == 1.0


def test_retrieval_metrics_use_distinct_recall_predictions() -> None:
    examples = [EvalExample(query_id="q1", query="term", answer="three years", positive_chunk_ids=["c20"])]
    predictions = {
        "q1": [
            RetrievalHit(chunk_id="c1", bi_score=0.9, rank_before=1),
            RetrievalHit(chunk_id="c2", bi_score=0.4, rank_before=2),
        ]
    }
    recall_predictions = {
        "q1": [
            RetrievalHit(chunk_id=f"c{i}", bi_score=1.0 / i, rank_before=i)
            for i in range(1, 21)
        ]
    }
    metrics = compute_retrieval_metrics(
        examples,
        predictions,
        recall_predictions=recall_predictions,
        recall_k=20,
        rank_k=5,
    )
    assert metrics["Recall@5"] == 0.0
    assert metrics["Recall@20"] == 1.0
