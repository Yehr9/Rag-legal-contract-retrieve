from legal_contract_rag.config import load_config
from legal_contract_rag.evaluation.experiments import run_experiment_suite


def test_experiment_suite_returns_retrieval_upgrade_line() -> None:
    config = load_config("configs/eval_sample.yaml")
    config.retrieval.use_elasticsearch = False

    results = run_experiment_suite(config, suite_names=["retrieval_upgrade_line"])

    assert "retrieval_upgrade_line" in results
    rows = results["retrieval_upgrade_line"]
    assert len(rows) == 3
    assert {row["retrieval_mode"] for row in rows} == {"dense_only", "hybrid_rrf"}
    assert any(row["rerank_mode"] == "base_reranker" for row in rows)
