from legal_contract_rag.config import AppConfig
from legal_contract_rag.corpus.loaders import normalize_document
from legal_contract_rag.corpus.chunking import Chunker, ChunkingPolicy
from legal_contract_rag.pipelines.rag import RAGPipeline
from legal_contract_rag.retrieval.base import TfidfRetriever


class LowScoreReranker:
    model_name = "stub-reranker"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.01 for _ in documents]


def test_rag_pipeline_retries_with_relaxed_rewrite_on_low_confidence() -> None:
    config = AppConfig()
    config.retrieval.use_elasticsearch = False
    config.routing.min_rerank_score = 0.5
    config.routing.min_must_term_coverage = 0.0
    config.routing.min_returned_docs = 1
    config.routing.min_top1_margin_over_top5_mean = -1.0
    config.graph.max_retry = 1

    document = normalize_document(
        {
            "doc_id": "doc1",
            "title": "Sample",
            "agreement_type": "supply",
            "raw_text": (
                "ARTICLE 10 GOVERNING LAW\n"
                "Section 10.1 Governing Law. This Agreement is governed by the laws of Delaware."
            ),
        }
    )
    chunker = Chunker(ChunkingPolicy(name="structure_aware_v2", chunk_size=40, overlap=8, max_size=80))
    chunks = chunker.chunk_document(document)
    pipeline = RAGPipeline(config=config, retriever=TfidfRetriever(), reranker=LowScoreReranker())
    pipeline.index(chunks)

    state = pipeline.run("What is the governing law?")

    assert state["retry_triggered"] is True
    assert state["retry_count"] == 1
    assert state["rewrite_mode"] == "relaxed"
    assert state["final_hits"]
