from legal_contract_rag.corpus.chunking import Chunker, ChunkingPolicy
from legal_contract_rag.corpus.loaders import normalize_document
from legal_contract_rag.pipelines.rag import RAGPipeline
from legal_contract_rag.retrieval.base import TfidfRetriever
from legal_contract_rag.training.builders import build_retriever_triplets
from legal_contract_rag.types import EvalExample


def test_build_retriever_triplets_returns_hard_negative() -> None:
    document = normalize_document(
        {
            "doc_id": "doc1",
            "title": "Sample",
            "raw_text": (
                "ARTICLE 1 TERM\n"
                "Section 1.1 Term. The term is three years.\n"
                "ARTICLE 2 PAYMENT\n"
                "Section 2.1 Fees. Payment is due in thirty days."
            ),
        }
    )
    chunker = Chunker(ChunkingPolicy(name="fixed_overlap", chunk_size=12, overlap=3, max_size=30))
    chunks = chunker.chunk_document(document)
    example = EvalExample(query_id="q1", query="How long is the term?", answer="three years", positive_chunk_ids=[chunks[0].chunk_id])
    pipeline = RAGPipeline(retriever=TfidfRetriever())
    triplets = build_retriever_triplets([example], chunks, pipeline, hard_negative_k=5)
    assert triplets
    assert triplets[0].positive_chunk_id != triplets[0].negative_chunk_id
