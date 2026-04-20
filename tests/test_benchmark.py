from legal_contract_rag.evaluation.benchmark import map_examples_to_positive_chunks
from legal_contract_rag.types import ChunkRecord, EvalExample


def test_map_examples_to_positive_chunks_exact_match() -> None:
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="doc1", content="Payment terms are net 30 days.", token_count=6),
        ChunkRecord(chunk_id="c2", doc_id="doc1", content="This agreement is governed by Delaware law.", token_count=7),
    ]
    examples = [
        EvalExample(
            query_id="q1",
            query="What is the governing law?",
            answer="This agreement is governed by Delaware law.",
            positive_chunk_ids=[],
            positive_doc_ids=["doc1"],
            metadata={"evidence_texts": ["This agreement is governed by Delaware law."]},
        )
    ]

    mapped = map_examples_to_positive_chunks(examples, chunks)

    assert mapped[0].positive_chunk_ids == ["c2"]


def test_map_examples_to_positive_chunks_falls_back_to_overlap() -> None:
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="doc1", content="The agreement will be governed by the laws", token_count=8),
        ChunkRecord(chunk_id="c2", doc_id="doc1", content="of the State of Delaware including conflicts rules.", token_count=9),
    ]
    examples = [
        EvalExample(
            query_id="q1",
            query="What is the governing law?",
            answer=None,
            positive_chunk_ids=[],
            positive_doc_ids=["doc1"],
            metadata={
                "evidence_texts": [
                    "The agreement will be governed by the laws of the State of Delaware including conflicts rules."
                ]
            },
        )
    ]

    mapped = map_examples_to_positive_chunks(examples, chunks)

    assert set(mapped[0].positive_chunk_ids) == {"c1", "c2"}
