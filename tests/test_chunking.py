from legal_contract_rag.corpus.chunking import Chunker, ChunkingPolicy
from legal_contract_rag.corpus.loaders import normalize_document


def test_structure_aware_chunking_links_neighbors() -> None:
    document = normalize_document(
        {
            "doc_id": "doc1",
            "title": "Doc 1",
            "raw_text": (
                "ARTICLE 1 DEFINITIONS\n"
                "Section 1.1 Term. The term is three years.\n"
                "ARTICLE 2 PAYMENT\n"
                "Section 2.1 Fees. Payment is due in thirty days."
            ),
            "agreement_type": "supply_agreement",
        }
    )
    chunker = Chunker(ChunkingPolicy(name="structure_aware", chunk_size=16, overlap=4, max_size=64))
    chunks = chunker.chunk_document(document)
    assert chunks
    assert chunks[0].chunk_id == "doc1_chunk_0000"
    assert chunks[0].prev_chunk_id is None
    if len(chunks) > 1:
        assert chunks[0].next_chunk_id == chunks[1].chunk_id


def test_fixed_overlap_generates_multiple_chunks() -> None:
    document = normalize_document({"doc_id": "doc2", "title": "Doc 2", "raw_text": " ".join(["token"] * 100)})
    chunker = Chunker(ChunkingPolicy(name="fixed_overlap", chunk_size=20, overlap=5, max_size=40))
    chunks = chunker.chunk_document(document)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 20 for chunk in chunks)
