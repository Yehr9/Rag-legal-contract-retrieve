from legal_contract_rag.config import GenerationConfig
from legal_contract_rag.answering.generation import AnswerGenerator
from legal_contract_rag.types import ChunkRecord


def test_answer_generator_fallback_selects_relevant_sentence() -> None:
    generator = AnswerGenerator(GenerationConfig(model_name=None, fallback_sentence_count=1))
    contexts = [
        ChunkRecord(
            chunk_id="c1",
            doc_id="doc1",
            content=(
                "Section 1.1 Term. The initial term of this Agreement begins on the Effective Date "
                "and continues for three years unless earlier terminated. "
                "Section 1.2 Renewal. The Agreement renews automatically for one-year periods."
            ),
            token_count=32,
        )
    ]
    answer = generator.generate("How long is the initial term of the agreement?", contexts)
    assert "three years" in answer.lower()


def test_answer_generator_returns_empty_without_context() -> None:
    generator = AnswerGenerator(GenerationConfig(model_name=None))
    assert generator.generate("Any question", []) == ""
