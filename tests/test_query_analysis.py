from legal_contract_rag.query_analysis import QueryAnalyzer


def test_query_analyzer_extracts_filters_and_must_terms() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze("What does Section 7.2 say about governing law in Delaware?", rewrite_mode="strict")

    assert result.intent == "governing_law"
    assert result.filters["clause_type"] == "governing_law"
    assert result.filters["section_path"].lower() == "section 7.2"
    assert result.filters["jurisdiction"] == "Delaware"
    assert "Section 7.2" in result.must_terms
    assert result.rewrite_mode == "strict"


def test_query_analyzer_relaxed_mode_adds_broader_variant() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze("How can the agreement be terminated?", rewrite_mode="relaxed")

    assert result.intent == "termination"
    assert result.rewrite_mode == "relaxed"
    assert any("termination provision" in variant for variant in result.query_variants)
