"""这个模块负责查询分析与轻量改写，在检索主线中为召回阶段提供标准化 query、过滤条件和 must terms。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legal_contract_rag.config import QueryAnalysisConfig
from legal_contract_rag.types import QueryAnalysisResult


STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
}

CLAUSE_KEYWORDS = {
    "governing_law": ("governing law", "law of", "jurisdiction"),
    "termination": ("terminate", "termination", "expired"),
    "payment": ("payment", "fee", "fees", "invoice", "invoices"),
    "confidentiality": ("confidential", "confidentiality", "non-disclosure", "nda"),
    "liability": ("liability", "damages", "indemn", "limitation of liability"),
    "assignment": ("assignment", "assign", "delegation"),
    "term": ("term", "renewal", "effective date"),
    "definition": ("means", "definition", "defined term"),
}

AGREEMENT_KEYWORDS = {
    "nda": ("nda", "non-disclosure", "confidentiality agreement"),
    "lease": ("lease", "landlord", "tenant"),
    "employment": ("employment", "employee", "employer"),
    "purchase": ("purchase agreement", "asset purchase", "stock purchase"),
    "supply": ("supply agreement", "supplier", "customer"),
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _title_terms(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Z][A-Z0-9\-]{1,}\b", text)))


def _money_terms(text: str) -> list[str]:
    pattern = r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|usd|%)\b"
    return sorted(set(match.strip() for match in re.findall(pattern, text, flags=re.IGNORECASE)))


def _clause_references(text: str) -> list[str]:
    pattern = r"\b(?:section|article|clause)\s+[0-9A-Za-z\.\-]+\b"
    return sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))


def _state_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [state.title() for state in STATE_NAMES if state in lowered]


@dataclass(slots=True)
class QueryAnalyzer:
    config: QueryAnalysisConfig = field(default_factory=QueryAnalysisConfig)

    def analyze(self, query: str, *, rewrite_mode: str = "strict") -> QueryAnalysisResult:
        normalized_query = _normalize_whitespace(query)
        intent = self._infer_intent(normalized_query)
        filters = self._extract_filters(normalized_query, intent)
        must_terms = self._extract_must_terms(normalized_query)
        query_variants = self._build_variants(normalized_query, intent, filters, rewrite_mode)
        expected_answer_type = "clause_span"
        return QueryAnalysisResult(
            normalized_query=normalized_query,
            query_variants=query_variants[: self.config.max_variants],
            must_terms=must_terms,
            filters=filters,
            intent=intent,
            expected_answer_type=expected_answer_type,
            rewrite_mode=rewrite_mode,
        )

    def _infer_intent(self, query: str) -> str:
        lowered = query.lower()
        for clause_type, keywords in CLAUSE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return clause_type
        return "misc"

    def _extract_filters(self, query: str, intent: str) -> dict[str, str]:
        lowered = query.lower()
        filters: dict[str, str] = {}
        for agreement_type, keywords in AGREEMENT_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                filters["agreement_type"] = agreement_type
                break
        clause_refs = _clause_references(query)
        if clause_refs:
            filters["section_path"] = clause_refs[0]
        if intent != "misc":
            filters["clause_type"] = intent
        states = _state_terms(query)
        if states:
            filters["jurisdiction"] = states[0]
        return filters

    def _extract_must_terms(self, query: str) -> list[str]:
        must_terms = _title_terms(query) + _money_terms(query) + _clause_references(query) + _state_terms(query)
        unique_terms: list[str] = []
        for term in must_terms:
            if term not in unique_terms:
                unique_terms.append(term)
        return unique_terms

    def _build_variants(
        self,
        query: str,
        intent: str,
        filters: dict[str, str],
        rewrite_mode: str,
    ) -> list[str]:
        variants = [query]
        clause_label = filters.get("clause_type", intent).replace("_", " ")
        if intent != "misc":
            variants.append(f"{clause_label} clause: {query}")
        if filters.get("section_path"):
            variants.append(f"{filters['section_path']} {query}")
        if rewrite_mode == "relaxed":
            relaxed_terms = [query]
            if filters.get("jurisdiction"):
                relaxed_terms.append(f"{intent.replace('_', ' ')} {filters['jurisdiction']}")
            if intent != "misc":
                relaxed_terms.append(f"{intent.replace('_', ' ')} provision")
            variants.extend(relaxed_terms)
        deduped: list[str] = []
        for variant in variants:
            cleaned = _normalize_whitespace(variant)
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped
