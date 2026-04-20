"""这个模块实现项目的检索主线，串联 query analysis、hybrid retrieval、rerank、重试路由和答案生成。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from legal_contract_rag.answering.generation import AnswerGenerator
from legal_contract_rag.config import AppConfig, QueryAnalysisConfig, RetrievalConfig, RoutingConfig
from legal_contract_rag.query_analysis import QueryAnalyzer
from legal_contract_rag.retrieval.base import BaseReranker, BaseRetriever, TfidfRetriever
from legal_contract_rag.retrieval.bge import BGEReranker
from legal_contract_rag.retrieval.hybrid import HybridRetriever
from legal_contract_rag.types import ChunkRecord, GraphTrace, RerankDecision, RetrievedCandidate, RetrievalHit

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency
    END = "__end__"
    StateGraph = None


class PipelineState(TypedDict, total=False):
    query: str
    normalized_query: str
    query_variants: list[str]
    must_terms: list[str]
    filters: dict[str, str]
    intent: str
    expected_answer_type: str
    rewrite_mode: str
    retrieval_mode: str
    rerank_mode: str
    retry_count: int
    retry_triggered: bool
    retrieved_candidates: list[RetrievedCandidate]
    retrieval_hits: list[RetrievalHit]
    final_hits: list[RetrievalHit]
    rerank_scores: list[float]
    confidence_signals: dict[str, float]
    rerank_decision: RerankDecision
    answer: str
    trace: list[GraphTrace]


@dataclass(slots=True)
class RAGPipeline:
    config: AppConfig = field(default_factory=AppConfig)
    retriever: BaseRetriever | None = None
    reranker: BaseReranker | None = None
    query_analyzer: QueryAnalyzer | None = None
    generator: AnswerGenerator | None = None
    hybrid_retriever: HybridRetriever | None = None
    _chunk_by_id: dict[str, ChunkRecord] = field(init=False, default_factory=dict)
    _app: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.retriever is not None:
            self.config.retrieval.use_elasticsearch = False
        self.query_analyzer = self.query_analyzer or QueryAnalyzer(self.config.query_analysis)
        self.generator = self.generator or AnswerGenerator(self.config.generation)
        dense_retriever = self.retriever
        self.hybrid_retriever = self.hybrid_retriever or HybridRetriever(
            config=self.config.retrieval,
            dense_retriever=dense_retriever,  # type: ignore[arg-type]
            sparse_retriever=TfidfRetriever(),
        )
        self.retriever = self.hybrid_retriever  # compatibility for existing call sites
        self._app = self._compile_graph()

    def index(self, chunks: list[ChunkRecord]) -> None:
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.hybrid_retriever.index(chunks)

    def retrieve_recall_candidates(
        self,
        query: str,
        *,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        state = self.run(query, retrieval_mode=retrieval_mode, rerank_mode="disabled")
        hits = state.get("retrieval_hits", [])
        if top_k is None:
            return hits
        return hits[:top_k]

    def retrieve(
        self,
        query: str,
        *,
        retrieval_mode: str | None = None,
        rerank_mode: str | None = None,
    ) -> list[RetrievalHit]:
        state = self.run(query, retrieval_mode=retrieval_mode, rerank_mode=rerank_mode)
        return state.get("final_hits", [])

    def run(
        self,
        query: str,
        *,
        retrieval_mode: str | None = None,
        rerank_mode: str | None = None,
    ) -> PipelineState:
        initial_state: PipelineState = {
            "query": query,
            "normalized_query": query,
            "query_variants": [query],
            "must_terms": [],
            "filters": {},
            "intent": "misc",
            "expected_answer_type": "clause_span",
            "rewrite_mode": "strict",
            "retrieval_mode": retrieval_mode or self.config.retrieval.retrieval_mode,
            "rerank_mode": rerank_mode or ("base_reranker" if self.reranker else "disabled"),
            "retry_count": 0,
            "retry_triggered": False,
            "retrieved_candidates": [],
            "retrieval_hits": [],
            "final_hits": [],
            "rerank_scores": [],
            "confidence_signals": {},
            "answer": "",
            "trace": [],
        }
        return self._app.invoke(initial_state)

    def build_context(self, hits: list[RetrievalHit]) -> list[ChunkRecord]:
        return [self._chunk_by_id[hit.chunk_id] for hit in hits if hit.chunk_id in self._chunk_by_id]

    def _compile_graph(self) -> Any:
        if StateGraph is None:
            return _FallbackGraph(self)
        builder = StateGraph(PipelineState)
        builder.add_node("query_analysis", self._query_analysis_node)
        builder.add_node("hybrid_retrieval", self._hybrid_retrieval_node)
        builder.add_node("rerank", self._rerank_node)
        builder.add_node("route_after_rerank", self._route_after_rerank_node)
        builder.add_node("generation", self._generation_node)
        builder.set_entry_point("query_analysis")
        builder.add_edge("query_analysis", "hybrid_retrieval")
        builder.add_edge("hybrid_retrieval", "rerank")
        builder.add_edge("rerank", "route_after_rerank")
        builder.add_conditional_edges(
            "route_after_rerank",
            self._resolve_route,
            {
                "retry": "query_analysis",
                "generate": "generation",
            },
        )
        builder.add_edge("generation", END)
        return builder.compile()

    def _query_analysis_node(self, state: PipelineState) -> PipelineState:
        analysis = self.query_analyzer.analyze(state["query"], rewrite_mode=state.get("rewrite_mode", "strict"))
        return {
            "normalized_query": analysis.normalized_query,
            "query_variants": analysis.query_variants,
            "must_terms": analysis.must_terms,
            "filters": analysis.filters,
            "intent": analysis.intent,
            "expected_answer_type": analysis.expected_answer_type,
            "rewrite_mode": analysis.rewrite_mode,
            "trace": self._append_trace(
                state,
                "query_analysis",
                {
                    "rewrite_mode": analysis.rewrite_mode,
                    "normalized_query": analysis.normalized_query,
                    "filters": analysis.filters,
                },
            ),
        }

    def _hybrid_retrieval_node(self, state: PipelineState) -> PipelineState:
        candidates: list[RetrievedCandidate] = []
        seen_chunk_ids: set[str] = set()
        for variant in state.get("query_variants", [state["query"]]):
            batch = self.hybrid_retriever.retrieve(
                variant,
                filters=state.get("filters", {}),
                retrieval_mode=state.get("retrieval_mode", self.config.retrieval.retrieval_mode),
                top_k=self.config.retrieval.top_k_recall,
            )
            for candidate in batch:
                if candidate.chunk_id in seen_chunk_ids:
                    continue
                candidates.append(candidate)
                seen_chunk_ids.add(candidate.chunk_id)
                if len(candidates) >= self.config.retrieval.top_k_recall:
                    break
            if len(candidates) >= self.config.retrieval.top_k_recall:
                break
        retrieval_hits = [candidate.to_hit() for candidate in candidates]
        return {
            "retrieved_candidates": candidates,
            "retrieval_hits": retrieval_hits,
            "trace": self._append_trace(
                state,
                "hybrid_retrieval",
                {
                    "retrieval_mode": state.get("retrieval_mode", self.config.retrieval.retrieval_mode),
                    "candidate_count": len(candidates),
                },
            ),
        }

    def _rerank_node(self, state: PipelineState) -> PipelineState:
        candidates = state.get("retrieved_candidates", [])
        top_k = self.config.retrieval.top_k_rerank
        rerank_mode = state.get("rerank_mode", "disabled")
        if rerank_mode == "disabled":
            final_hits = [candidate.to_hit() for candidate in candidates[:top_k]]
            return {
                "final_hits": final_hits,
                "rerank_scores": [],
                "trace": self._append_trace(
                    state,
                    "rerank",
                    {"rerank_mode": "disabled", "final_hit_count": len(final_hits)},
                ),
            }
        reranker = self._ensure_reranker()
        if reranker is None:
            final_hits = [candidate.to_hit() for candidate in candidates[:top_k]]
            return {
                "final_hits": final_hits,
                "rerank_scores": [],
                "trace": self._append_trace(
                    state,
                    "rerank",
                    {"rerank_mode": "fallback_disabled", "final_hit_count": len(final_hits)},
                ),
            }

        query = state.get("normalized_query", state["query"])
        documents = [candidate.content for candidate in candidates]
        scores = reranker.score(query, documents) if documents else []
        ordered = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_k]
        final_hits: list[RetrievalHit] = []
        for rank, (candidate, score) in enumerate(ordered, start=1):
            hit = candidate.to_hit()
            hit.rerank_score = float(score)
            hit.rank_after = rank
            hit.source = "rerank"
            final_hits.append(hit)
        return {
            "final_hits": final_hits,
            "rerank_scores": [float(score) for _, score in ordered],
            "trace": self._append_trace(
                state,
                "rerank",
                {"rerank_mode": rerank_mode, "final_hit_count": len(final_hits)},
            ),
        }

    def _route_after_rerank_node(self, state: PipelineState) -> PipelineState:
        decision = self._build_rerank_decision(
            final_hits=state.get("final_hits", []),
            must_terms=state.get("must_terms", []),
            rerank_mode=state.get("rerank_mode", "disabled"),
            retry_count=state.get("retry_count", 0),
        )
        updates: PipelineState = {
            "confidence_signals": decision.confidence_signals,
            "rerank_decision": decision,
            "trace": self._append_trace(
                state,
                "route_after_rerank",
                {"action": decision.action, "reason": decision.reason, **decision.confidence_signals},
            ),
        }
        if decision.action == "retry":
            updates["retry_count"] = state.get("retry_count", 0) + 1
            updates["rewrite_mode"] = "relaxed"
            updates["retry_triggered"] = True
        return updates

    def _generation_node(self, state: PipelineState) -> PipelineState:
        answer = self.generator.generate(state["query"], self.build_context(state.get("final_hits", [])))
        return {
            "answer": answer,
            "trace": self._append_trace(
                state,
                "generation",
                {"context_count": len(state.get("final_hits", [])), "answer_length": len(answer)},
            ),
        }

    def _resolve_route(self, state: PipelineState) -> str:
        decision = state.get("rerank_decision")
        if not decision:
            return "generate"
        return decision.action

    def _build_rerank_decision(
        self,
        *,
        final_hits: list[RetrievalHit],
        must_terms: list[str],
        rerank_mode: str,
        retry_count: int,
    ) -> RerankDecision:
        signals = self._confidence_signals(final_hits, must_terms)
        if rerank_mode == "disabled":
            return RerankDecision(action="generate", reason="rerank_disabled", confidence_signals=signals)
        if retry_count >= self.config.graph.max_retry:
            return RerankDecision(action="generate", reason="max_retry_reached", confidence_signals=signals)
        routing = self.config.routing
        should_retry = (
            signals["top1_score"] < routing.min_rerank_score
            or signals["must_term_coverage"] < routing.min_must_term_coverage
            or signals["returned_docs_count"] < routing.min_returned_docs
            or signals["top1_margin_over_top5_mean"] < routing.min_top1_margin_over_top5_mean
        )
        action = "retry" if should_retry else "generate"
        reason = "low_confidence_rerank" if should_retry else "rerank_confident"
        return RerankDecision(action=action, reason=reason, confidence_signals=signals)

    def _confidence_signals(self, final_hits: list[RetrievalHit], must_terms: list[str]) -> dict[str, float]:
        scores = [
            hit.rerank_score
            if hit.rerank_score is not None
            else hit.fusion_score
            if hit.fusion_score is not None
            else hit.bi_score
            for hit in final_hits
        ]
        top1 = float(scores[0]) if scores else 0.0
        top5_mean = float(sum(scores[:5]) / max(1, min(5, len(scores)))) if scores else 0.0
        margin = top1 - top5_mean
        coverage = self._must_term_coverage(final_hits, must_terms)
        return {
            "top1_score": top1,
            "top1_margin_over_top5_mean": margin,
            "must_term_coverage": coverage,
            "returned_docs_count": float(len(final_hits)),
        }

    def _must_term_coverage(self, final_hits: list[RetrievalHit], must_terms: list[str]) -> float:
        if not must_terms:
            return 1.0
        combined_text = " ".join(
            self._chunk_by_id[hit.chunk_id].content for hit in final_hits if hit.chunk_id in self._chunk_by_id
        ).lower()
        matched = sum(1 for term in must_terms if term.lower() in combined_text)
        return matched / len(must_terms)

    def _append_trace(self, state: PipelineState, node_name: str, payload: dict[str, Any]) -> list[GraphTrace]:
        trace = list(state.get("trace", []))
        trace.append(GraphTrace(node_name=node_name, payload=payload))
        return trace

    def _ensure_reranker(self) -> BaseReranker | None:
        if self.reranker is None:
            try:
                self.reranker = BGEReranker(model_name=self.config.retrieval.reranker_model)
            except Exception:
                self.reranker = None
        return self.reranker


class _FallbackGraph:
    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def invoke(self, state: PipelineState) -> PipelineState:
        current_state = dict(state)
        while True:
            current_state.update(self.pipeline._query_analysis_node(current_state))
            current_state.update(self.pipeline._hybrid_retrieval_node(current_state))
            current_state.update(self.pipeline._rerank_node(current_state))
            current_state.update(self.pipeline._route_after_rerank_node(current_state))
            if self.pipeline._resolve_route(current_state) != "retry":
                current_state.update(self.pipeline._generation_node(current_state))
                return current_state
