"""这个模块负责基于检索到的上下文生成答案，在主流程中既支持 LLM 生成，也支持无模型时的抽取式 fallback。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from legal_contract_rag.config import GenerationConfig
from legal_contract_rag.types import ChunkRecord


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "must",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "under",
    "what",
    "when",
    "which",
    "who",
    "with",
}


@dataclass(slots=True)
class AnswerGenerator:
    config: GenerationConfig = field(default_factory=GenerationConfig)
    device: str | None = None
    _fallback: bool = field(init=False, default=False)
    _tokenizer: object | None = field(init=False, default=None)
    _model: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.config.model_name:
            self._fallback = True
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.device)
            self._model.eval()
        except Exception:
            self._fallback = True

    def generate(self, query: str, contexts: list[ChunkRecord]) -> str:
        trimmed_contexts = contexts[: self.config.context_chunks]
        if not trimmed_contexts:
            return ""
        if self._fallback:
            return self._extractive_fallback(query, trimmed_contexts)
        prompt = self._build_prompt(query, trimmed_contexts)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        answer = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return answer or self._extractive_fallback(query, trimmed_contexts)

    def _build_prompt(self, query: str, contexts: list[ChunkRecord]) -> str:
        rendered_contexts = []
        for index, chunk in enumerate(contexts, start=1):
            section = " > ".join(chunk.section_path) if chunk.section_path else "Unknown section"
            rendered_contexts.append(
                f"[Context {index}]\n"
                f"Document: {chunk.doc_id}\n"
                f"Section: {section}\n"
                f"Content: {chunk.content}"
            )
        return (
            "You are a legal contract QA assistant.\n"
            "Answer the question using only the supplied contract excerpts.\n"
            "If the answer is not supported by the excerpts, say you cannot find it in the provided context.\n\n"
            f"Question: {query}\n\n"
            "Contract Excerpts:\n"
            f"{chr(10).join(rendered_contexts)}\n\n"
            "Answer:"
        )

    def _extractive_fallback(self, query: str, contexts: list[ChunkRecord]) -> str:
        query_terms = _content_terms(query)
        candidates: list[tuple[float, str]] = []
        for chunk in contexts:
            for sentence in _split_sentences(chunk.content):
                score = _sentence_score(sentence, query_terms)
                if score > 0:
                    candidates.append((score, sentence))
        if not candidates:
            return contexts[0].content.split("\n")[0][:300]
        best_sentences = [sentence for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True)]
        deduped: list[str] = []
        for sentence in best_sentences:
            if sentence not in deduped:
                deduped.append(sentence)
            if len(deduped) >= self.config.fallback_sentence_count:
                break
        return " ".join(deduped)


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _sentence_score(sentence: str, query_terms: set[str]) -> float:
    sentence_terms = _content_terms(sentence)
    if not sentence_terms:
        return 0.0
    overlap = len(query_terms & sentence_terms)
    if overlap == 0:
        return 0.0
    return overlap + overlap / len(sentence_terms)
