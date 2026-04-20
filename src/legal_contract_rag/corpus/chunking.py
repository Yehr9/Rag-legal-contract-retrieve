"""这个模块实现固定分块和结构感知分块逻辑，是数据处理、检索建库和实验对比流程的基础组件。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from legal_contract_rag.types import ChunkRecord, DocumentRecord
from legal_contract_rag.utils import approx_token_count, simple_tokenize


SECTION_PATTERNS = [
    re.compile(r"^(article|section)\s+[a-z0-9ivx\.\-]+", re.IGNORECASE),
    re.compile(r"^\d+(\.\d+){0,3}\s+"),
    re.compile(r"^\([a-z0-9]+\)\s+"),
    re.compile(r"^[A-Z][A-Z\s\-]{4,}$"),
]


@dataclass(slots=True)
class ChunkingPolicy:
    name: str
    chunk_size: int = 512
    overlap: int = 100
    max_size: int = 1536


class Chunker:
    def __init__(self, policy: ChunkingPolicy) -> None:
        self.policy = policy

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        if self.policy.name == "fixed":
            return self._fixed_chunks(document, overlap=0)
        if self.policy.name == "fixed_overlap":
            return self._fixed_chunks(document, overlap=self.policy.overlap)
        return self._structure_aware_chunks(document)

    def _fixed_chunks(self, document: DocumentRecord, overlap: int) -> list[ChunkRecord]:
        tokens = simple_tokenize(document.cleaned_text)
        step = max(1, self.policy.chunk_size - overlap)
        chunks: list[ChunkRecord] = []
        for index, start in enumerate(range(0, len(tokens), step)):
            end = min(start + self.policy.chunk_size, len(tokens))
            content = " ".join(tokens[start:end]).strip()
            if not content:
                continue
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}_chunk_{index:04d}",
                    doc_id=document.doc_id,
                    content=content,
                    token_count=end - start,
                    source_span={"start_token": start, "end_token": end},
                    metadata={
                        "agreement_type": document.agreement_type,
                        "policy": self.policy.name,
                        "clause_type": _infer_clause_type(content, []),
                        "source_dataset": document.metadata.get("source_dataset", document.source),
                        "parent_doc_id": document.doc_id,
                        "summary": _build_summary(content),
                    },
                )
            )
        return _link_chunks(chunks)

    def _structure_aware_chunks(self, document: DocumentRecord) -> list[ChunkRecord]:
        units = self._split_structural_units(document.cleaned_text)
        merged_units = self._merge_structural_units(units)
        chunks: list[ChunkRecord] = []
        buffer: list[tuple[str, list[str]]] = []
        buffer_tokens = 0
        chunk_index = 0

        for unit_text, section_path in merged_units:
            unit_tokens = approx_token_count(unit_text)
            if buffer and buffer_tokens + unit_tokens > self.policy.chunk_size:
                chunks.append(self._make_buffer_chunk(document, buffer, chunk_index))
                chunk_index += 1
                buffer = self._smart_overlap(buffer)
                buffer_tokens = sum(approx_token_count(text) for text, _ in buffer)
            buffer.append((unit_text, section_path))
            buffer_tokens += unit_tokens

        if buffer:
            chunks.append(self._make_buffer_chunk(document, buffer, chunk_index))
        return _link_chunks(chunks)

    def _split_structural_units(self, text: str) -> list[tuple[str, list[str]]]:
        units: list[tuple[str, list[str]]] = []
        current_lines: list[str] = []
        current_path: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self._looks_like_heading(line):
                if current_lines:
                    units.append(("\n".join(current_lines), current_path.copy()))
                    current_lines = []
                current_path = self._update_section_path(current_path, line)
            current_lines.append(line)
        if current_lines:
            units.append(("\n".join(current_lines), current_path.copy()))
        return units

    def _merge_structural_units(self, units: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
        if not units:
            return []
        merged: list[tuple[str, list[str]]] = [units[0]]
        for text, path in units[1:]:
            prev_text, prev_path = merged[-1]
            if self._should_merge(prev_text, text) and approx_token_count(prev_text + " " + text) <= self.policy.max_size:
                merged[-1] = (prev_text + "\n" + text, prev_path or path)
            else:
                merged.append((text, path))
        return merged

    def _should_merge(self, left: str, right: str) -> bool:
        if left.strip().endswith((":", ";", ",")):
            return True
        if re.match(r"^\([a-z0-9]+\)|^\d+\.", right.strip(), re.IGNORECASE):
            return True
        if right.strip().lower().startswith(("provided that", "however", "except", "including", "subject to")):
            return True
        if _unbalanced_parentheses(left):
            return True
        return approx_token_count(left) < 80

    def _smart_overlap(self, buffer: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
        if self.policy.overlap <= 0 or not buffer:
            return []
        overlap_units: list[tuple[str, list[str]]] = []
        token_budget = 0
        for text, path in reversed(buffer):
            overlap_units.insert(0, (text, path))
            token_budget += approx_token_count(text)
            if token_budget >= self.policy.overlap:
                break
        return overlap_units

    def _make_buffer_chunk(
        self,
        document: DocumentRecord,
        buffer: list[tuple[str, list[str]]],
        chunk_index: int,
    ) -> ChunkRecord:
        content = "\n".join(text for text, _ in buffer).strip()
        section_path = next((path for _, path in reversed(buffer) if path), [])
        return ChunkRecord(
            chunk_id=f"{document.doc_id}_chunk_{chunk_index:04d}",
            doc_id=document.doc_id,
            content=content,
            token_count=approx_token_count(content),
            section_path=section_path,
            source_span={"unit_count": len(buffer)},
            metadata={
                "agreement_type": document.agreement_type,
                "policy": self.policy.name,
                "clause_type": _infer_clause_type(content, section_path),
                "source_dataset": document.metadata.get("source_dataset", document.source),
                "parent_doc_id": document.doc_id,
                "summary": _build_summary(content),
            },
        )

    def _looks_like_heading(self, line: str) -> bool:
        return any(pattern.match(line) for pattern in SECTION_PATTERNS)

    def _update_section_path(self, current_path: list[str], line: str) -> list[str]:
        if re.match(r"^\d+(\.\d+)+", line):
            level = line.split()[0].count(".") + 1
            return current_path[: level - 1] + [line]
        if re.match(r"^(article|section)\s+", line, re.IGNORECASE):
            return [line]
        return current_path[:1] + [line]


def _link_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    for index, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
        chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
    return chunks


def _infer_clause_type(content: str, section_path: list[str]) -> str:
    lowered = f"{' '.join(section_path)} {content}".lower()
    if "governing law" in lowered or "laws of" in lowered:
        return "governing_law"
    if "confidential" in lowered:
        return "confidentiality"
    if "terminate" in lowered or "termination" in lowered:
        return "termination"
    if "payment" in lowered or "fee" in lowered or "invoice" in lowered:
        return "payment"
    if "liability" in lowered or "indemn" in lowered:
        return "liability"
    if "term" in lowered or "renewal" in lowered:
        return "term"
    if "definition" in lowered or " means " in lowered:
        return "definition"
    return "misc"


def _build_summary(content: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+|\n+", content.strip())[0]
    return sentence[:160]


def _unbalanced_parentheses(text: str) -> bool:
    return text.count("(") > text.count(")")
