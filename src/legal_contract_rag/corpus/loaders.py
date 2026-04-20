"""这个模块负责把原始 json/jsonl/txt 文档统一规范化为 DocumentRecord，供数据处理、训练构建和评测流程使用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from legal_contract_rag.types import DocumentRecord
from legal_contract_rag.utils import clean_text


def normalize_document(payload: dict, source: str | None = None) -> DocumentRecord:
    raw_text = payload.get("raw_text") or payload.get("text") or payload.get("document") or ""
    cleaned_text = payload.get("cleaned_text") or clean_text(raw_text)
    metadata = dict(payload.get("metadata", {}))
    for key in ("filing_date", "agreement_type", "section_headers", "page_spans", "source"):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]
    return DocumentRecord(
        doc_id=str(payload.get("doc_id") or payload.get("id") or payload.get("document_id")),
        source=payload.get("source", source or "unknown"),
        source_url=payload.get("source_url"),
        agreement_type=payload.get("agreement_type"),
        title=payload.get("title", payload.get("doc_id", "untitled")),
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        metadata=metadata,
    )


def load_document_records(paths: Iterable[str]) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for path_str in paths:
        path = Path(path_str)
        source = path.stem
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        documents.append(normalize_document(json.loads(line), source=source))
            continue
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                documents.extend(normalize_document(item, source=source) for item in payload)
            else:
                documents.append(normalize_document(payload, source=source))
            continue
        if path.suffix.lower() == ".txt":
            documents.append(
                normalize_document(
                    {
                        "doc_id": path.stem,
                        "title": path.stem,
                        "raw_text": path.read_text(encoding="utf-8"),
                    },
                    source=source,
                )
            )
            continue
        raise ValueError(f"Unsupported document file type: {path}")
    return documents
