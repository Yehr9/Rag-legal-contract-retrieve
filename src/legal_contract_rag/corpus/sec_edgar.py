"""这个模块封装 SEC EDGAR 相关的数据访问与元数据处理逻辑，服务于真实法律合同语料的扩展与清洗。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from legal_contract_rag.types import DocumentRecord
from legal_contract_rag.utils import clean_text


@dataclass(slots=True)
class EdgarDownloader:
    user_agent: str
    delay_seconds: float = 0.2

    def __post_init__(self) -> None:
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def fetch_json(self, url: str) -> dict:
        response = requests.get(url, headers=self.headers, timeout=60)
        response.raise_for_status()
        time.sleep(self.delay_seconds)
        return response.json()

    def download_text(self, url: str) -> str:
        response = requests.get(url, headers=self.headers, timeout=60)
        response.raise_for_status()
        time.sleep(self.delay_seconds)
        return response.text


def build_submission_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"


def accession_to_archive_path(accession: str) -> str:
    return accession.replace("-", "")


def build_exhibit_url(cik: str, accession: str, filename: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_to_archive_path(accession)}/{filename}"


def make_document_from_sec_payload(
    *,
    doc_id: str,
    title: str,
    agreement_type: str,
    filing_date: str,
    source_url: str,
    text: str,
    metadata: dict | None = None,
) -> DocumentRecord:
    payload = dict(metadata or {})
    payload["filing_date"] = filing_date
    payload["agreement_type"] = agreement_type
    return DocumentRecord(
        doc_id=doc_id,
        source="sec_edgar",
        source_url=source_url,
        agreement_type=agreement_type,
        title=title,
        raw_text=text,
        cleaned_text=clean_text(text),
        metadata=payload,
    )


def save_documents_jsonl(documents: Iterable[DocumentRecord], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
