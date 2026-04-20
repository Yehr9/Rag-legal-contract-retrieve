"""这个模块负责下载并整理公开法律合同数据集，生成真实语料、评测样本和训练种子文件，供数据处理与实验流程使用。"""

from __future__ import annotations

import csv
import json
import random
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, list_repo_files

from legal_contract_rag.types import DocumentRecord, EvalExample
from legal_contract_rag.utils import clean_text, read_jsonl, write_json, write_jsonl


CUAD_REPO = "theatticusproject/cuad"
MAUD_REPO = "theatticusproject/maud"
ACORD_REPO = "theatticusproject/acord"


@dataclass(slots=True)
class PreparedPublicData:
    primary_documents: list[DocumentRecord]
    acord_documents: list[DocumentRecord]
    eval_examples: list[EvalExample]
    acord_reranker_rows: list[dict]
    manifest: dict


def prepare_public_legal_data(
    output_dir: str | Path,
    *,
    max_cuad_examples: int = 350,
    max_maud_examples: int = 250,
    max_self_authored_examples: int = 120,
    random_seed: int = 42,
) -> PreparedPublicData:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed)

    cuad_json_path = Path(hf_hub_download(repo_id=CUAD_REPO, filename="CUAD_v1/CUAD_v1.json", repo_type="dataset"))
    maud_files = _download_maud_files()
    acord_root = _extract_acord_dataset(output_dir / "raw" / "acord")

    cuad_documents, cuad_examples = _build_cuad_records(cuad_json_path, max_examples=max_cuad_examples, rng=rng)
    maud_documents, maud_examples = _build_maud_records(maud_files, max_examples=max_maud_examples, rng=rng)
    self_authored = _build_self_authored_examples(
        [*cuad_documents, *maud_documents],
        max_examples=max_self_authored_examples,
        rng=rng,
    )
    acord_documents, acord_rows = _build_acord_records(acord_root)

    primary_documents = _dedupe_documents([*cuad_documents, *maud_documents])
    eval_examples = [*cuad_examples, *maud_examples, *self_authored]
    manifest = {
        "sources": [
            {
                "name": "CUAD",
                "repo": CUAD_REPO,
                "downloaded_file": str(cuad_json_path),
                "url": "https://huggingface.co/datasets/theatticusproject/cuad",
            },
            {
                "name": "MAUD",
                "repo": MAUD_REPO,
                "downloaded_files": [str(path) for path in maud_files],
                "url": "https://huggingface.co/datasets/theatticusproject/maud",
            },
            {
                "name": "ACORD",
                "repo": ACORD_REPO,
                "extracted_root": str(acord_root),
                "url": "https://huggingface.co/datasets/theatticusproject/acord",
            },
        ],
        "stats": {
            "primary_document_count": len(primary_documents),
            "acord_clause_count": len(acord_documents),
            "eval_example_count": len(eval_examples),
            "acord_reranker_seed_count": len(acord_rows),
        },
    }
    return PreparedPublicData(
        primary_documents=primary_documents,
        acord_documents=acord_documents,
        eval_examples=eval_examples,
        acord_reranker_rows=acord_rows,
        manifest=manifest,
    )


def save_prepared_public_data(prepared: PreparedPublicData, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "contracts_corpus.jsonl", [row.to_dict() for row in prepared.primary_documents])
    write_jsonl(output_dir / "acord_clause_corpus.jsonl", [row.to_dict() for row in prepared.acord_documents])
    write_jsonl(output_dir / "retrieval_eval.jsonl", [row.to_dict() for row in prepared.eval_examples])
    write_jsonl(output_dir / "acord_reranker_seed.jsonl", prepared.acord_reranker_rows)
    write_json(output_dir / "source_manifest.json", prepared.manifest)


def _build_cuad_records(cuad_json_path: Path, *, max_examples: int, rng: random.Random) -> tuple[list[DocumentRecord], list[EvalExample]]:
    payload = json.loads(cuad_json_path.read_text(encoding="utf-8"))
    documents: list[DocumentRecord] = []
    examples: list[EvalExample] = []
    for item in payload["data"]:
        paragraph = item["paragraphs"][0]
        context = clean_text(paragraph["context"])
        doc_id = f"cuad::{item['title']}"
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                source="cuad",
                source_url="https://huggingface.co/datasets/theatticusproject/cuad",
                agreement_type=_infer_cuad_agreement_type(item["title"]),
                title=item["title"],
                raw_text=paragraph["context"],
                cleaned_text=context,
                metadata={"source_dataset": "CUAD", "title": item["title"]},
            )
        )
        valid_qas = [
            qa for qa in paragraph["qas"]
            if qa.get("answers") and qa["answers"][0].get("text", "").strip()
        ]
        rng.shuffle(valid_qas)
        for qa in valid_qas[:2]:
            evidence_texts = [answer["text"] for answer in qa["answers"] if answer.get("text", "").strip()]
            examples.append(
                EvalExample(
                    query_id=f"cuad::{qa['id']}",
                    query=qa["question"],
                    answer=evidence_texts[0] if evidence_texts else None,
                    positive_chunk_ids=[],
                    positive_doc_ids=[doc_id],
                    metadata={
                        "source": "cuad",
                        "evidence_texts": evidence_texts,
                        "clause_label": qa["id"].split("__")[-1],
                    },
                )
            )
    rng.shuffle(examples)
    return documents, examples[:max_examples]


def _build_maud_records(maud_files: list[Path], *, max_examples: int, rng: random.Random) -> tuple[list[DocumentRecord], list[EvalExample]]:
    contract_files = [
        path for path in maud_files
        if "MAUD_v1/contracts/" in path.as_posix().replace("\\", "/")
    ]
    csv_files = [path for path in maud_files if path.suffix.lower() == ".csv"]
    documents: list[DocumentRecord] = []
    for contract_file in contract_files:
        contract_name = contract_file.stem
        raw_text = contract_file.read_text(encoding="utf-8")
        documents.append(
            DocumentRecord(
                doc_id=f"maud::{contract_name}",
                source="maud",
                source_url="https://huggingface.co/datasets/theatticusproject/maud",
                agreement_type="merger_agreement",
                title=contract_name,
                raw_text=raw_text,
                cleaned_text=clean_text(raw_text),
                metadata={"source_dataset": "MAUD", "contract_name": contract_name},
            )
        )
    examples: list[EvalExample] = []
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("text", "").strip():
                    continue
                contract_name = row["contract_name"]
                query = _humanize_maud_query(row)
                examples.append(
                    EvalExample(
                        query_id=f"maud::{csv_file.stem}::{row['id']}",
                        query=query,
                        answer=row.get("answer") or None,
                        positive_chunk_ids=[],
                        positive_doc_ids=[f"maud::{contract_name}"],
                        metadata={
                            "source": "maud",
                            "evidence_texts": [clean_text(row["text"])],
                            "category": row.get("category"),
                            "text_type": row.get("text_type"),
                            "split": csv_file.stem,
                            "label": row.get("label"),
                        },
                    )
                )
    rng.shuffle(examples)
    return documents, examples[:max_examples]


def _build_acord_records(acord_root: Path) -> tuple[list[DocumentRecord], list[dict]]:
    corpus_rows = read_jsonl(acord_root / "corpus.jsonl")
    query_rows = {row["_id"]: row for row in read_jsonl(acord_root / "queries.jsonl")}
    qrel_files = [
        acord_root / "qrels" / "train.tsv",
        acord_root / "qrels" / "valid.tsv",
        acord_root / "qrels" / "test.tsv",
    ]
    qrels: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for qrel_file in qrel_files:
        with qrel_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                qrels[row["query-id"]].append((row["corpus-id"], float(row["score"])))

    documents = [
        DocumentRecord(
            doc_id=f"acord::{row['_id']}",
            source="acord",
            source_url="https://huggingface.co/datasets/theatticusproject/acord",
            agreement_type="contract_clause",
            title=row["_id"],
            raw_text=row["text"],
            cleaned_text=clean_text(row["text"]),
            metadata={"source_dataset": "ACORD"},
        )
        for row in corpus_rows
    ]
    corpus_by_id = {row["_id"]: clean_text(row["text"]) for row in corpus_rows}
    reranker_rows: list[dict] = []
    for query_id, labels in qrels.items():
        if query_id not in query_rows:
            continue
        sorted_labels = sorted(labels, key=lambda item: item[1], reverse=True)
        positives = [corpus_by_id[doc_id] for doc_id, score in sorted_labels if score >= 2 and doc_id in corpus_by_id][:3]
        negatives = [corpus_by_id[doc_id] for doc_id, score in sorted_labels if score < 2 and doc_id in corpus_by_id][:3]
        if not positives or not negatives:
            continue
        reranker_rows.append(
            {
                "query": query_rows[query_id]["text"],
                "pos": positives,
                "neg": negatives,
                "metadata": {"source": "acord", "query_id": query_id},
            }
        )
    return documents, reranker_rows


def _build_self_authored_examples(
    documents: Iterable[DocumentRecord],
    *,
    max_examples: int,
    rng: random.Random,
) -> list[EvalExample]:
    rules = [
        ("governing law", re.compile(r"laws? of [A-Z][A-Za-z ]+|governed by", re.IGNORECASE), "What does the contract say about governing law?"),
        ("termination", re.compile(r"terminate|termination", re.IGNORECASE), "What termination rights does the contract provide?"),
        ("payment", re.compile(r"invoice|pay|payment|fees?", re.IGNORECASE), "What are the payment terms in the contract?"),
        ("confidentiality", re.compile(r"confidential|non[- ]disclosure", re.IGNORECASE), "What confidentiality obligations does the contract impose?"),
        ("assignment", re.compile(r"assign(?:ment)?", re.IGNORECASE), "What does the contract say about assignment?"),
        ("notice", re.compile(r"notices?|written notice", re.IGNORECASE), "What notice requirements does the contract include?"),
    ]
    examples: list[EvalExample] = []
    for document in documents:
        sentences = re.split(r"(?<=[\.\?!])\s+", document.cleaned_text)
        used_tags: set[str] = set()
        for sentence in sentences:
            stripped = sentence.strip()
            if len(stripped) < 40:
                continue
            for tag, pattern, query in rules:
                if tag in used_tags:
                    continue
                if pattern.search(stripped):
                    used_tags.add(tag)
                    examples.append(
                        EvalExample(
                            query_id=f"self::{document.doc_id}::{tag}",
                            query=query,
                            answer=None,
                            positive_chunk_ids=[],
                            positive_doc_ids=[document.doc_id],
                            metadata={"source": "self_authored", "evidence_texts": [stripped], "tag": tag},
                        )
                    )
                    break
    rng.shuffle(examples)
    return examples[:max_examples]


def _extract_acord_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(hf_hub_download(repo_id=ACORD_REPO, filename="ACORD Dataset & ReadMe.zip", repo_type="dataset"))
    marker = output_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
        marker.write_text("ok", encoding="utf-8")
    root = output_dir / "ACORD Dataset & ReadMe (external)"
    return root


def _download_maud_files() -> list[Path]:
    repo_files = list_repo_files(MAUD_REPO, repo_type="dataset")
    wanted = [file_name for file_name in repo_files if file_name.endswith(".csv") or file_name.startswith("MAUD_v1/contracts/")]
    return [Path(hf_hub_download(repo_id=MAUD_REPO, filename=file_name, repo_type="dataset")) for file_name in wanted]


def _infer_cuad_agreement_type(title: str) -> str:
    if "_" in title:
        tail = title.split("_")[-1]
        return tail.lower().replace(" ", "_")
    return "contract"


def _humanize_maud_query(row: dict[str, str]) -> str:
    base = row.get("text_type") or row.get("question") or "contract provision"
    base = base.replace("-", " ").replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base)
    subquestion = (row.get("subquestion") or "").strip()
    if subquestion and subquestion != "<NONE>":
        return f"What does the merger agreement say about {base}: {subquestion}?"
    return f"What does the merger agreement say about {base}?"


def _dedupe_documents(documents: Iterable[DocumentRecord]) -> list[DocumentRecord]:
    seen: dict[str, DocumentRecord] = {}
    for document in documents:
        seen[document.doc_id] = document
    return list(seen.values())
