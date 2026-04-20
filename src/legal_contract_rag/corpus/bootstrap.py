"""这个模块负责按配置下载公开数据源文件，主要被 bootstrap_public_sources.py 调用，用于数据获取阶段。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
import yaml


@dataclass(slots=True)
class DownloadSpec:
    name: str
    url: str
    output_path: str
    headers: dict[str, str] | None = None


def load_download_specs(path: str | Path) -> list[DownloadSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [DownloadSpec(**spec) for spec in raw.get("sources", [])]


def download_sources(specs: list[DownloadSpec], output_root: str | Path = ".") -> list[Path]:
    output_root = Path(output_root)
    saved_paths: list[Path] = []
    for spec in specs:
        destination = output_root / spec.output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers = {
            "User-Agent": "legal-contract-rag/0.1 your-email@example.com",
            "Accept-Encoding": "gzip, deflate",
        }
        headers.update(spec.headers or {})
        response = requests.get(spec.url, headers=headers, timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
        saved_paths.append(destination)
    return saved_paths
