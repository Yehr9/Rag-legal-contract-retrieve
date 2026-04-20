from __future__ import annotations

"""这个脚本用于下载并整理 CUAD、MAUD、ACORD 等公开法律合同数据，产出真实实验所需的语料、评测集和训练种子文件。"""

import argparse
from pathlib import Path

from legal_contract_rag.corpus.public_datasets import prepare_public_legal_data, save_prepared_public_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/real")
    parser.add_argument("--max-cuad-examples", type=int, default=350)
    parser.add_argument("--max-maud-examples", type=int, default=250)
    parser.add_argument("--max-self-authored-examples", type=int, default=120)
    args = parser.parse_args()

    prepared = prepare_public_legal_data(
        Path(args.output_dir),
        max_cuad_examples=args.max_cuad_examples,
        max_maud_examples=args.max_maud_examples,
        max_self_authored_examples=args.max_self_authored_examples,
    )
    save_prepared_public_data(prepared, Path(args.output_dir))
    print(f"primary_documents={len(prepared.primary_documents)}")
    print(f"acord_documents={len(prepared.acord_documents)}")
    print(f"eval_examples={len(prepared.eval_examples)}")
    print(f"acord_reranker_rows={len(prepared.acord_reranker_rows)}")


if __name__ == "__main__":
    main()
