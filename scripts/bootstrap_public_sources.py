"""这个脚本用于按配置下载公开数据源的原始文件，服务于数据处理流程，供后续语料准备与真实数据构建使用。"""

import argparse

from legal_contract_rag.corpus.bootstrap import download_sources, load_download_specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs\\public_sources.yaml")
    args = parser.parse_args()
    specs = load_download_specs(args.config)
    paths = download_sources(specs)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
