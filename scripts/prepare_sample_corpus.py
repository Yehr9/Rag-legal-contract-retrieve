"""这个脚本用于基于 sample 配置构建示例语料与分块结果，调用 legal_contract_rag.cli 中的语料准备入口，服务于本地快速验证检索主线。"""

import sys

from legal_contract_rag.cli import prepare_corpus_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs\\corpus_sample.yaml"])
    prepare_corpus_main()
