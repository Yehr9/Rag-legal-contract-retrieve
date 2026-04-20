"""这个脚本用于基于 sample 配置运行示例评测，调用 legal_contract_rag.cli 中的实验入口，服务于检索与重排效果验证。"""

import sys

from legal_contract_rag.cli import run_eval_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs\\eval_sample.yaml"])
    run_eval_main()
