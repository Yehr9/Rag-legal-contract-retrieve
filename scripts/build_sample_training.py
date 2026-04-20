"""这个脚本用于基于 sample 配置生成训练样本，调用 legal_contract_rag.cli 中的训练数据构建入口，服务于实验与微调准备流程。"""

import sys

from legal_contract_rag.cli import build_training_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs\\training_sample.yaml"])
    build_training_main()
