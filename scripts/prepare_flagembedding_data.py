"""这个脚本用于基于 sample 配置生成 FlagEmbedding 微调数据，调用 legal_contract_rag.cli 中的对应入口，服务于嵌入器和重排器训练流程。"""

import sys

from legal_contract_rag.cli import prepare_flagembedding_data_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs\\flagembedding_sample.yaml"])
    prepare_flagembedding_data_main()
