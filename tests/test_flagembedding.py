from legal_contract_rag.training.flagembedding import (
    FlagEmbeddingFinetuneConfig,
    build_embedder_command,
    build_embedder_train_rows,
)
from legal_contract_rag.types import ChunkRecord, EvalExample


def test_build_embedder_train_rows_uses_pos_and_neg_lists() -> None:
    example = EvalExample(query_id="q1", query="term length", answer=None, positive_chunk_ids=["c1"])
    chunk_by_id = {
        "c1": ChunkRecord(chunk_id="c1", doc_id="d1", content="The term is three years.", token_count=5),
        "c2": ChunkRecord(chunk_id="c2", doc_id="d1", content="Invoices are due in thirty days.", token_count=6),
    }
    rows = build_embedder_train_rows([example], chunk_by_id, {"q1": ["c1", "c2"]})
    assert rows == [{"query": "term length", "pos": ["The term is three years."], "neg": ["Invoices are due in thirty days."]}]


def test_build_embedder_command_targets_flagembedding_module() -> None:
    command = build_embedder_command(
        FlagEmbeddingFinetuneConfig(
            train_data_path="artifacts/flagembedding/embedder_train.jsonl",
            output_dir="artifacts/models/embedder",
            base_model="BAAI/bge-m3",
        )
    )
    assert command[1:10] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--standalone",
        "--rdzv-conf",
        "use_libuv=0",
        "-m",
        "FlagEmbedding.finetune.embedder.encoder_only.base",
    ]
    assert "--do_train" in command
