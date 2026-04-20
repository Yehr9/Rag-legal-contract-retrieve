# Legal Contract RAG

English legal-domain RAG project focused on commercial contracts. The repository implements:

- a LangGraph state-machine pipeline with query analysis, retrieval, rerank routing, and generation
- corpus normalization for CUAD, MAUD, ACORD, and SEC EDGAR style inputs
- structure-aware chunking with overlap and ablation-friendly baselines
- Elasticsearch-first hybrid retrieval with dense-only and hybrid RRF modes
- BGE bi-encoder retrieval and BGE reranker re-ranking
- training-data generation for FlagEmbedding domain fine-tuning
- retrieval and end-to-end evaluation
- experiment runners that save resume-friendly comparison tables

## Environment

```powershell
conda activate legal-rag
pip install -e .[dev]
```

If `conda` is not on `PATH`, use:

```powershell
& C:\Users\yhr\miniconda3\Scripts\conda.exe activate legal-rag
```

## Repository Layout

```text
configs/
data/sample/
docs/
scripts/
src/legal_contract_rag/
tests/
```

## Quick Start

```powershell
python scripts\prepare_sample_corpus.py --config configs\corpus_sample.yaml
python scripts\build_sample_training.py --config configs\training_sample.yaml
python scripts\run_sample_eval.py --config configs\eval_sample.yaml
```

Outputs are written under `artifacts/`.

The default architecture is:

```text
query_analysis -> hybrid_retrieval -> rerank -> route_after_rerank -> generation
```

When rerank confidence is low, the graph retries once with a relaxed query rewrite before generating an answer.

For the real legal-contract benchmark built from CUAD, MAUD, and ACORD:

```powershell
python scripts\prepare_public_legal_data.py --output-dir data\real
legal-rag-run-eval --config configs\eval_real.yaml
```

This writes the formal retrieval experiment tables under `artifacts\real\eval\`.

The main experiment suites are:

- `chunking_ablation`
- `retrieval_upgrade_line`
- `rerank_finetune_ablation`

The sample evaluation config leaves `generation.model_name` as `null`, which enables an extractive fallback answerer that picks the most relevant retrieved sentence. To use a local Hugging Face causal LLM instead, set `generation.model_name` to a text-generation checkpoint available in your environment.

Answer generation and answer-level metrics are isolated behind `evaluation.include_answer_eval`. The sample config keeps that flag disabled so the default workflow stays focused on chunking, retrieval, reranking, and fine-tuning experiments.

To bootstrap public source files into `data/raw/`:

```powershell
python scripts\bootstrap_public_sources.py --config configs\public_sources.yaml
```

## Production Workflow

Recommended corpus path:

1. Seed with CUAD and MAUD.
2. Expand with SEC EDGAR contract exhibits.
3. Build supervised training examples from ACORD and evidence-linked clauses.

The repository includes:

- `scripts/bootstrap_public_sources.py` for downloading starter public-source files
- `src/legal_contract_rag/corpus/sec_edgar.py` for SEC submission and exhibit URL handling
- `legal-rag-prepare-corpus` for normalizing local raw documents into corpus and chunk artifacts
- `legal-rag-prepare-flagembedding`, `legal-rag-train-embedder`, and `legal-rag-train-reranker` for FlagEmbedding fine-tuning

Default production models:

- bi-encoder: `BAAI/bge-m3`
- reranker: `BAAI/bge-reranker-v2-m3`

Retrieval defaults to `hybrid_rrf`, but the code falls back to an in-memory dense/sparse implementation when Elasticsearch or LangChain ES integrations are unavailable. The pipeline also falls back to a lightweight TF-IDF retriever and lexical reranker when transformer weights are unavailable so local tests stay runnable.

## FlagEmbedding Fine-Tuning

Prepare training data in FlagEmbedding format:

```powershell
python scripts\prepare_flagembedding_data.py --config configs\flagembedding_sample.yaml
```

This writes:

- `artifacts/flagembedding/embedder_train.jsonl`
- `artifacts/flagembedding/reranker_train.jsonl`

Each row follows the format expected by FlagEmbedding fine-tuning:

- `{"query": str, "pos": [str], "neg": [str]}`

Run embedder fine-tuning:

```powershell
legal-rag-train-embedder --train-data artifacts\flagembedding\embedder_train.jsonl --output-dir artifacts\models\embedder --base-model BAAI/bge-m3 --dry-run
```

Run reranker fine-tuning:

```powershell
legal-rag-train-reranker --train-data artifacts\flagembedding\reranker_train.jsonl --output-dir artifacts\models\reranker --base-model BAAI/bge-reranker-v2-m3 --dry-run
```

Remove `--dry-run` to launch the actual training command. On Windows, `FlagEmbedding[finetune]` may require extra setup because `deepspeed` often fails to build; the repository therefore uses the base `FlagEmbedding` package plus generated official module commands.

## Real Data Sources

- CUAD: <https://github.com/TheAtticusProject/cuad>
- MAUD: <https://github.com/TheAtticusProject/maud>
- ACORD: <https://huggingface.co/datasets/theatticusproject/acord>
- SEC EDGAR APIs: <https://www.sec.gov/edgar/sec-api-documentation>
- SEC developer resources: <https://www.sec.gov/about/developer-resources>
