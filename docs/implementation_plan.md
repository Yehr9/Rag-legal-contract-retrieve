# English Legal Contract RAG With Chunking, Rerank, and Domain Fine-Tuning

## Summary
- Build the project as a narrow legal-domain RAG system for commercial contract retrieval and QA, not broad legal RAG.
- Use a corpus centered on U.S. public commercial contracts from SEC EDGAR exhibits, with CUAD and MAUD as the initial high-quality seed sets for corpus bootstrapping, supervised evaluation, and fine-tuning.
- Follow the pipeline implied by the PDFs: structure-aware chunking -> BGE bi-encoder recall -> BGE reranker re-rank -> LLM answer generation, then compare before and after rerank and before and after domain fine-tuning.
- Use FlagEmbedding as the fine-tuning framework for both the embedder and reranker stages.

## Key Changes
- Corpus strategy
  - Phase 1 seed corpus: start from CUAD full contract texts and MAUD merger agreements so the project can run quickly with expert-labeled legal documents.
  - Phase 2 expansion corpus: crawl or download SEC EDGAR filing exhibits, prioritizing material contracts and merger-related agreements, then normalize to plain text plus metadata.
  - Store each document with metadata including `doc_id`, filing source, agreement type, filing date, section headers, and page or paragraph provenance for citation-style retrieval.
- Chunking strategy
  - Use the `chunk.pdf` direction instead of naive fixed windows: begin from a token target around 512, keep 100-token smart overlap, and split by legal structure such as article or section numbering, headings, clauses, and enumerations.
  - Add chunk metadata like `chunk_id`, `prev_chunk_id`, `next_chunk_id`, `section_path`, and agreement type so retrieval and evaluation can inspect boundary quality.
  - Run an ablation between fixed chunks without overlap, fixed chunks with overlap, and structure-aware chunks with smart merge and overlap.
- Retrieval and rerank
  - Base retriever: `BAAI/bge-m3` as the bi-encoder for embeddings and Top-K recall.
  - Reranker: `BAAI/bge-reranker-v2-m3` as the cross-encoder reranker, matching `rerank.pdf`.
  - Standard inference path: retrieve Top-20 with the bi-encoder, rerank to Top-5, pass only Top-5 to the generator.
  - Keep a no-rerank baseline so the resume project shows a clean before or after comparison on retrieval and end-to-end QA.
- Domain fine-tuning
  - Fine-tune the bi-encoder on legal contract retrieval pairs or triplets built from ACORD, CUAD, and held-out SEC contract clauses.
  - Fine-tune the reranker on `(query, positive, negative)` data using positives from expert labels and hard negatives from the Top-50 recalled candidates.
  - Use ACORD relevance labels and CUAD or MAUD evidence spans to construct supervised positives; use in-batch negatives plus hard negatives from the baseline retriever.
  - Final comparison matrix: base retriever only, base retriever plus reranker, fine-tuned bi-encoder only, and fine-tuned bi-encoder plus fine-tuned reranker.
- Evaluation system
  - Retrieval metrics: `Recall@20` for first-stage recall, then `Precision@5`, `Recall@5`, and `NDCG@5` after rerank.
  - End-to-end metrics: answer `EM/F1` where labels exist, plus citation or evidence hit rate measuring whether the gold clause or span appears in retrieved chunks.
  - Produce the exact ablation tables needed for a resume: chunking ablation, rerank ablation, and fine-tune ablation.
  - Hold out a legal QA dev or test split by document, not by chunk, to avoid leakage.

## Public APIs / Interfaces / Types
- `document = {doc_id, source, source_url, agreement_type, title, raw_text, cleaned_text, metadata}`
- `chunk = {chunk_id, doc_id, content, token_count, section_path, prev_chunk_id, next_chunk_id, source_span}`
- `retrieval_hit = {chunk_id, bi_score, rerank_score, rank_before, rank_after}`
- `triplet = {query, positive_chunk_id, negative_chunk_id}`
- `rerank_pair = {query, chunk_text, label}`

## Test Plan
- Corpus ingestion tests: raw SEC, CUAD, and MAUD documents can be normalized into stable document records with preserved metadata.
- Chunking tests: section-based splitting does not lose headings, clause numbering, or adjacency links.
- Retrieval tests: the correct gold clause appears in Top-20 for the benchmark split.
- Rerank tests: reranked Top-5 improves `Precision@5`, `Recall@5`, or `NDCG@5` over the retriever-only baseline.
- Fine-tuning tests: fine-tuned bi-encoder improves first-stage recall; fine-tuned reranker improves Top-5 ranking quality.
- End-to-end tests: answer quality and evidence grounding improve after rerank and after legal-domain fine-tuning.

## Assumptions And Defaults
- The project is English-only and focused on commercial contracts, not case law or statutes.
- The recommended knowledge-base path is CUAD plus MAUD as the seed expert corpus, then SEC EDGAR contract exhibits as the scalable real-world corpus.
- Default models: `BAAI/bge-m3` for embeddings and `BAAI/bge-reranker-v2-m3` for rerank, with FlagEmbedding used for fine-tuning both stages.
- Default retrieval settings: Top-20 recall -> Top-5 rerank -> generator.
- Default chunk settings: 512-token target + 100-token smart overlap + structure-aware splitting.
