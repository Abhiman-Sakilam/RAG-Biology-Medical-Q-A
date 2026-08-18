# RAG Methodology Upgrade — Design Spec

Date: 2026-08-19
Status: Approved for planning
Target repo: `RAG-Biology-Medical-Q-A/` (this repo; the sibling copy one level up is a scratch mirror and is out of scope)

## 1. Motivation

The current pipeline (`src/pipeline/rag.py`) is: BM25 search → concatenate passages → single LLM call → light text cleanup. This works, but it is missing every technique that separates a 2023-era RAG demo from a 2025–2026-current one:

- No dense/semantic retrieval — BM25 alone misses passages that are conceptually relevant but lexically different from the question (a real problem in biomedical QA, where questions are phrased colloquially and passages are phrased in technical/abstract style).
- No reranking of retrieved candidates.
- No chunking strategy — passages are used exactly as stored, regardless of length.
- No query understanding — the raw question string is used verbatim for retrieval.
- No citation enforcement — the LLM is shown passages but never required to attribute claims to them.
- No groundedness/hallucination check on the generated answer.
- No evaluation harness — despite the repo already containing a gold-labeled eval set (`data/json/question-answer-passages/*.json`, 4,012 QA pairs with `relevant_passage_ids`) that has never been used.

This spec upgrades the methodology in two phases: **Phase 1** fixes retrieval quality and adds a real eval harness so every later change can be measured. **Phase 2** fixes generation quality (grounding, citations, query understanding) on top of the Phase 1 baseline.

## 2. Current State (for reference)

- Corpus: 40,221 passages (`data/json/text-corpus/`), word count median 204, p90 294, p99 475, max 4,215 (long tail is a small minority).
- Eval set: 4,012 QA pairs (`data/json/question-answer-passages/`) with `question`, `answer`, `relevant_passage_ids` — usable directly for Recall/MRR/nDCG scoring.
- Retrieval: `src/retrieval/sparse.py`, BM25Okapi over whitespace/regex tokenized passages, no persistence (rebuilt in memory on first request via a lazy global in `src/pipeline/rag.py::_get_index`).
- Generation: `src/generation/llm.py`, single OpenAI-compatible chat call against Groq (default) or OpenAI.
- Config: `config.toml`, `[retrieval] top_k`; `[llm] model, max_tokens, temperature`.
- `artifacts/` exists, is git-ignored, is mounted into the Docker container, and is currently unused by any code — designated for exactly this kind of persisted index/cache data.

## 3. Phase 1 — Retrieval Quality

### 3.1 Hybrid retrieval (BM25 + dense, fused with RRF)

New module `src/retrieval/dense.py`:
- Embeddings via OpenAI `text-embedding-3-small` (reuses the existing `openai` client/key already in the project — no new provider for this stage).
- Index: FAISS `IndexFlatIP` over L2-normalized vectors (cosine similarity). At 40k passages this is a few tens of MB, fully in-memory, no vector DB server needed.
- `embed_texts(texts) -> np.ndarray`, `build_dense_index(corpus) -> faiss.Index`, `dense_search(query, index, k) -> List[Tuple[id, text, score]]` — same return shape as `src/retrieval/sparse.py::search` so both are drop-in interchangeable for fusion.

New script `scripts/build_embeddings.py`: computes embeddings once, persists the FAISS index + a parallel id-mapping array to `artifacts/dense_index/`, keyed by a hash of the corpus file (mtime+size or content hash) so a corpus change forces a rebuild. `src/pipeline/rag.py` loads this persisted index at startup instead of recomputing embeddings per process — critical since re-embedding 40k passages on every server restart would be wasteful and slow.

New module `src/retrieval/fusion.py`: Reciprocal Rank Fusion — `rrf_fuse(sparse_results, dense_results, k=60) -> List[(id, text, fused_score)]`, `score = Σ 1/(k + rank)` over whichever result lists contain that id. RRF is used because BM25 scores and cosine similarities are not comparable on the same scale; naively summing them is wrong.

### 3.2 Reranking (Voyage AI `rerank-2`)

New module `src/retrieval/rerank.py`: takes the fused top ~20-30 candidates, calls Voyage's `rerank-2` endpoint (`VOYAGE_API_KEY` in `setup/.env`), returns the reordered top `rerank_top_n`. Chosen over LLM-based reranking because it is non-autoregressive (faster, ~100-400ms for the whole batch vs. autoregressive generation), cheaper per query, and more accurate for pure relevance ranking (purpose-trained on ranking objectives, unlike a general chat LLM prompted ad hoc). Also chosen over Cohere `rerank-v3.5` for its long-document support (up to 16K tokens), relevant given the corpus's long-tail passages.

### 3.3 Pipeline integration

`src/pipeline/rag.py::rag_query()` becomes:

```
question
  → BM25 search (sparse_top_n)          ┐
  → dense search (dense_top_n)          ┘→ RRF fuse → Voyage rerank → top_k
  → format_passages_as_context()
  → build_rag_prompt()
  → generate()
  → parse_answer()
```

Every stage after BM25 is config-gated:
- `mode = "bm25"` (default, backward-compatible) skips dense search, fusion, and reranking entirely — output is byte-identical to today.
- `mode = "hybrid"` enables dense search + RRF.
- `rerank = true/false` independently toggles the Voyage stage on top of either mode.

`config.toml` additions:

```toml
[retrieval]
mode = "bm25"        # "bm25" | "hybrid"
top_k = 5
sparse_top_n = 20
dense_top_n = 20
fusion_k = 60
rerank = false
rerank_top_n = 5
```

New env var: `VOYAGE_API_KEY` (optional — only required when `rerank = true`).
New dependencies: `faiss-cpu`, `numpy`, `voyageai` (or a direct HTTP call via `httpx` if the SDK adds unwanted weight — decide during implementation).

### 3.4 Evaluation harness

New script `scripts/evaluate_retrieval.py`:
- Loads the test split's QA pairs (`data/json/question-answer-passages/test-00000-of-00001.json`) and their gold `relevant_passage_ids`.
- For each of {bm25, hybrid, hybrid+rerank}, runs retrieval for every question, computes Recall@k, MRR@k, nDCG@k (k matching `top_k`/`rerank_top_n`).
- Writes a comparison table (JSON + printed summary) to `artifacts/eval/retrieval_<timestamp>.json`.
- This is the acceptance gate for Phase 1: hybrid+rerank must show a measurable Recall/MRR improvement over the BM25 baseline on the same test set before Phase 1 is considered done.

### 3.5 Targeted fix to existing code

`src/pipeline/rag.py::_get_index()` is a lazy global with no concurrency guard — under FastAPI's threaded request handling, two concurrent first-requests could both trigger `build_index()`. Since Phase 1 already touches this function to add the dense index alongside it, both indices will be **loaded once at server startup** (in `scripts/run_server.py`, before `uvicorn.run`) rather than lazily on first request, removing the race instead of adding a second copy of it.

## 4. Phase 2 — Generation Quality

Built and measured against the Phase 1 hybrid+rerank baseline.

### 4.1 Chunking (targeted, not blanket)

New module `src/data/chunking.py`. Only passages over a threshold (~500 words) are split, at sentence boundaries, into ~200-word chunks with slight overlap; the ~99% of passages already at abstract length are left untouched. Chunk ids are `f"{parent_id}::{chunk_idx}"`.

**Critical invariant:** every retrieval hit (BM25, dense, fused, reranked) is mapped back to its parent passage id before being (a) shown to the user, (b) scored against eval gold labels, or (c) deduplicated in fusion — because `relevant_passage_ids` in the eval set refers to original passage ids, not chunk ids. This mapping lives in one place (`src/data/chunking.py::parent_id_of(chunk_id)`) and both retrieval modules and `scripts/evaluate_retrieval.py` use it.

### 4.2 Query understanding: HyDE + keyword expansion

New module `src/query/rewrite.py`. One LLM call (reusing `src/generation/llm.py`) produces two things from the raw question:
1. A hypothetical passage that would answer the question — embedded and used as the **dense-retrieval query** instead of the raw question (HyDE). This closes the phrasing gap between colloquial questions and technical abstracts.
2. A short list of synonym/acronym expansions (e.g. "heart attack" → "myocardial infarction") appended to the **BM25 query** only — dense search uses the hypothetical passage, not this expansion, to avoid mixing signals.

Config-gated (`[rewrite] enabled = false` default). On LLM failure, falls back to raw-question search for both legs rather than failing the request — this stage is an enhancement, not a hard dependency.

### 4.3 Citation-grounded generation

- `src/prompts/templates.py`: system/user templates updated to require inline citations per claim, e.g. `[Passage N]`.
- `src/generation/parser.py`: extract cited passage numbers via regex; cross-check against the actually-retrieved set; citations to passages that weren't retrieved are stripped (and logged, not silently kept) rather than trusted at face value.
- API response (`scripts/run_server.py::QueryResponse`) gains a `citations: List[{marker: str, passage_id: int}]` field mapping each in-text citation to its real passage.

### 4.4 Groundedness guardrail

A second, cheap LLM call (`src/generation/groundedness.py`) compares the final answer against only its cited passages and returns a 0–1 groundedness score. Below a configurable threshold (`[guardrail] groundedness_threshold`, default 0.5), the response is flagged (`groundedness_flag: bool` in the API response) — not blocked or retried, to keep latency bounded. No new provider; reuses the Groq/OpenAI client.

### 4.5 Eval extension

New script `scripts/evaluate_answers.py`: for the test QA split, generates an answer per question through the full Phase 2 pipeline and scores it against the gold `answer` field using an LLM-as-judge faithfulness score (string-overlap metrics like ROUGE are weak on free-text biomedical answers). Reports aggregate faithfulness and the guardrail's own groundedness scores across the test set, written to `artifacts/eval/answers_<timestamp>.json`. This gives Phase 2 the same before/after measurability Phase 1 gets from `evaluate_retrieval.py`.

## 5. Config Schema (final, both phases)

```toml
[retrieval]
mode = "bm25"                 # "bm25" | "hybrid"
top_k = 5
sparse_top_n = 20
dense_top_n = 20
fusion_k = 60
rerank = false
rerank_top_n = 5

[rewrite]
enabled = false               # HyDE + BM25 keyword expansion

[citation]
enforce = false                # require inline [Passage N] citations

[guardrail]
groundedness_enabled = false
groundedness_threshold = 0.5

[llm]
model = "gpt-4o-mini"
max_tokens = 512
temperature = 0.2
```

All new flags default to today's behavior (off/bm25) — enabling the new methodology is opt-in via config, not a breaking change.

## 6. New Dependencies & Env Vars

| Dependency | Purpose |
|---|---|
| `faiss-cpu` | dense vector index |
| `numpy` | embedding arrays |
| `voyageai` (or `httpx` direct call) | reranking |

| Env var | Required when |
|---|---|
| `VOYAGE_API_KEY` | `rerank = true` |
| `OPENAI_API_KEY` | dense embeddings (`mode = "hybrid"`) — already supported as an optional key today for LLM generation, now also used for embeddings |

## 7. Testing Plan

- Unit tests: RRF fusion math against hand-computed fixtures; chunker sentence-boundary behavior and parent-id mapping; citation regex extraction/validation against malformed LLM output; config default backward-compatibility (`mode="bm25"` path produces identical results to pre-upgrade code, given a fixed corpus + mocked LLM response).
- Integration: `scripts/evaluate_retrieval.py` and `scripts/evaluate_answers.py` run end-to-end against the real test split as both a functional test and the quality acceptance gate.
- No test infrastructure exists in the repo today — this introduces `pytest` as a new dev dependency and a `tests/` directory.

## 8. Out of Scope / Non-Goals

- No move to a hosted vector DB (Qdrant/Pinecone/Chroma server) — FAISS in-process is sufficient at 40k passages.
- No local/GPU embedding or reranking models — API-only per explicit decision.
- No conversation memory / multi-turn chat — single-question-answer remains the interaction model.
- No streaming responses — out of scope for this upgrade, could be a future spec.
- No retry/self-correction loop on low groundedness scores — flag only, to keep latency bounded.

## 9. Risks

- Per-query latency grows with hybrid + rerank + (Phase 2) rewrite + guardrail — each is config-gated independently so latency/quality can be tuned, but the fully-on configuration could reach several seconds per query. Not measured until Phase 1/2 eval scripts are run; no explicit latency budget is set in this spec.
- Voyage adds a new external dependency/provider; if it has an outage, `rerank = true` requests fail — no fallback-to-unreranked-order is specified here and should be decided during implementation.
