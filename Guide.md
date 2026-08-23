# RAG Codebase Guide

This guide explains what the project does, what technologies and patterns are used, why they are used, and how they work together.

---

## Overview

This is a **RAG (Retrieval-Augmented Generation)** application for **biology/medical Q&A**. It:

1. **Retrieves** relevant text passages from a corpus using **BM25** (default) or **hybrid retrieval** — BM25 + OpenRouter LFM2.5 embeddings fused via Reciprocal Rank Fusion, with optional **OpenRouter NVIDIA Nemotron reranking** — as an opt-in mode set in config
2. **Prompts** an **LLM** (Groq or OpenAI) with those passages and the user question
3. **Returns** an answer plus the supporting passages

You can run it via **CLI**, **Web UI**, or **Docker**.

---

## Technologies & Components

### 1. BM25 (sparse retrieval) — `src/retrieval/sparse.py`

| | |
|---|---|
| **What** | BM25 is a bag-of-words ranking algorithm (keyword-based). This is the default retrieval mode (`[retrieval] mode = "bm25"`). |
| **Why** | Simple, fast, no embeddings; works well when exact/overlapping terms matter (e.g. medical terms). |
| **How** | Corpus is loaded from `data/json/text-corpus/*.json` (each item has `passage` and `id`). Passages are tokenized (lowercase `\w+`), then `BM25Okapi(tokenized)` builds the index. `search(query, bm25, corpus, id_to_passage, k)` tokenizes the query, gets BM25 scores, and returns top-k `(passage_id, text, score)`. |

**Dependency:** `rank_bm25`

---

### 1b. Hybrid retrieval + reranking (opt-in) — `src/retrieval/dense.py`, `src/retrieval/fusion.py`, `src/retrieval/rerank.py`

| | |
|---|---|
| **What** | An opt-in retrieval mode (`[retrieval] mode = "hybrid"`) that combines BM25 with dense (embedding-based) retrieval, fuses the two rankings, and optionally reranks the fused candidates. |
| **Why** | Dense embeddings catch semantically related passages that don't share exact keywords with the query; combining with BM25 (via fusion) tends to beat either alone. Reranking with a cross-encoder-style model further sharpens the final top-k. |
| **How** | `src/retrieval/dense.py`: `build_dense_index()`/`embed_texts()` call Voyage AI's `voyage-3-lite` embeddings API (via direct `httpx` calls) to embed the corpus into a FAISS `IndexFlatIP`; `dense_search()` embeds the query and searches the index. The index is built and persisted once via `scripts/build_embeddings.py` (into `artifacts/dense_index/`) rather than rebuilt per request. `src/retrieval/fusion.py`: `rrf_fuse(sparse_results, dense_results, k)` combines both rankings using Reciprocal Rank Fusion. `src/retrieval/rerank.py`: `rerank(query, candidates, top_n)` calls OpenRouter's free NVIDIA Nemotron reranking model (`nvidia/llama-nemotron-rerank-vl-1b-v2:free`, via the OpenRouter completions API) to re-score and reorder the fused candidate pool, used only when `[retrieval] rerank = true`. Voyage AI handles embeddings and OpenRouter handles reranking — hybrid mode with reranking enabled needs both `VOYAGE_API_KEY` and `OPENROUTER_API_KEY`. All of this is orchestrated by `src.pipeline.rag.retrieve_candidates()` (see below), which is the single implementation shared by the live pipeline and the evaluation harness. |

**Dependencies:** `faiss-cpu`, `numpy`, `httpx`. Requires `VOYAGE_API_KEY` for hybrid mode's embeddings, and `OPENROUTER_API_KEY` when `rerank = true`.

---

### 2. LLM (Groq / OpenAI) — `src/generation/llm.py`

| | |
|---|---|
| **What** | Uses the `openai` Python client to call Groq (default) or OpenAI. |
| **Why** | To generate a single, concise answer given the retrieved context. |
| **How** | API key is read from `setup/.env` (`GROQ_API_KEY` or `OPENAI_API_KEY`). `generate(prompt, model, max_tokens, temperature)` calls `client.chat.completions.create(...)` with one user message. Model and params can be overridden in `config.toml` or env (e.g. `GROQ_MODEL`). `.env` loading is centralized in `src/config/env.py` (`load_env()`), called at import time by `llm.py`, `src/retrieval/dense.py`, and `src/retrieval/rerank.py`, so `setup/.env` is loaded regardless of which entry point/script runs first. `VOYAGE_API_KEY` is required for hybrid mode's embeddings, and `OPENROUTER_API_KEY` is required, separately, whenever `[retrieval] rerank = true`. |

**Dependency:** `openai`, `python-dotenv`

---

### 3. Config — `config.toml` (from `setup/config.toml`)

| | |
|---|---|
| **What** | TOML file with `[retrieval]` and `[llm]` sections. |
| **Why** | Single place for retrieval mode/params, model, max_tokens, temperature without code changes. |
| **How** | `src/pipeline/rag.py` uses `tomllib`/`tomli` to load `config.toml` at project root and merge with defaults. `[retrieval]` keys: `mode` (`"bm25"` or `"hybrid"`), `top_k` (final passages returned), `sparse_top_n`/`dense_top_n` (candidates fetched from each retriever before fusion), `fusion_k` (RRF constant), `rerank` (enable OpenRouter NVIDIA Nemotron reranking), `rerank_top_n` (candidates kept after reranking, before the final `top_k` cut). `[llm]` keys: `model`, `max_tokens`, `temperature`. |

---

### 4. Prompts — `src/prompts/`

| | |
|---|---|
| **What** | System + user template and a formatter that turns passages into context text. |
| **Why** | Keeps RAG behavior consistent: “answer only from context; if not enough info, say so.” |
| **How** | `templates.py`: `SYSTEM_TEMPLATE` + `USER_TEMPLATE` with `{context}` and `{question}`. `formatter.py`: `format_passages_as_context(passages)` builds `[Passage 1] ... [Passage 2] ...` (with optional scores, truncation at ~6000 chars). `build_rag_prompt(context, question)` produces the full prompt string. |

---

### 5. Answer parsing — `src/generation/parser.py`

| | |
|---|---|
| **What** | Light cleanup of the raw LLM output. |
| **Why** | Avoid huge answers and messy whitespace in the UI/API. |
| **How** | `parse_answer(raw)` strips, collapses many newlines to double newline, and truncates to `max_length` (default 2000). |

---

### 6. Data loaders — `src/data/loaders.py` & `lookup.py`

| | |
|---|---|
| **What** | Load JSON corpora and build id → passage map. |
| **Why** | Corpus lives in `data/json/`; retrieval needs list of docs + fast id → text. |
| **How** | `load_corpus(split, data_dir)` → list of `{id, passage}` from `data/json/text-corpus/{split}-00000-of-00001.json`. `load_qa(...)` for question-answer-passages from `data/json/question-answer-passages/`. `build_lookup(corpus)` → `{id: passage}` for looking up text by id after BM25 returns indices/ids. |

---

### 7. RAG pipeline — `src/pipeline/rag.py`

| | |
|---|---|
| **What** | Orchestrates retrieval → context → prompt → LLM → parse. |
| **Why** | Single entry point for “ask a question, get answer + passages.” |
| **How** | Loads config; `load_indices(mode)` builds/loads the BM25 index (and, in hybrid mode, the persisted dense index) once at process startup, and `_ensure_loaded(mode)` lazily loads them on first use otherwise. `retrieve_candidates(question, r_cfg)` runs BM25 (+ dense + RRF fusion + optional OpenRouter rerank, per `r_cfg`) and returns the final top-k candidates — this is the single retrieval implementation shared by `rag_query()` and `scripts/evaluate_retrieval.py`, so the eval harness always measures exactly what the live pipeline does. `rag_query(question, top_k=..., config=...)` runs: `retrieve_candidates()` → `format_passages_as_context()` → `build_rag_prompt()` → `generate()` → `parse_answer()`. Returns `(passages, answer)`. |

---

### 8. FastAPI server — `scripts/run_server.py`

| | |
|---|---|
| **What** | Serves the React app and the RAG API. |
| **Why** | One process for both UI and `POST /query`. |
| **How** | Serves `frontend/dist` at `/` (and `/assets`). `POST /query` with `{"question": "..."}` calls `rag_query(req.question)` and returns `{question, answer, passages}`. Runs with uvicorn on port 8060. |

**Dependencies:** `fastapi`, `uvicorn`, `pydantic`

---

### 9. Frontend — React + Vite

| | |
|---|---|
| **What** | React 18, Vite 5, no extra UI lib. |
| **Why** | Simple UI to type a question and see answer + passages. |
| **How** | `App.jsx`: form submit → `queryApi(question)` (POST to `/query`) → shows `ResultCard` with answer and passages, and `Status` for loading/error. `api.js`: `fetch('/query', { method: 'POST', body: JSON.stringify({ question }) })`. Vite dev server (port 5173) proxies `/query` to `http://localhost:8060` so the same backend is used in dev. |

**Dependencies:** `react`, `react-dom`, `vite`, `@vitejs/plugin-react`

---

### 10. Scripts

| Script | Purpose |
|--------|---------|
| **`run_rag.py`** | CLI: `python scripts/run_rag.py -q "..."` (and optional `--out result.json`, `--top-k`). Calls `rag_query()`, prints answer and passages. |
| **`search_index.py`** | BM25-only: build index and run `search(query, ...)` to print top-k passages (no LLM). Useful for debugging retrieval. |
| **`build_embeddings.py`** | Builds the dense (embedding) index from the corpus and persists it to `artifacts/dense_index/` (FAISS index + corpus id list + a corpus hash to skip rebuilding when unchanged). Run once before enabling `mode = "hybrid"`. |
| **`evaluate_retrieval.py`** | Phase 1 retrieval quality evaluation harness: computes Recall/MRR/nDCG@k for bm25 vs hybrid vs hybrid+rerank against the gold `relevant_passage_ids` in the test QA split, using the exact same `retrieve_candidates()` logic as the live pipeline. Writes results to `artifacts/eval/retrieval_results.json`. |

---

### 11. Docker

| | |
|---|---|
| **What** | Multi-stage Dockerfile (Node for frontend build, then Python image with built frontend + API). |
| **Why** | Single image to run server + UI; data (and artifacts) can be mounted. |
| **How** | Builds frontend, then copies `frontend/dist` into Python image; CMD runs `python scripts/run_server.py`. `docker-compose.yaml` mounts `../data` and `../artifacts` into the container. |

---

## Significance of the `artifacts` Folder

- **In the repo:** The **artifacts** directory is **not in the repo**: it’s listed in `.gitignore`. It is, however, actively read and written by the application code:
  - **`artifacts/dense_index/`** holds the persisted FAISS index (`index.faiss`), the corpus id order (`corpus_ids.json`), and a corpus content hash (`corpus.sha256`) used to detect when a rebuild is needed. Built and written by `scripts/build_embeddings.py` (`build_and_persist()`/`save_index()`), and read by `src.retrieval.dense.load_index()` (via `src.pipeline.rag.load_indices()`) whenever `mode = "hybrid"`.
  - **`artifacts/eval/`** holds `retrieval_results.json`, written by `scripts/evaluate_retrieval.py` after each evaluation run.

- **In Docker:** In `setup/docker-compose.yaml`, the project root’s `artifacts` is mounted into the container as `/app/artifacts`, so the persisted dense index and eval output survive across container restarts/rebuilds without being baked into the image. The `.dockerignore` excludes `artifacts/` from the image build, so the image doesn’t copy it; only the volume mount provides `/app/artifacts` at runtime.

**Summary:** The **artifacts** folder is a **reserved, git-ignored directory**, mounted into the app in Docker, that holds generated runtime state: the persisted dense (FAISS) index used by hybrid retrieval, and retrieval evaluation output.

---

## Quick reference

| Area | Location |
|------|----------|
| Config | `config.toml` (from `setup/config.toml`), `setup/.env` |
| Corpus | `data/json/text-corpus/` |
| Retrieval (sparse) | `src/retrieval/sparse.py` |
| Retrieval (dense/fusion/rerank) | `src/retrieval/dense.py`, `src/retrieval/fusion.py`, `src/retrieval/rerank.py` |
| Env loading | `src/config/env.py` |
| Prompts | `src/prompts/` |
| LLM | `src/generation/llm.py` |
| Pipeline | `src/pipeline/rag.py` |
| Server | `scripts/run_server.py` |
| Embedding build script | `scripts/build_embeddings.py` |
| Retrieval eval harness | `scripts/evaluate_retrieval.py` |
| Tests | `tests/` (run with `.venv/bin/pytest`; deps in `setup/requirements-dev.txt`) |
| Frontend | `frontend/` |
| Docker | `setup/Dockerfile`, `setup/docker-compose.yaml` |
