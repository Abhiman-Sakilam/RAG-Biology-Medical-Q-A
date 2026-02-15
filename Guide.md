# RAG Codebase Guide

This guide explains what the project does, what technologies and patterns are used, why they are used, and how they work together.

---

## Overview

This is a **RAG (Retrieval-Augmented Generation)** application for **biology/medical Q&A**. It:

1. **Retrieves** relevant text passages from a corpus using **BM25**
2. **Prompts** an **LLM** (Groq or OpenAI) with those passages and the user question
3. **Returns** an answer plus the supporting passages

You can run it via **CLI**, **Web UI**, or **Docker**.

---

## Technologies & Components

### 1. BM25 (sparse retrieval) — `src/retrieval/sparse.py`

| | |
|---|---|
| **What** | BM25 is a bag-of-words ranking algorithm (keyword-based). |
| **Why** | Simple, fast, no embeddings; works well when exact/overlapping terms matter (e.g. medical terms). |
| **How** | Corpus is loaded from `data/json/text-corpus/*.json` (each item has `passage` and `id`). Passages are tokenized (lowercase `\w+`), then `BM25Okapi(tokenized)` builds the index. `search(query, bm25, corpus, id_to_passage, k)` tokenizes the query, gets BM25 scores, and returns top-k `(passage_id, text, score)`. |

**Dependency:** `rank_bm25`

---

### 2. LLM (Groq / OpenAI) — `src/generation/llm.py`

| | |
|---|---|
| **What** | Uses the `openai` Python client to call Groq (default) or OpenAI. |
| **Why** | To generate a single, concise answer given the retrieved context. |
| **How** | API key is read from `setup/.env` (`GROQ_API_KEY` or `OPENAI_API_KEY`). `generate(prompt, model, max_tokens, temperature)` calls `client.chat.completions.create(...)` with one user message. Model and params can be overridden in `config.toml` or env (e.g. `GROQ_MODEL`). |

**Dependency:** `openai`, `python-dotenv`

---

### 3. Config — `config.toml` (from `setup/config.toml`)

| | |
|---|---|
| **What** | TOML file with `[retrieval]` and `[llm]` sections. |
| **Why** | Single place for top_k, model, max_tokens, temperature without code changes. |
| **How** | `src/pipeline/rag.py` uses `tomllib`/`tomli` to load `config.toml` at project root and merge with defaults. Used for `top_k`, `model`, `max_tokens`, `temperature` in `rag_query()`. |

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
| **How** | Loads config; lazily builds one global BM25 index via `_get_index()` (calls `build_index()` once). `rag_query(question, top_k=..., config=...)` runs: `search()` → `format_passages_as_context()` → `build_rag_prompt()` → `generate()` → `parse_answer()`. Returns `(passages, answer)`. |

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

---

### 11. Docker

| | |
|---|---|
| **What** | Multi-stage Dockerfile (Node for frontend build, then Python image with built frontend + API). |
| **Why** | Single image to run server + UI; data (and artifacts) can be mounted. |
| **How** | Builds frontend, then copies `frontend/dist` into Python image; CMD runs `python scripts/run_server.py`. `docker-compose.yaml` mounts `../data` and `../artifacts` into the container. |

---

## Significance of the `artifacts` Folder

- **In the repo:** The **artifacts** directory is **not in the repo**: it’s listed in `.gitignore`. **No Python code in this project reads or writes to `artifacts`.** So in the current codebase it is unused.

- **In Docker:** In `setup/docker-compose.yaml`, the project root’s `artifacts` is mounted into the container as `/app/artifacts`. So **artifacts** is a **designated place for runtime/generated files** when running in Docker, e.g.:
  - Persisting the BM25 index so you don’t rebuild it every run
  - Logs, caches, or other outputs you want to keep on the host
  The `.dockerignore` excludes `artifacts/` from the image build, so the image doesn’t copy it; only the volume mount provides `/app/artifacts` at runtime.

**Summary:** The **artifacts** folder is a **reserved, git-ignored directory** that is **mounted into the app in Docker** for future or manual use (e.g. index persistence, logs, cache). The application code does not use it yet; it’s there so you can add that later or use it from the host without changing the repo layout.

---

## Quick reference

| Area | Location |
|------|----------|
| Config | `config.toml` (from `setup/config.toml`), `setup/.env` |
| Corpus | `data/json/text-corpus/` |
| Retrieval | `src/retrieval/sparse.py` |
| Prompts | `src/prompts/` |
| LLM | `src/generation/llm.py` |
| Pipeline | `src/pipeline/rag.py` |
| Server | `scripts/run_server.py` |
| Frontend | `frontend/` |
| Docker | `setup/Dockerfile`, `setup/docker-compose.yaml` |
