# RAG

Retrieval-augmented Q&A over biology and medical literature: BM25 retrieval + LLM (Groq or OpenAI). Ask questions and get answers grounded in the corpus via CLI, web UI, or Docker.

## Setup

```bash
./setup/setup.sh
```

Copies `setup/config.toml` → `config.toml` and creates `setup/.env` from `setup/.env.example` if missing. Set **`GROQ_API_KEY`** in `setup/.env` (get one at https://console.groq.com).

## Run

**CLI**

```bash
python scripts/run_rag.py --question "What is medulloblastoma?"
python scripts/run_rag.py -q "..." --out result.json
```

**Web UI**

```bash
cd frontend && npm install && npm run build && cd ..
python scripts/run_server.py
```

Open http://localhost:8060. Dev: `cd frontend && npm run dev` → http://localhost:5173 (proxy to backend on 8060).

**Docker**

```bash
./setup/docker-setup.sh
# or: docker compose -f setup/docker-compose.yaml up --build
```

## API

- `GET /` – React UI
- `POST /query` – body `{"question": "..."}` → `{"answer": "...", "passages": [...]}`

## Config

- `config.toml` (project root): `[retrieval]` top_k, `[llm]` model, max_tokens, temperature
- `setup/.env`: GROQ_API_KEY (required). Optional: GROQ_MODEL, or OPENAI_API_KEY + OPENAI_BASE_URL for OpenAI.
