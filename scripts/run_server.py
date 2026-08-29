#!/usr/bin/env python3
"""FastAPI server: GET / = React UI, POST /query = RAG."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import APIStatusError, APIConnectionError

from src.pipeline.rag import rag_query_full, load_indices, _load_config

_cfg = _load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the indices once, before the first request, so concurrent requests
    # never race to lazily build them.
    load_indices(mode=_cfg.get("retrieval", {}).get("mode", "bm25"))
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


web_dir = project_root / "frontend" / "dist"
if (web_dir / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(web_dir / "assets")), name="assets")


def _provider_error_detail(e: APIStatusError) -> str:
    """Turn a provider error into something a caller can act on."""
    if e.status_code == 413:
        return (
            "The prompt was too large for the configured model. Lower "
            "[retrieval] top_k in config.toml, or use a model with a bigger "
            "request limit."
        )
    if e.status_code == 429:
        return "The LLM provider rate-limited this request. Retry shortly."
    return f"LLM provider returned {e.status_code}: {e.message}"


class QueryRequest(BaseModel):
    question: str


class PassageOut(BaseModel):
    # Chunked passages carry a "parent::idx" string id (see src/data/chunking.py),
    # so this must not be narrowed to int.
    passage_id: int | str
    passage: str
    score: float


class CitationOut(BaseModel):
    marker: str
    passage_id: int | str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    passages: list[PassageOut]
    citations: list[CitationOut] = []
    groundedness_score: float | None = None
    groundedness_flagged: bool = False


@app.get("/")
def index():
    path = web_dir / "index.html"
    if not path.exists():
        raise HTTPException(404, "Build frontend first: cd frontend && npm run build")
    return FileResponse(path)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must be non-empty")
    try:
        result = rag_query_full(req.question)
    except ValueError as e:
        # Missing/invalid configuration, e.g. no API key.
        raise HTTPException(503, str(e)) from e
    except APIStatusError as e:
        detail = _provider_error_detail(e)
        # 413/429 are the client's problem to retry or the operator's to tune;
        # either way it is not an internal server fault.
        status = 413 if e.status_code == 413 else (429 if e.status_code == 429 else 502)
        raise HTTPException(status, detail) from e
    except APIConnectionError as e:
        raise HTTPException(504, f"Could not reach the LLM provider: {e}") from e

    # Build citations list from result
    citations = [
        CitationOut(
            marker=c["marker"],
            passage_id=c["passage_id"],
            text=c["text"],
            score=round(c["score"], 4),
        )
        for c in result.get("citations", [])
    ]

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        passages=[
            PassageOut(passage_id=p[0], passage=p[1], score=round(p[2], 4))
            for p in result["passages"]
        ],
        citations=citations,
        groundedness_score=result.get("groundedness_score"),
        groundedness_flagged=result.get("groundedness_flagged", False),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8060)
