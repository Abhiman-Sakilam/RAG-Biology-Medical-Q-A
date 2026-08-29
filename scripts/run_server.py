#!/usr/bin/env python3
"""FastAPI server: GET / = React UI, POST /query = RAG."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.rag import rag_query_full, load_indices, _load_config

app = FastAPI(title="RAG API")
_cfg = _load_config()


@app.on_event("startup")
def _load_indices_on_startup():
    load_indices(mode=_cfg.get("retrieval", {}).get("mode", "bm25"))


web_dir = project_root / "frontend" / "dist"
if (web_dir / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(web_dir / "assets")), name="assets")


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
        raise HTTPException(503, str(e)) from e

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
