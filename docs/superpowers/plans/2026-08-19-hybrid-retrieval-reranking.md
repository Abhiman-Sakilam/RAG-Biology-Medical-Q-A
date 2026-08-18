# Hybrid Retrieval + Reranking (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade retrieval from BM25-only to hybrid (BM25 + dense embeddings, fused with Reciprocal Rank Fusion) with Voyage AI reranking, and add a retrieval evaluation harness that proves the upgrade actually improves Recall/MRR/nDCG on the existing gold-labeled test set.

**Architecture:** Two new retrieval backends (`src/retrieval/dense.py`, using OpenAI embeddings + an in-process FAISS index persisted to `artifacts/dense_index/`) and a reranker (`src/retrieval/rerank.py`, calling Voyage's `rerank-2` REST API) are fused with a new `src/retrieval/fusion.py` (Reciprocal Rank Fusion) and wired into the existing `src/pipeline/rag.py::rag_query()` behind config flags. Every new stage is off by default (`mode="bm25"`, `rerank=false`), so existing behavior is unchanged until explicitly enabled.

**Tech Stack:** Python, FAISS (`faiss-cpu`), OpenAI embeddings API (`text-embedding-3-small`), Voyage AI rerank API (`rerank-2`) via `httpx`, `pytest` for tests (new to this repo).

**Spec:** `docs/superpowers/specs/2026-08-19-rag-methodology-upgrade-design.md` (sections 3 and 5-7 cover this plan; section 4 is Phase 2, a separate follow-up plan).

## Global Constraints

- All new config flags default to today's exact behavior — `mode="bm25"`, `rerank=false` — so this upgrade is opt-in, not breaking. (spec §3.3, §5)
- API-only: no local/GPU ML models for embeddings or reranking. (spec §8)
- Target repo is `RAG-Biology-Medical-Q-A/` only — the sibling `temp/` mirror is out of scope. (spec header)
- Dense index row order must exactly match corpus row order; any mismatch must raise an error rather than silently return wrong passages for a FAISS row index. (spec §3.5, correctness risk identified during planning)
- Retrieval hit tuples keep the existing shape `(passage_id: int, text: str, score: float)` across BM25, dense, fusion, and rerank so every stage is interchangeable. (spec §3.1)
- New env vars (`OPENAI_API_KEY` for embeddings, `VOYAGE_API_KEY` for reranking) are only required when the corresponding feature is enabled via config. (spec §6)

---

### Task 1: Test Infrastructure + Reciprocal Rank Fusion

**Files:**
- Create: `setup/requirements-dev.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_fusion.py`
- Create: `src/retrieval/fusion.py`

**Interfaces:**
- Produces: `rrf_fuse(sparse_results: List[Tuple[int, str, float]], dense_results: List[Tuple[int, str, float]], k: int = 60, top_n: int = None) -> List[Tuple[int, str, float]]` — used by Task 6.

This repo has no test suite yet. This task adds `pytest` and a `tests/` package that can import from `src/` the same way `scripts/run_rag.py` already does (`sys.path.insert` of the project root), and delivers the first new retrieval component: Reciprocal Rank Fusion, which combines a BM25 ranked list and a dense-embedding ranked list into one fused ranking without needing the two score scales to be comparable.

- [ ] **Step 1: Add the dev dependency file**

Create `setup/requirements-dev.txt`:

```
pytest>=8.0.0
```

- [ ] **Step 2: Install it**

Run: `pip install -r setup/requirements-dev.txt`

- [ ] **Step 3: Add test path setup**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
```

- [ ] **Step 4: Write the failing tests**

Create `tests/test_fusion.py`:

```python
from src.retrieval.fusion import rrf_fuse


def test_rrf_fuse_ranks_items_in_both_lists_higher():
    sparse = [(1, "a", 5.0), (2, "b", 3.0), (3, "c", 1.0)]
    dense = [(2, "b", 0.9), (3, "c", 0.8), (4, "d", 0.7)]
    fused = rrf_fuse(sparse, dense, k=60)
    fused_ids = [pid for pid, _, _ in fused]
    assert fused_ids[0] == 2
    assert set(fused_ids) == {1, 2, 3, 4}


def test_rrf_fuse_respects_top_n():
    sparse = [(1, "a", 5.0), (2, "b", 3.0)]
    dense = [(3, "c", 0.9)]
    fused = rrf_fuse(sparse, dense, k=60, top_n=2)
    assert len(fused) == 2


def test_rrf_fuse_score_matches_hand_computed_value():
    sparse = [(1, "a", 5.0)]
    dense = [(1, "a", 0.9)]
    fused = rrf_fuse(sparse, dense, k=60)
    expected = 1.0 / 61 + 1.0 / 61
    assert fused[0] == (1, "a", expected)


def test_rrf_fuse_handles_sparse_only_id():
    sparse = [(1, "a", 5.0)]
    dense = []
    fused = rrf_fuse(sparse, dense, k=60)
    assert fused == [(1, "a", 1.0 / 61)]
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.fusion'`

- [ ] **Step 6: Implement Reciprocal Rank Fusion**

Create `src/retrieval/fusion.py`:

```python
from typing import Dict, List, Optional, Tuple


def rrf_fuse(
    sparse_results: List[Tuple[int, str, float]],
    dense_results: List[Tuple[int, str, float]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Tuple[int, str, float]]:
    scores: Dict[int, float] = {}
    texts: Dict[int, str] = {}
    for rank, (pid, text, _) in enumerate(sparse_results, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        texts[pid] = text
    for rank, (pid, text, _) in enumerate(dense_results, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        texts[pid] = text
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        fused = fused[:top_n]
    return [(pid, texts[pid], score) for pid, score in fused]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_fusion.py -v`
Expected: PASS (4 tests)

- [ ] **Step 8: Commit**

```bash
git add setup/requirements-dev.txt tests/conftest.py tests/test_fusion.py src/retrieval/fusion.py
git commit -m "feat: add Reciprocal Rank Fusion and test infrastructure"
```

---

### Task 2: Dense Retrieval Module (Embeddings + FAISS)

**Files:**
- Modify: `setup/requirements.txt`
- Create: `src/retrieval/dense.py`
- Create: `tests/test_dense.py`

**Interfaces:**
- Consumes: `src.data.loaders.load_corpus() -> List[dict]`, `src.data.lookup.build_lookup(corpus) -> Dict[int, str]` (both existing, unchanged).
- Produces: `embed_texts(texts: List[str], client=None, batch_size: int = 100) -> np.ndarray`, `build_dense_index(corpus: List[dict] = None, client=None) -> Tuple[faiss.Index, List[dict], dict]`, `dense_search(query: str, index, corpus: List[dict], id_to_passage: dict, k: int = 5, client=None) -> List[Tuple[int, str, float]]`, `save_index(index, corpus: List[dict], path: Path) -> None`, `load_index(path: Path, corpus: List[dict]) -> faiss.Index`. Used by Task 3 and Task 6.

This is the dense half of hybrid retrieval: OpenAI embeddings (`text-embedding-3-small`) indexed in a FAISS `IndexFlatIP` over L2-normalized vectors (so inner product = cosine similarity). `load_index` includes an integrity check — the persisted corpus id order must match the corpus passed in, otherwise a FAISS row index would silently map to the wrong passage id.

- [ ] **Step 1: Add production dependencies**

Add to `setup/requirements.txt`:

```
faiss-cpu>=1.8.0
numpy>=1.26.0
```

Run: `pip install -r setup/requirements.txt`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dense.py`:

```python
import numpy as np
import pytest
import faiss

from src.retrieval.dense import (
    embed_texts,
    build_dense_index,
    dense_search,
    save_index,
    load_index,
)


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbeddingItem(e) for e in embeddings]


class _FakeEmbeddingClient:
    def __init__(self, embedding_map):
        self.embedding_map = embedding_map
        self.embeddings = self

    def create(self, model, input):
        return _FakeEmbeddingResponse([self.embedding_map[text] for text in input])


def test_embed_texts_normalizes_vectors():
    client = _FakeEmbeddingClient({"a": [3.0, 4.0]})
    vectors = embed_texts(["a"], client=client)
    assert vectors.shape == (1, 2)
    norm = float(np.linalg.norm(vectors[0]))
    assert abs(norm - 1.0) < 1e-6


def test_build_dense_index_and_dense_search_returns_best_match():
    corpus = [{"id": 1, "passage": "alpha"}, {"id": 2, "passage": "beta"}]
    client = _FakeEmbeddingClient({
        "alpha": [1.0, 0.0],
        "beta": [0.0, 1.0],
        "query like alpha": [1.0, 0.0],
    })
    index, built_corpus, id_to_passage = build_dense_index(corpus, client=client)
    results = dense_search("query like alpha", index, built_corpus, id_to_passage, k=1, client=client)
    assert results[0][0] == 1
    assert results[0][1] == "alpha"


def test_save_and_load_index_round_trip(tmp_path):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    corpus = [{"id": 10, "passage": "a"}, {"id": 20, "passage": "b"}]
    save_index(index, corpus, tmp_path)
    loaded = load_index(tmp_path, corpus)
    assert loaded.ntotal == 2


def test_load_index_raises_on_corpus_mismatch(tmp_path):
    vectors = np.array([[1.0, 0.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    save_index(index, [{"id": 10, "passage": "a"}], tmp_path)
    with pytest.raises(ValueError):
        load_index(tmp_path, [{"id": 999, "passage": "different"}])


def test_load_index_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path, [{"id": 1, "passage": "a"}])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_dense.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.dense'`

- [ ] **Step 4: Implement the dense retrieval module**

Create `src/retrieval/dense.py`:

```python
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from src.data.loaders import load_corpus

EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embedding_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY in setup/.env to use hybrid retrieval (dense embeddings)"
        )
    return OpenAI(api_key=api_key)


def embed_texts(
    texts: List[str],
    client: Optional[OpenAI] = None,
    batch_size: int = 100,
) -> np.ndarray:
    client = client or _get_embedding_client()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_dense_index(
    corpus: List[dict] = None,
    client: Optional[OpenAI] = None,
) -> Tuple["faiss.Index", List[dict], dict]:
    if corpus is None:
        corpus = load_corpus()
    from src.data.lookup import build_lookup

    texts = [item["passage"] for item in corpus]
    vectors = embed_texts(texts, client=client)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    id_to_passage = build_lookup(corpus)
    return index, corpus, id_to_passage


def dense_search(
    query: str,
    index: "faiss.Index",
    corpus: List[dict],
    id_to_passage: dict,
    k: int = 5,
    client: Optional[OpenAI] = None,
) -> List[Tuple[int, str, float]]:
    q_vec = embed_texts([query], client=client)
    scores, indices = index.search(q_vec, k)
    out = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        i = int(idx)
        pid = corpus[i]["id"]
        text = id_to_passage[pid]
        out.append((int(pid), text, float(score)))
    return out


def save_index(index: "faiss.Index", corpus: List[dict], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))
    with open(path / "corpus_ids.json", "w") as f:
        json.dump([item["id"] for item in corpus], f)


def load_index(path: Path, corpus: List[dict]) -> "faiss.Index":
    index_path = path / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            f'No dense index found at {path}. Run scripts/build_embeddings.py first, '
            f'or set mode="bm25" in config.toml.'
        )
    with open(path / "corpus_ids.json") as f:
        saved_ids = json.load(f)
    current_ids = [item["id"] for item in corpus]
    if saved_ids != current_ids:
        raise ValueError(
            "Dense index corpus order does not match the current corpus. "
            "Re-run scripts/build_embeddings.py to rebuild it."
        )
    return faiss.read_index(str(index_path))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_dense.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add setup/requirements.txt src/retrieval/dense.py tests/test_dense.py
git commit -m "feat: add dense embedding retrieval with FAISS"
```

---

### Task 3: Persist the Dense Index (`build_embeddings.py`)

**Files:**
- Create: `scripts/build_embeddings.py`
- Create: `tests/test_build_embeddings.py`

**Interfaces:**
- Consumes: `src.retrieval.dense.build_dense_index`, `src.retrieval.dense.save_index` (Task 2).
- Produces: `build_and_persist(corpus: List[dict], artifacts_dir: Path, corpus_hash_value: str, client=None) -> bool` (True if a new index was built). Used by the script's `main()`; not consumed by later tasks directly, but documents the on-disk layout (`artifacts/dense_index/index.faiss`, `corpus_ids.json`, `corpus.sha256`) that Task 6 reads via `load_index`.

A one-off script that embeds the full corpus once and persists the FAISS index to `artifacts/dense_index/` (the previously-unused, Docker-mounted directory), so the server never has to re-embed 40k passages on every restart. A sha256 hash of the corpus file gates rebuilding — an unchanged corpus is a no-op.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_embeddings.py`:

```python
from unittest import mock

from scripts.build_embeddings import build_and_persist


def _fake_client():
    client = mock.Mock()
    client.embeddings.create.return_value = mock.Mock(
        data=[mock.Mock(embedding=[1.0, 0.0]), mock.Mock(embedding=[0.0, 1.0])]
    )
    return client


def test_build_and_persist_creates_index(tmp_path):
    corpus = [{"id": 1, "passage": "alpha"}, {"id": 2, "passage": "beta"}]
    built = build_and_persist(corpus, tmp_path, "hash-v1", client=_fake_client())
    assert built is True
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "corpus_ids.json").exists()
    assert (tmp_path / "corpus.sha256").read_text().strip() == "hash-v1"


def test_build_and_persist_skips_when_hash_matches(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "corpus.sha256").write_text("hash-v1")
    client = _fake_client()
    built = build_and_persist([{"id": 1, "passage": "alpha"}], tmp_path, "hash-v1", client=client)
    assert built is False
    client.embeddings.create.assert_not_called()
```

Note: `scripts/` needs to be importable as a package for `from scripts.build_embeddings import build_and_persist` to work.

- [ ] **Step 2: Make `scripts/` importable**

Create `scripts/__init__.py` (empty file) if it does not already exist.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_build_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_embeddings'`

- [ ] **Step 4: Implement the script**

Create `scripts/build_embeddings.py`:

```python
#!/usr/bin/env python3
"""Build and persist the dense (embedding) index into artifacts/dense_index/.

Run this once after the corpus changes, or before enabling
`mode = "hybrid"` in config.toml for the first time.
"""
import hashlib
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_corpus
from src.retrieval.dense import build_dense_index, save_index

ARTIFACTS_DIR = project_root / "artifacts" / "dense_index"
CORPUS_PATH = project_root / "data" / "json" / "text-corpus" / "train-00000-of-00001.json"


def corpus_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_persist(
    corpus: List[dict],
    artifacts_dir: Path,
    corpus_hash_value: str,
    client=None,
) -> bool:
    """Build and persist the dense index. Returns True if rebuilt, False if
    the cached index already matches corpus_hash_value."""
    hash_file = artifacts_dir / "corpus.sha256"
    if hash_file.exists() and hash_file.read_text().strip() == corpus_hash_value:
        return False
    index, built_corpus, _ = build_dense_index(corpus, client=client)
    save_index(index, built_corpus, artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(corpus_hash_value)
    return True


def main():
    corpus = load_corpus()
    new_hash = corpus_file_hash(CORPUS_PATH)
    print(f"Embedding {len(corpus)} passages (skipped if unchanged since last run)...")
    built = build_and_persist(corpus, ARTIFACTS_DIR, new_hash)
    if built:
        print(f"Built and saved dense index to {ARTIFACTS_DIR}")
    else:
        print("Dense index already up to date — nothing to do.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_build_embeddings.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/build_embeddings.py scripts/__init__.py tests/test_build_embeddings.py
git commit -m "feat: add script to build and persist the dense embedding index"
```

---

### Task 4: Voyage Reranking Module

**Files:**
- Modify: `setup/requirements.txt`
- Modify: `setup/.env.example`
- Create: `src/retrieval/rerank.py`
- Create: `tests/test_rerank.py`

**Interfaces:**
- Produces: `rerank(query: str, candidates: List[Tuple[int, str, float]], top_n: int = 5, api_key: str = None, client=None) -> List[Tuple[int, str, float]]`. Used by Task 6.

Calls Voyage AI's `rerank-2` model via a direct REST call (`httpx`, no SDK — one endpoint doesn't need a dependency) to reorder the fused candidate list by true relevance before it's truncated to `top_k`.

- [ ] **Step 1: Add the dependency and env var placeholder**

Add to `setup/requirements.txt`:

```
httpx>=0.27.0
```

Add to `setup/.env.example` (append; do not remove existing content):

```
# Optional: required only when [retrieval] rerank = true in config.toml
VOYAGE_API_KEY=
```

Run: `pip install -r setup/requirements.txt`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_rerank.py`:

```python
import pytest

from src.retrieval.rerank import rerank


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        return _FakeResponse(self._payload)

    def close(self):
        pass


def test_rerank_reorders_by_relevance_score():
    candidates = [(1, "low relevance", 0.9), (2, "high relevance", 0.5)]
    fake_client = _FakeClient({
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.2},
        ]
    })
    result = rerank("query", candidates, top_n=2, api_key="test-key", client=fake_client)
    assert result == [(2, "high relevance", 0.95), (1, "low relevance", 0.2)]


def test_rerank_empty_candidates_returns_empty_without_calling_api():
    fake_client = _FakeClient({"results": []})
    assert rerank("query", [], api_key="test-key", client=fake_client) == []
    assert fake_client.calls == []


def test_rerank_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        rerank("query", [(1, "a", 1.0)])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval.rerank'`

- [ ] **Step 4: Implement the reranker**

Create `src/retrieval/rerank.py`:

```python
import os
from typing import List, Optional, Tuple

import httpx

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
VOYAGE_MODEL = "rerank-2"


def _get_api_key() -> str:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set VOYAGE_API_KEY in setup/.env to use reranking ([retrieval] rerank = true)"
        )
    return api_key


def rerank(
    query: str,
    candidates: List[Tuple[int, str, float]],
    top_n: int = 5,
    api_key: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> List[Tuple[int, str, float]]:
    if not candidates:
        return []
    api_key = api_key or _get_api_key()
    documents = [text for _, text, _ in candidates]
    payload = {
        "query": query,
        "documents": documents,
        "model": VOYAGE_MODEL,
        "top_k": min(top_n, len(candidates)),
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    owns_client = client is None
    client = client or httpx.Client(timeout=10.0)
    try:
        resp = client.post(VOYAGE_RERANK_URL, json=payload, headers=headers)
        resp.raise_for_status()
        results = resp.json()["results"]
    finally:
        if owns_client:
            client.close()
    out = []
    for item in results:
        pid, text, _ = candidates[item["index"]]
        out.append((pid, text, float(item["relevance_score"])))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rerank.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add setup/requirements.txt setup/.env.example src/retrieval/rerank.py tests/test_rerank.py
git commit -m "feat: add Voyage AI rerank-2 reranking module"
```

---

### Task 5: Config Schema for Hybrid Retrieval & Reranking

**Files:**
- Modify: `setup/config.toml`
- Modify: `src/pipeline/rag.py` (only `_load_config`, lines 20-30 in the current file)
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_load_config(path: Path = None) -> Dict[str, Any]` now returns a `retrieval` dict with keys `mode, top_k, sparse_top_n, dense_top_n, fusion_k, rerank, rerank_top_n`. Used by Task 6.

Adds the new config keys from spec §5 with backward-compatible defaults, and makes `_load_config` accept an explicit path so it's testable without touching the real `config.toml`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
from src.pipeline.rag import _load_config


def test_load_config_defaults_when_no_file(tmp_path):
    cfg = _load_config(tmp_path / "missing.toml")
    assert cfg["retrieval"]["mode"] == "bm25"
    assert cfg["retrieval"]["rerank"] is False
    assert cfg["retrieval"]["sparse_top_n"] == 20
    assert cfg["retrieval"]["dense_top_n"] == 20
    assert cfg["retrieval"]["fusion_k"] == 60
    assert cfg["retrieval"]["rerank_top_n"] == 5


def test_load_config_merges_partial_overrides(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[retrieval]\nmode = "hybrid"\nrerank = true\n')
    cfg = _load_config(config_file)
    assert cfg["retrieval"]["mode"] == "hybrid"
    assert cfg["retrieval"]["rerank"] is True
    assert cfg["retrieval"]["top_k"] == 5  # untouched default preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `TypeError: _load_config() takes 0 positional arguments but 1 was given`

- [ ] **Step 3: Update `_load_config`**

In `src/pipeline/rag.py`, replace the existing `_load_config` function with:

```python
def _load_config(path: Path = None) -> Dict[str, Any]:
    path = path or (_project_root() / "config.toml")
    defaults = {
        "retrieval": {
            "mode": "bm25",
            "top_k": 5,
            "sparse_top_n": 20,
            "dense_top_n": 20,
            "fusion_k": 60,
            "rerank": False,
            "rerank_top_n": 5,
        },
        "llm": {"model": "llama-3.3-70b-versatile", "max_tokens": 512, "temperature": 0.2},
    }
    if not path.exists():
        return defaults
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {
        "retrieval": {**defaults["retrieval"], **(data.get("retrieval") or {})},
        "llm": {**defaults["llm"], **(data.get("llm") or {})},
    }
```

- [ ] **Step 4: Update `setup/config.toml`**

Replace the `[retrieval]` section in `setup/config.toml` with:

```toml
[retrieval]
mode = "bm25"          # "bm25" | "hybrid"
top_k = 5
sparse_top_n = 20
dense_top_n = 20
fusion_k = 60
rerank = false
rerank_top_n = 5
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full test suite so far to check for regressions**

Run: `pytest -v`
Expected: All tests from Tasks 1-5 PASS.

- [ ] **Step 7: Commit**

```bash
git add setup/config.toml src/pipeline/rag.py tests/test_config.py
git commit -m "feat: add hybrid retrieval and reranking config schema"
```

---

### Task 6: Wire Hybrid Retrieval + Reranking into the RAG Pipeline

**Files:**
- Modify: `src/pipeline/rag.py` (imports, index-loading globals, `rag_query`)
- Modify: `scripts/run_server.py` (load indices at startup)
- Create: `tests/test_rag_pipeline.py`
- Create: `tests/test_run_server_startup.py`

**Interfaces:**
- Consumes: `src.retrieval.sparse.build_index`, `src.retrieval.sparse.search` (existing); `src.retrieval.dense.load_index`, `src.retrieval.dense.dense_search` (Task 2); `src.retrieval.fusion.rrf_fuse` (Task 1); `src.retrieval.rerank.rerank` (Task 4); `_load_config` (Task 5).
- Produces: `load_indices(mode: str = "bm25") -> None` (module-level, populates `_bm25_state`/`_dense_state`). `rag_query(question, top_k=None, max_tokens=None, config=None) -> Tuple[List[Tuple[int,str,float]], str]` signature unchanged (backward compatible for `scripts/run_rag.py` and `scripts/run_server.py`, which are not otherwise modified).

This is the integration task: `rag_query` gains hybrid + rerank stages, fully gated by config so `mode="bm25", rerank=false` reproduces the exact pre-upgrade behavior. Index loading moves from a lazy, unguarded global (a race under FastAPI's threaded request handling) to an explicit `load_indices()` called once at server startup via a FastAPI startup event.

- [ ] **Step 1: Write the failing pipeline tests**

Create `tests/test_rag_pipeline.py`:

```python
import src.pipeline.rag as rag_module


def _fixture_corpus_state():
    corpus = [{"id": 1, "passage": "alpha"}, {"id": 2, "passage": "beta"}]
    id_to_passage = {1: "alpha", 2: "beta"}
    return ("fake-bm25", corpus, id_to_passage)


def _base_config(**retrieval_overrides):
    retrieval = {
        "mode": "bm25",
        "top_k": 1,
        "sparse_top_n": 20,
        "dense_top_n": 20,
        "fusion_k": 60,
        "rerank": False,
        "rerank_top_n": 5,
    }
    retrieval.update(retrieval_overrides)
    return {"retrieval": retrieval, "llm": {"model": "m", "max_tokens": 10, "temperature": 0.0}}


def test_bm25_mode_matches_pre_upgrade_behavior(monkeypatch):
    rag_module._bm25_state = _fixture_corpus_state()
    rag_module._dense_state = None
    monkeypatch.setattr(rag_module, "bm25_search", lambda q, bm25, corpus, lut, k: [(1, "alpha", 5.0)])
    monkeypatch.setattr(rag_module, "generate", lambda prompt, model, max_tokens, temperature: "The answer.")
    passages, answer = rag_module.rag_query("q?", config=_base_config())
    assert passages == [(1, "alpha", 5.0)]
    assert answer == "The answer."


def test_hybrid_mode_fuses_sparse_and_dense_results(monkeypatch):
    corpus, id_to_passage = _fixture_corpus_state()[1], _fixture_corpus_state()[2]
    rag_module._bm25_state = _fixture_corpus_state()
    rag_module._dense_state = ("fake-dense-index", corpus, id_to_passage)
    monkeypatch.setattr(rag_module, "bm25_search", lambda q, bm25, corpus, lut, k: [(1, "alpha", 5.0)])
    monkeypatch.setattr(rag_module, "dense_search", lambda q, idx, corpus, lut, k: [(2, "beta", 0.9)])
    monkeypatch.setattr(rag_module, "generate", lambda prompt, model, max_tokens, temperature: "answer")
    passages, _ = rag_module.rag_query("q?", config=_base_config(mode="hybrid", top_k=2))
    assert {p[0] for p in passages} == {1, 2}


def test_rerank_stage_invoked_when_enabled(monkeypatch):
    rag_module._bm25_state = _fixture_corpus_state()
    rag_module._dense_state = None
    monkeypatch.setattr(
        rag_module, "bm25_search",
        lambda q, bm25, corpus, lut, k: [(1, "alpha", 5.0), (2, "beta", 3.0)],
    )
    monkeypatch.setattr(
        rag_module, "voyage_rerank",
        lambda q, candidates, top_n: list(reversed(candidates))[:top_n],
    )
    monkeypatch.setattr(rag_module, "generate", lambda prompt, model, max_tokens, temperature: "answer")
    passages, _ = rag_module.rag_query("q?", config=_base_config(rerank=True, rerank_top_n=1))
    assert passages == [(2, "beta", 3.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_pipeline.py -v`
Expected: FAIL — `AttributeError: module 'src.pipeline.rag' has no attribute '_bm25_state'` (or similar, since these names don't exist yet)

- [ ] **Step 3: Update `src/pipeline/rag.py`**

Replace the full contents of `src/pipeline/rag.py` with:

```python
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from src.retrieval.sparse import build_index, search as bm25_search
from src.retrieval.dense import load_index as load_dense_index, dense_search
from src.retrieval.fusion import rrf_fuse
from src.retrieval.rerank import rerank as voyage_rerank
from src.prompts.formatter import format_passages_as_context
from src.prompts.templates import build_rag_prompt
from src.generation.llm import generate
from src.generation.parser import parse_answer


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_config(path: Path = None) -> Dict[str, Any]:
    path = path or (_project_root() / "config.toml")
    defaults = {
        "retrieval": {
            "mode": "bm25",
            "top_k": 5,
            "sparse_top_n": 20,
            "dense_top_n": 20,
            "fusion_k": 60,
            "rerank": False,
            "rerank_top_n": 5,
        },
        "llm": {"model": "llama-3.3-70b-versatile", "max_tokens": 512, "temperature": 0.2},
    }
    if not path.exists():
        return defaults
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {
        "retrieval": {**defaults["retrieval"], **(data.get("retrieval") or {})},
        "llm": {**defaults["llm"], **(data.get("llm") or {})},
    }


# Retrieval indices, loaded once (see load_indices) and shared across requests.
_bm25_state: Optional[Tuple[Any, List[dict], dict]] = None
_dense_state: Optional[Tuple[Any, List[dict], dict]] = None


def load_indices(mode: str = "bm25") -> None:
    """Build/load retrieval indices. Call once at process startup
    (see scripts/run_server.py) to avoid concurrent lazy-build races
    under FastAPI's threaded request handling."""
    global _bm25_state, _dense_state
    _bm25_state = build_index()
    if mode == "hybrid":
        _, corpus, id_to_passage = _bm25_state
        dense_index = load_dense_index(_project_root() / "artifacts" / "dense_index", corpus)
        _dense_state = (dense_index, corpus, id_to_passage)
    else:
        _dense_state = None


def _ensure_loaded(mode: str) -> None:
    if _bm25_state is None:
        load_indices(mode)


def rag_query(
    question: str,
    top_k: int = None,
    max_tokens: int = None,
    config: Dict[str, Any] = None,
) -> Tuple[List[Tuple[int, str, float]], str]:
    cfg = config or _load_config()
    r_cfg = cfg.get("retrieval", {})
    llm_cfg = cfg.get("llm", {})
    top_k = top_k or r_cfg.get("top_k", 5)
    max_tokens = max_tokens or llm_cfg.get("max_tokens", 512)
    temperature = llm_cfg.get("temperature", 0.2)
    model = llm_cfg.get("model", "llama-3.3-70b-versatile")
    mode = r_cfg.get("mode", "bm25")
    do_rerank = r_cfg.get("rerank", False)

    _ensure_loaded(mode)
    bm25, corpus, id_to_passage = _bm25_state

    # Fetch more than top_k candidates whenever a later stage (fusion or
    # rerank) needs a larger pool to work with.
    sparse_top_n = r_cfg.get("sparse_top_n", 20)
    fetch_k = sparse_top_n if (mode == "hybrid" or do_rerank) else top_k

    sparse_results = bm25_search(question, bm25, corpus, id_to_passage, k=fetch_k)

    if mode == "hybrid" and _dense_state is not None:
        dense_index, d_corpus, d_id_to_passage = _dense_state
        dense_results = dense_search(
            question, dense_index, d_corpus, d_id_to_passage, k=r_cfg.get("dense_top_n", 20)
        )
        candidates = rrf_fuse(sparse_results, dense_results, k=r_cfg.get("fusion_k", 60))
    else:
        candidates = sparse_results

    if do_rerank:
        passages = voyage_rerank(question, candidates, top_n=r_cfg.get("rerank_top_n", top_k))[:top_k]
    else:
        passages = candidates[:top_k]

    context = format_passages_as_context(passages)
    prompt = build_rag_prompt(context=context, question=question)
    raw = generate(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    answer = parse_answer(raw)
    return passages, answer
```

- [ ] **Step 4: Run pipeline tests to verify they pass**

Run: `pytest tests/test_rag_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing server-startup test**

Create `tests/test_run_server_startup.py`:

```python
import scripts.run_server as run_server


def test_startup_loads_indices_with_configured_mode(monkeypatch):
    called = {}

    def fake_load_indices(mode):
        called["mode"] = mode

    monkeypatch.setattr(run_server, "load_indices", fake_load_indices)
    monkeypatch.setattr(run_server, "_cfg", {"retrieval": {"mode": "hybrid"}})
    run_server._load_indices_on_startup()
    assert called["mode"] == "hybrid"
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `pytest tests/test_run_server_startup.py -v`
Expected: FAIL — `AttributeError: module 'scripts.run_server' has no attribute '_load_indices_on_startup'`

- [ ] **Step 7: Update `scripts/run_server.py`**

In `scripts/run_server.py`, change the import line and add the startup handler. Replace:

```python
from src.pipeline.rag import rag_query
```

with:

```python
from src.pipeline.rag import rag_query, load_indices, _load_config
```

Then, immediately after `app = FastAPI(title="RAG API")`, add:

```python
_cfg = _load_config()


@app.on_event("startup")
def _load_indices_on_startup():
    load_indices(mode=_cfg.get("retrieval", {}).get("mode", "bm25"))
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `pytest tests/test_run_server_startup.py -v`
Expected: PASS

- [ ] **Step 9: Run the full test suite**

Run: `pytest -v`
Expected: All tests from Tasks 1-6 PASS.

- [ ] **Step 10: Manual smoke test in `bm25` mode (no new API keys needed)**

Run: `python scripts/run_rag.py -q "What is medulloblastoma?"`
Expected: Same behavior as before this plan — an answer plus top passages printed to stderr.

- [ ] **Step 11: Commit**

```bash
git add src/pipeline/rag.py scripts/run_server.py tests/test_rag_pipeline.py tests/test_run_server_startup.py
git commit -m "feat: wire hybrid retrieval and reranking into the RAG pipeline"
```

---

### Task 7: Retrieval Evaluation Harness (Phase 1 Acceptance Gate)

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/metrics.py`
- Create: `scripts/evaluate_retrieval.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_evaluate_retrieval.py`

**Interfaces:**
- Consumes: `src.data.loaders.load_qa` (existing), `src.retrieval.sparse.build_index`/`search` (existing), `src.retrieval.dense.load_index`/`dense_search` (Task 2), `src.retrieval.fusion.rrf_fuse` (Task 1), `src.retrieval.rerank.rerank` (Task 4).
- Produces: `recall_at_k`, `mrr_at_k`, `ndcg_at_k` (pure metric functions) and `evaluate_mode(qa_pairs, retrieve_fn) -> Dict[str, float]`. Not consumed by other tasks in this plan — this is the final deliverable and Phase 1's acceptance gate.

Uses the test split's gold `relevant_passage_ids` (already in the repo, never previously used) to score BM25-only vs. hybrid vs. hybrid+rerank retrieval on the same questions, and answers the only question that actually matters: did this upgrade help?

- [ ] **Step 1: Write the failing metrics tests**

Create `src/eval/__init__.py` (empty file).

Create `tests/test_metrics.py`:

```python
from src.eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def test_recall_at_k_counts_gold_hits():
    assert recall_at_k([1, 2, 3], [2, 4]) == 0.5


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k([1, 2], []) == 0.0


def test_mrr_at_k_uses_first_hit_rank():
    assert mrr_at_k([5, 6, 2], [2]) == 1.0 / 3


def test_mrr_at_k_no_hit_returns_zero():
    assert mrr_at_k([5, 6], [2]) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2], [1, 2]) == 1.0


def test_ndcg_at_k_lower_when_relevant_ranked_lower():
    high = ndcg_at_k([1, 3], [1])
    low = ndcg_at_k([3, 1], [1])
    assert high == 1.0
    assert low < high
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.metrics'`

- [ ] **Step 3: Implement the metrics**

Create `src/eval/metrics.py`:

```python
import math
from typing import List


def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    return len(relevant_set & retrieved_set) / len(relevant_set)


def mrr_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Write the failing evaluation-harness test**

Create `tests/test_evaluate_retrieval.py`:

```python
from scripts.evaluate_retrieval import evaluate_mode


def test_evaluate_mode_averages_metrics_across_questions():
    qa_pairs = [
        {"question": "q1", "relevant_passage_ids": [10]},
        {"question": "q2", "relevant_passage_ids": [20]},
    ]

    def fake_retrieve(question):
        return [10] if question == "q1" else [99]

    result = evaluate_mode(qa_pairs, fake_retrieve)
    assert result["recall_at_k"] == 0.5
    assert result["mrr_at_k"] == 0.5
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `pytest tests/test_evaluate_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.evaluate_retrieval'`

- [ ] **Step 7: Implement the evaluation harness**

Create `scripts/evaluate_retrieval.py`:

```python
#!/usr/bin/env python3
"""Evaluate retrieval quality (Recall/MRR/nDCG) for bm25 vs hybrid vs
hybrid+rerank against the gold relevant_passage_ids in the test QA split.

This is the Phase 1 acceptance gate: hybrid+rerank should show a
measurable improvement over the bm25 baseline before Phase 1 is
considered done.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_qa
from src.eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def evaluate_mode(
    qa_pairs: List[dict],
    retrieve_fn: Callable[[str], List[int]],
) -> Dict[str, float]:
    recalls, mrrs, ndcgs = [], [], []
    for qa in qa_pairs:
        retrieved_ids = retrieve_fn(qa["question"])
        gold_ids = qa["relevant_passage_ids"]
        recalls.append(recall_at_k(retrieved_ids, gold_ids))
        mrrs.append(mrr_at_k(retrieved_ids, gold_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, gold_ids))
    n = len(qa_pairs)
    return {
        "recall_at_k": sum(recalls) / n,
        "mrr_at_k": sum(mrrs) / n,
        "ndcg_at_k": sum(ndcgs) / n,
    }


def _bm25_retrieve_fn(bm25, corpus, id_to_passage, k):
    from src.retrieval.sparse import search

    def _fn(question):
        return [pid for pid, _, _ in search(question, bm25, corpus, id_to_passage, k=k)]

    return _fn


def _hybrid_retrieve_fn(
    bm25, corpus, id_to_passage, dense_index, k, sparse_top_n, dense_top_n, fusion_k, rerank_top_n=None
):
    from src.retrieval.sparse import search as bm25_search
    from src.retrieval.dense import dense_search
    from src.retrieval.fusion import rrf_fuse
    from src.retrieval.rerank import rerank as voyage_rerank

    def _fn(question):
        sparse = bm25_search(question, bm25, corpus, id_to_passage, k=sparse_top_n)
        dense = dense_search(question, dense_index, corpus, id_to_passage, k=dense_top_n)
        fused = rrf_fuse(sparse, dense, k=fusion_k)
        if rerank_top_n:
            candidate_pool = fused[: max(sparse_top_n, dense_top_n)]
            reranked = voyage_rerank(question, candidate_pool, top_n=rerank_top_n)
            return [pid for pid, _, _ in reranked[:k]]
        return [pid for pid, _, _ in fused[:k]]

    return _fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against gold relevant_passage_ids.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N QA pairs (cost control)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    qa_pairs = load_qa(split="test")
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    from src.retrieval.sparse import build_index
    from src.retrieval.dense import load_index as load_dense_index

    bm25, corpus, id_to_passage = build_index()
    results = {"bm25": evaluate_mode(qa_pairs, _bm25_retrieve_fn(bm25, corpus, id_to_passage, k=args.top_k))}

    dense_index_path = project_root / "artifacts" / "dense_index"
    if (dense_index_path / "index.faiss").exists():
        dense_index = load_dense_index(dense_index_path, corpus)
        results["hybrid"] = evaluate_mode(
            qa_pairs,
            _hybrid_retrieve_fn(
                bm25, corpus, id_to_passage, dense_index,
                k=args.top_k, sparse_top_n=20, dense_top_n=20, fusion_k=60,
            ),
        )
        results["hybrid_rerank"] = evaluate_mode(
            qa_pairs,
            _hybrid_retrieve_fn(
                bm25, corpus, id_to_passage, dense_index,
                k=args.top_k, sparse_top_n=20, dense_top_n=20, fusion_k=60, rerank_top_n=args.top_k,
            ),
        )
    else:
        print(
            "No dense index found at artifacts/dense_index — run scripts/build_embeddings.py "
            "first to include hybrid/hybrid_rerank modes.",
            file=sys.stderr,
        )

    print(json.dumps(results, indent=2))

    out_dir = project_root / "artifacts" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `pytest tests/test_evaluate_retrieval.py -v`
Expected: PASS

- [ ] **Step 9: Run the full test suite**

Run: `pytest -v`
Expected: All tests from Tasks 1-7 PASS.

- [ ] **Step 10: Commit**

```bash
git add src/eval/__init__.py src/eval/metrics.py scripts/evaluate_retrieval.py tests/test_metrics.py tests/test_evaluate_retrieval.py
git commit -m "feat: add retrieval evaluation harness (Recall/MRR/nDCG)"
```

- [ ] **Step 11: Run the real acceptance gate (manual, costs API calls)**

This step requires `OPENAI_API_KEY` set and, to see the `hybrid_rerank` row, `VOYAGE_API_KEY` set plus the dense index built.

```bash
python scripts/build_embeddings.py
python scripts/evaluate_retrieval.py --limit 100 --top-k 5
```

Expected: A `bm25` row always appears. If `artifacts/dense_index/` exists, `hybrid` and `hybrid_rerank` rows also appear, and their `recall_at_k`/`mrr_at_k`/`ndcg_at_k` should be **equal to or better than** the `bm25` row — if not, do not enable `mode = "hybrid"` or `rerank = true` in the shipped `config.toml` default; investigate first. This is the Phase 1 acceptance gate from spec §3.4.

- [ ] **Step 12: Once satisfied, run without `--limit` for the full test-set numbers and save them**

```bash
python scripts/evaluate_retrieval.py --top-k 5
```

Note: `artifacts/` is git-ignored (see `.gitignore`), so `artifacts/eval/retrieval_results.json` will not be committed — that's expected; it's a local/Docker-volume artifact, not a repo artifact. Keep the printed JSON output as the record of the Phase 1 acceptance result.
