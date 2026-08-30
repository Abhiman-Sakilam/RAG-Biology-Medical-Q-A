# Architecture

How this RAG system works, module by module, and why each choice was made.
For setup, configuration and running instructions, see [README.md](README.md).

---

## Design principles

Three ideas shape the whole codebase.

**1. Every advanced feature is opt-in and independently switchable.**
Hybrid retrieval, reranking, query rewriting, citations and the groundedness guardrail each
sit behind a single flag in `config.toml`, all defaulting to off. The default path is plain
BM25 → LLM: no embedding index, no second API key, no extra latency. Each flag can be turned
on alone and measured against the baseline, so you always know which change bought which
improvement.

**2. One retrieval implementation, shared by production and evaluation.**
`src.pipeline.rag.retrieve_candidates()` is the only place retrieval happens. `rag_query()`
calls it and so does `scripts/evaluate_retrieval.py`. There is no separate "eval retriever"
that can drift from the real one — the metrics describe the shipped system by construction.

**3. Enhancements degrade, they don't fail.**
A reranker that returns garbage, a rewrite call that times out, a groundedness judge that
errors — each is caught and the pipeline continues with the previous, still-correct state
(fusion order, the original question, no score). Losing an optional improvement is always
better than losing the answer.

---

## Data layer

### Corpus and QA splits — `src/data/loaders.py`

The corpus is 40,221 biology/medical passages in
`data/json/text-corpus/{split}-00000-of-00001.json`, each record `{id, passage}`. The same
data is mirrored as CSV and Parquet under `data/csv/` and `data/parquet/`; JSON is what the
code reads.

`load_corpus(split, data_dir, threshold)` reads the JSON, normalizes `passage` → `text`,
runs chunking, then restores the `passage` key and adds `parent_id`. Every downstream
consumer therefore sees a uniform `{id, passage, parent_id}` record whether or not the
passage was split.

`load_qa(split)` reads `data/json/question-answer-passages/` — records of
`{question, answer, id, relevant_passage_ids}`. The 707-pair test split is the gold set the
evaluation harness scores against.

### Chunking — `src/data/chunking.py`

**What.** `chunk_passages(corpus, threshold=500)` leaves passages of ≤500 words untouched and
splits longer ones into overlapping chunks with ids of the form `{parent_id}::{chunk_idx}`.

**Why.** Retrieval precision and prompt budget both suffer when a single retrieved unit is
enormous — a 3,000-word passage buries the two relevant sentences and eats the entire context
window. Chunking makes the retrieval unit closer to the size of an actual answer. The 500-word
default was chosen so that most real passages in this corpus don't split at all, preserving
the original Phase 1 retrieval behavior; the mechanism is there for the long tail.

**How.** Text is split at sentence boundaries (a lookbehind on `[.!?]` followed by whitespace,
so terminal punctuation stays attached to its sentence), then sentences are greedily grouped
into ~500-word chunks. Roughly 50 words are carried from the end of one chunk into the start
of the next, so a fact spanning a chunk boundary still appears whole in at least one chunk.

Three edge cases the implementation handles explicitly:

- **Overlap larger than the budget.** A fixed 50-word overlap exceeds the whole budget for
  small thresholds, which drove the word-splitter's step size to 1 and emitted one chunk per
  word. `_effective_overlap()` caps overlap at `threshold // 2`, keeping every step at least
  half a chunk and the chunk count linear in input size.
- **Overlap consuming the entire chunk.** `_overlap_tail()` never returns every sentence — it
  always drops at least the first — otherwise a chunk would be duplicated verbatim into its
  successor and chunks would grow without bound.
- **A single sentence longer than the budget.** Cannot be grouped, so it falls back to
  `_split_by_words()` rather than producing one wildly oversized chunk.

Text with no sentence boundaries at all falls back to raw word-count splitting.

**Ids.** `parent_id_of(chunk_id)` recovers the original passage id from either a chunked id
(`"1234::2"`) or a bare one. This is what lets evaluation score chunk-level retrieval against
passage-level gold labels. Because ids may be `int` *or* `str`, every id-carrying signature in
the codebase is typed `Union[int, str]` — narrowing it to `int` is a real bug.

### Lookup — `src/data/lookup.py`

`build_lookup(corpus)` builds `{id: text}` so retrievers, which return positions or ids, can
resolve text in O(1) without scanning the corpus.

---

## Retrieval layer

### Sparse: BM25 — `src/retrieval/sparse.py`

**What.** `BM25Okapi` (from `rank_bm25`) over the chunked corpus. This is the default and only
always-on retriever.

**Why BM25.** It is a strong, well-understood lexical baseline that needs no model, no API key,
no index build, and no GPU. In a medical corpus that matters more than usual: queries and
documents share precise, low-frequency terminology ("medulloblastoma", "CYP2D6"), and exact
term matching on rare terms is exactly what BM25's IDF weighting rewards. It is also the honest
baseline — any dense or neural addition has to beat it to justify its cost.

**How.** `build_index()` tokenizes each chunk with `re.findall(r"\w+", text.lower())` and builds
the index; `search()` tokenizes the query the same way, scores the corpus, and returns the top-k
as `(id, text, score)` triples. The tokenizer is deliberately simple — no stemming, no stopword
list — because domain terms are the signal and aggressive normalization tends to damage them.

The index is built once per process (see *Index lifecycle* below), not per request.

### Dense: embeddings + FAISS — `src/retrieval/dense.py`

**What.** Passages are embedded with OpenRouter's `liquid/lfm2.5-embedding-350m` and searched
with a FAISS `IndexFlatIP`.

**Why.** BM25 cannot match a query to a passage that answers it in different words. Dense
retrieval scores by meaning, so "what causes high blood sugar" can retrieve a passage about
hyperglycemia pathophysiology that shares almost no vocabulary with the query. Dense and sparse
fail in different ways, which is precisely what makes fusing them worthwhile.

**Why `IndexFlatIP`.** Exact inner-product search — no approximation, no training step, no
tuning. Vectors are L2-normalized on the way in (`faiss.normalize_L2`), which makes inner
product equal to cosine similarity. At 40k passages, exact search is fast enough that trading
accuracy for an approximate index (IVF, HNSW) would be premature optimization.

**Rate limiting.** Embedding the corpus is the expensive part, and the free tier enforces a
tokens-per-minute ceiling. `embed_texts()` batches 10 texts per request, sleeps 2s between
batches, tracks tokens consumed in a rolling minute window, and pauses when usage crosses a
safety margin below the limit. The window resets when a minute genuinely elapses — without
that reset the counter grew forever and the throttle could never fire again after the first
trip. On a 429 the request retries up to 3 times with linear backoff.

**Persistence.** The index is built ahead of time by `scripts/build_embeddings.py`, never at
request time. `save_index()` writes `index.faiss` plus `corpus_ids.json` (the exact id order
the vectors were added in); the build script also writes `corpus.sha256` so a rebuild is
skipped when the corpus file is unchanged.

**Staleness is an error, not a silent wrong answer.** `load_index()` compares the saved id
order against the current corpus and raises if they differ. A FAISS index maps *positions* to
vectors — if the corpus changed, position `i` no longer means the same passage, and every
result would be silently mislabeled. A hard failure telling you to re-run `build_embeddings.py`
is the only safe behavior.

### Fusion: Reciprocal Rank Fusion — `src/retrieval/fusion.py`

**What.** `rrf_fuse(sparse, dense, k=60)` merges two ranked lists:

```
score(doc) = Σ  1 / (k + rank_in_list)
```

**Why RRF.** BM25 scores are unbounded relevance sums; cosine similarities live in [-1, 1].
They are not comparable, and normalizing them requires per-query calibration that is fragile
and corpus-specific. RRF sidesteps the problem entirely by discarding scores and using only
*rank*, which is comparable across any retrievers. It needs no training data and no tuning,
and it rewards documents that both retrievers liked — the strongest available signal that a
passage is genuinely relevant.

**Why `k = 60`.** The standard value from the original RRF paper. `k` damps the contribution
of top ranks: with `k = 60`, rank 1 scores 1/61 and rank 2 scores 1/62 — close together, so a
single retriever's confident top hit cannot dominate agreement between the two. A small `k`
would let either retriever unilaterally decide the winner.

### Reranking — `src/retrieval/rerank.py`

**What.** `rerank(query, candidates, top_n)` sends the fused candidate pool to OpenRouter's
`nvidia/llama-nemotron-rerank-vl-1b-v2:free` and re-orders it by the returned ranking. Used
only when `[retrieval] rerank = true`.

**Why.** First-stage retrievers score the query and document independently — BM25 by term
overlap, dense by two separately-computed vectors. A reranker reads the query and document
*together*, so it can judge whether a passage actually answers this question rather than
merely resembling it. That is far more expensive per document, which is why it runs on ~20
fused candidates rather than 40,000 passages. This retrieve-then-rerank split is the standard
way to buy cross-attention accuracy at first-stage cost.

**How.** The model is prompted with the query and numbered candidate snippets (first 200 chars
each) and asked to return the top indices, one per line, at `temperature: 0`. The returned
order becomes the new ranking, with synthetic descending scores `(n - rank) / n` so the tuple
shape stays uniform for downstream code.

**Failure handling, in layers.** Reranking is an optimization, so no failure mode may cost the
user their answer:

| Failure | Behavior |
|---------|----------|
| Malformed 200 response | `_extract_content()` returns `""`, logs, does not raise |
| Model ignores the output format | `_parse_ranked_indices()` scans each line for a leading integer instead of requiring a bare number |
| Out-of-range or duplicate indices | Filtered and de-duplicated |
| No parseable indices at all | Returns the fusion order unchanged |
| HTTP error, timeout, or unusable shape | Caught in `retrieve_candidates()`; logs a warning and returns unreranked candidates |

---

## Query understanding

### HyDE + keyword expansion — `src/query/rewrite.py`

**What.** When `[rewrite] enabled = true`, `rewrite_query()` asks the LLM for two things:
a *hypothetical passage* that would answer the question, and 2–3 synonym/expansion keywords.

**Why (the HyDE idea).** Questions and passages are different kinds of text. "What causes
X?" is short and interrogative; the passage answering it is long and declarative. Embedding
a question and embedding a passage puts them in systematically different regions of the
vector space. A hypothetical *answer* — even a factually imperfect one — is written in the
register and vocabulary of the target passages, so its embedding lands much closer to real
answers. You search with a fake answer to find the real one.

**Why the two rewrites go to different retrievers.** In `retrieve_candidates()`, the
hypothesis is used as the **dense** query (it helps precisely because the embedding space
cares about phrasing and register) while the original question plus expansion keywords is
used as the **sparse** query (BM25 wants more matchable terms, and dumping a whole synthetic
paragraph into it would drown the actual query terms in noise). Each retriever gets the form
of the query it can actually use.

**Failure handling.** Empty response, unparseable format, or an exception all fall back to
the original question with no expansions, and the pipeline proceeds normally.

---

## Generation layer

### Provider client — `src/generation/llm.py`

`_get_client()` returns an `openai.OpenAI` client pointed at whichever provider is configured:
Groq (via its OpenAI-compatible endpoint at `api.groq.com/openai/v1`) if `GROQ_API_KEY` is
set, otherwise OpenAI or any endpoint given by `OPENAI_BASE_URL`. One client class covers all
of them because the API surface is identical.

`resolve_model()` is the single place a model id is decided — `config.toml` value, else
`GROQ_MODEL`, else `OPENAI_MODEL`, else the default. Generation, query rewriting and
groundedness scoring all route through it, so no caller can hardcode a model id that the
configured provider cannot serve. (That was a real bug: helper modules defaulting to a model
the active key had no access to.)

### Prompts — `src/prompts/`

`templates.py` holds `SYSTEM_TEMPLATE` ("answer based only on the provided context; if the
context doesn't contain enough information, say so"), `USER_TEMPLATE` (context + question),
and `CITATION_INSTRUCTION`. `build_rag_prompt()` assembles them, appending the citation
instruction only when `require_citations` is set.

That coupling is deliberate: citation extraction can only find markers the model was actually
asked to write. Enabling extraction without enabling the instruction would silently produce
zero citations forever.

`formatter.py`'s `format_passages_as_context()` renders passages as
`[Passage 1] … [Passage N]`, capped at 6,000 characters. When the budget runs out mid-passage
it truncates that passage and stops; when the budget is already spent it breaks rather than
appending an untrimmed passage that would blow past the cap. The **1-based ordinal label is
the contract** with the citation parser — the number the model writes is a position in the
context, never a corpus id.

### Answer parsing and citations — `src/generation/parser.py`

`parse_answer()` trims whitespace, collapses 3+ newlines to a paragraph break, and truncates
at 2,000 characters.

`extract_and_validate_citations(answer, ordered_passage_ids)` finds `[Passage N]` / `[P N]`
markers and resolves each ordinal to the real id of the passage that occupied that position.
This is why `ordered_passage_ids` must be passed in exactly the order the passages were
rendered — the mapping is positional.

A marker is **valid** when its ordinal addresses a passage that was actually retrieved. Markers
outside that range are hallucinated: they are stripped from the answer text and
`_tidy_whitespace()` repairs the spacing left behind (removing the space before a now-adjacent
period, collapsing double spaces). Duplicates are preserved in the citation list, since citing
the same passage twice is legitimate.

### Groundedness guardrail — `src/generation/groundedness.py`

**What.** When `[guardrail] groundedness_enabled = true`, an LLM judge scores 0.0–1.0 how well
each claim in the answer is supported by its evidence passages. Scores below
`groundedness_threshold` (default 0.5) set `groundedness_flagged`.

**Why.** Retrieval quality metrics say nothing about whether the *answer* stayed inside the
retrieved context. This is the check on the last step, where hallucination actually happens.

**What the judge sees.** Full passage text, not citation markers — a grader shown only
`[Passage 2]` has no way to verify a claim and can only guess from the answer's shape. Evidence
defaults to the cited passages, falling back to all retrieved passages when citations are
disabled or absent, since that is the context the answer was written from. Each passage is
truncated to 1,500 characters so a long tail cannot push the judge request over the provider's
size limit.

**Fallbacks.** No evidence at all → `0.0` (an answer with nothing to be grounded in is
ungrounded — more honest than asking the judge to guess). LLM error or missing client → `0.5`,
neutral, and the request still succeeds. `_parse_score_from_response()` accepts a bare decimal,
a labeled score, a bare integer, or a 0–100 value, clamping to [0, 1]; anything unparseable
returns 0.5 with a warning.

---

## Orchestration — `src/pipeline/rag.py`

### Configuration

`_load_config()` reads `config.toml` with `tomllib` (Python 3.11+, falling back to `tomli`)
and merges it **per section** over built-in defaults. Per-section merging matters: a user who
sets only `[retrieval] top_k` still gets every other retrieval default, rather than an empty
dict. A missing `config.toml` yields defaults, so the system runs uncommitted out of the box.

### Index lifecycle

`load_indices(mode)` builds the BM25 index — and, in hybrid mode, loads the persisted dense
index — into module-level state. `scripts/run_server.py` calls it from a FastAPI `lifespan`
hook, before the first request is served.

**Why eagerly.** FastAPI runs sync endpoints in a thread pool. With lazy loading, several
concurrent first requests would each see `_bm25_state is None` and start building the index
simultaneously — minutes of duplicated work and a memory spike. Building once at startup
removes the race. `_ensure_loaded()` still provides lazy loading for CLI and test callers,
where there is no concurrency.

### `retrieve_candidates(question, r_cfg, hypothesis, expansions)`

The shared retrieval core:

1. Choose queries — hypothesis for dense (if rewriting ran), question + expansions for sparse.
2. Fetch `sparse_top_n` from BM25 when fusing or reranking, otherwise just `top_k`. Fetching
   only `top_k` before a rerank would leave the reranker nothing to re-order.
3. In hybrid mode, fetch `dense_top_n` and fuse with RRF; otherwise use the sparse list.
4. If reranking, rerank to `max(top_k, rerank_top_n)` — never fewer candidates than the final
   cut needs.
5. Return the top `top_k`.

### `rag_query_full(question, ...)`

The full pipeline: optional rewrite → `retrieve_candidates()` → format context → build prompt
→ generate → parse → extract citations → optional groundedness score. Returns a dict of
`passages`, `answer`, `citations`, `groundedness_score`, `groundedness_flagged`.

`rag_query()` wraps it and returns just `(passages, answer)`, preserving the original
two-value contract for existing CLI and test callers.

Ordering inside this function is load-bearing: citations must be extracted *before*
groundedness is scored, because the judge grades the cleaned answer against the passages the
answer actually cited.

---

## Serving layer

### API — `scripts/run_server.py`

FastAPI serves the built React app at `/` and `/assets`, and the pipeline at `POST /query`.
Pydantic models define the response; `passage_id` is typed `int | str` because chunked ids are
strings — narrowing it to `int` would reject every chunked passage at serialization time.

Provider errors are translated into statuses a caller can act on rather than a blanket 500:

| Condition | Status | Rationale |
|-----------|--------|-----------|
| Empty question | 400 | Client's malformed input |
| Missing/invalid API key (`ValueError`) | 503 | Operator configuration, not a code fault |
| Provider 413 | 413 | Prompt too large — the message tells you to lower `top_k` |
| Provider 429 | 429 | Rate limit, retryable |
| Other provider error | 502 | Upstream failure, not ours |
| `APIConnectionError` | 504 | Upstream unreachable |

### Frontend — `frontend/`

React 18 + Vite 5, no UI framework. `App.jsx` holds question/result/loading state; `api.js`
POSTs to `/query` and surfaces the FastAPI `detail` field as the error message, so the
actionable server-side messages above reach the user instead of a generic failure. `QueryForm`,
`Status` and `ResultCard` (answer plus a collapsible passage list) are the three components.

In production one process serves both UI and API from port 8060. In development, Vite serves
on 5173 and proxies `/query` to 8060, so dev and prod hit the same backend.

The UI does not yet render `citations` or `groundedness_score`, though `/query` returns both.

### Docker — `setup/`

A two-stage build: `node:20-alpine` builds the frontend, then `python:3.11-slim` installs
requirements, copies `src/`, `scripts/` and the built `frontend/dist`, and runs the server.
Only the built assets reach the final image — no `node_modules`.

`docker-compose.yaml` mounts `../data` and `../artifacts` as volumes. `artifacts/` is excluded
from the image by `.dockerignore` and provided only by the mount, so the dense index and eval
output survive rebuilds without being baked into the image.

---

## Evaluation

### Metrics — `src/eval/metrics.py`

All four take `k` and truncate internally. That truncation is deliberate: callers used to be
trusted to pass exactly `k` ids, and passing more silently produced `recall@len(retrieved)`
reported under an `@k` name.

| Metric | Measures |
|--------|----------|
| `recall_at_k` | Fraction of gold passages appearing in the top k. Answers "did we find the evidence?" |
| `max_recall_at_k` | The best score a *perfect* retriever could achieve — `min(k, n_gold) / n_gold`. |
| `mrr_at_k` | Reciprocal rank of the first relevant hit. Answers "is a good passage at the top?" |
| `ndcg_at_k` | Position-discounted gain over all relevant hits. Answers "is the whole ranking good?" |

**Why `max_recall_at_k` exists.** Test questions average ~8.7 gold passages, so recall@5 is
mathematically capped at roughly 5/8.7 ≈ 0.57 no matter how perfect the retriever is. Reporting
0.503 alone reads like a failing system; reporting it against its ceiling (`recall_vs_ceiling`)
shows retrieval finding most of what it possibly could.

### Retrieval harness — `scripts/evaluate_retrieval.py`

Runs bm25, hybrid, and hybrid+rerank over the 707-pair test split and writes
`artifacts/eval/retrieval_results.json`. Hybrid rows are skipped with a message when no dense
index exists.

Two properties make the results trustworthy:

- **Same code path.** Retrieval goes through `retrieve_candidates()` — the production function.
- **Chunk-aware scoring.** Retrieved ids are mapped through `parent_id_of()` before comparison,
  because gold labels are passage-level while retrieval is chunk-level. Without this, a
  correctly retrieved chunk `1234::0` would count as a miss against gold id `1234`.

`--limit N` caps the number of questions, which matters when rerank mode makes one API call
per question.

Current BM25 baseline at k=5: **Recall 0.503 · MRR 0.831 · nDCG 0.737**.

### Answer harness — `scripts/evaluate_answers.py`

Generates answers through the full pipeline and scores them against gold answers with an
LLM-as-judge faithfulness rubric, writing timestamped results to `artifacts/eval/`. This
measures the generation half, where retrieval metrics say nothing.

---

## Cross-cutting concerns

### Environment loading — `src/config/env.py`

`load_env()` is called at import time by every module that needs a key (`llm.py`, `dense.py`,
`rerank.py`). Two rules:

- **Idempotent.** A module-level `_loaded` flag means repeated calls are free, so entry points
  need not coordinate about who loads first.
- **Never clobbers the environment.** `load_dotenv(..., override=False)` means an exported
  shell variable always beats the file. Root `.env` is consulted before `setup/.env`.

This is why any script can be run directly without a wrapper that sets up the environment first.

### Tests — `tests/`

147 pytest tests — 140 run offline (network calls are mocked throughout, so no API key is
needed) and 7 `evaluate_answers.py` integration tests skip without a live provider. Coverage
concentrates on the places where bugs are expensive and invisible: chunking edge
cases (overlap larger than budget, oversized single sentences, unbounded growth), rerank
response parsing and every fallback layer, citation ordinal-to-id mapping and hallucinated
marker removal, groundedness score parsing across response formats, metric truncation
semantics, per-section config merging, and server startup index loading.

`tests/test_bugfix_regressions.py` pins previously-fixed defects so they cannot silently return.

---

## Extension points

| To add | Touch | Notes |
|--------|-------|-------|
| A different embedding model | `src/retrieval/dense.py` (`EMBEDDING_MODEL`) | Rebuild the index; dimension is read from the vectors, not hardcoded |
| An approximate FAISS index | `build_dense_index()` | Swap `IndexFlatIP`; keep vectors L2-normalized |
| A different reranker | `src/retrieval/rerank.py` | Keep the `(id, text, score)` return shape and the fallback layers |
| A new retriever | `src/retrieval/`, then `retrieve_candidates()` | `rrf_fuse` merges exactly two lists today |
| Different fusion weighting | `src/retrieval/fusion.py` | Weighted RRF is a small change to the score accumulation |
| A new LLM provider | `src/generation/llm.py` (`_get_client`) | OpenAI-compatible endpoints need only a base URL |
| A new guardrail | `src/generation/`, then `rag_query_full()` | Follow the pattern: config flag, try/except, neutral fallback |
