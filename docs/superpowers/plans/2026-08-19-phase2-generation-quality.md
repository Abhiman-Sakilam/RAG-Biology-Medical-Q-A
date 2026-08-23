# Phase 2 Implementation Plan — Generation Quality & Citations

**Date:** 2026-08-19  
**Status:** Ready for review  
**Baseline:** Phase 1 complete (BM25 Recall: 0.502, MRR: 0.833, nDCG: 0.735)  
**Goal:** Improve generation quality via chunking, query rewriting, citations, and guardrails

---

## 1. Overview

Phase 2 builds on Phase 1's retrieval baseline to address generation quality:

| Aspect | Current | Target |
|--------|---------|--------|
| **Chunking** | All passages used as-is (some 4K+ words) | Targeted: long passages only, split at sentence boundaries |
| **Query** | Raw question verbatim | Rewritten: HyDE hypothesis + BM25 keyword expansion |
| **Citations** | LLM shown passages but not required to cite them | Enforced: every claim requires `[Passage N]` marker |
| **Grounding** | No check for hallucination | Guardrail: 0–1 confidence score, flag if below threshold |
| **Eval** | Retrieval only (Recall/MRR/nDCG) | Extended: answer faithfulness via LLM-as-judge |

---

## 2. Architecture

### 2.1 Chunking Module

**File:** `src/data/chunking.py`

- **Function:** `chunk_passages(corpus, threshold=500)` → List of chunks with parent-id mapping
- **Logic:**
  - Measure word count per passage (split on whitespace)
  - Passages ≤ threshold: keep as-is (id = `passage_id`)
  - Passages > threshold: split at sentence boundaries (regex: `[.!?]\s+`), keeping 50-word overlap
  - Chunk id format: `f"{parent_id}::{chunk_idx}"` (e.g., `12345::0`, `12345::1`)
- **Invariant:** Every chunk maps back to its parent id via `parent_id_of(chunk_id)`
  - Used by: BM25 tokenization, dense search, citation parsing, eval scoring
  - Must be in one place; called by retrieval modules and eval harness

**Impact on existing code:**
- Corpus loader picks up chunks instead of passages (happens before BM25/dense indexing)
- Retrieval still returns `(chunk_id, text, score)` tuples
- Before showing to user or scoring against eval gold, map chunk_id → parent_id via `parent_id_of()`

### 2.2 Query Rewriting Module

**File:** `src/query/rewrite.py`

- **Function:** `rewrite_query(question, llm_client)` → Tuple[str, str]
  - Returns: (hypothesis_passage, keyword_expansions)
- **Logic:**
  - One LLM call (via existing `src/generation/llm.py` client, cost ~0.5¢)
  - System prompt: "You are a biomedical search expert. Given a question, generate a hypothetical passage that would answer it, and list 2–3 synonym/acronym expansions."
  - Parse response: extract hypothesis (first paragraph) and expansions (bulleted list)
- **Config:** `[rewrite] enabled = false` (default, opt-in)
- **Fallback:** On LLM error, return `(question, "")` silently (enhancement, not hard dependency)

**Impact on retrieval:**
- Dense search uses `hypothesis_passage` as query instead of raw question
- BM25 search appends ` <expansions>` to the question (keyword boost)
- Example:
  ```
  Question: "What is a heart attack?"
  Hypothesis: "A heart attack occurs when blood flow to the heart muscle is blocked..."
  Expansions: "myocardial infarction, MI, acute coronary syndrome"
  
  BM25 query: "What is a heart attack? myocardial infarction MI acute coronary syndrome"
  Dense query: "A heart attack occurs when blood flow to the heart muscle is blocked..."
  ```

### 2.3 Citation Extraction & Validation

**File:** `src/generation/parser.py` (extend existing)

- **Function:** `extract_and_validate_citations(answer, retrieved_passage_ids)` → (answer_cleaned, citations)
- **Logic:**
  - Regex: find all `[Passage N]` or `[P N]` markers in answer
  - Extract passage indices
  - Validate: only keep citations where `N` is in `retrieved_passage_ids` (prevent hallucinated references)
  - Strip invalid citations and log them
  - Return cleaned answer + citation list: `[(marker="[Passage 1]", passage_id=123), ...]`

**Impact on prompts:**
- System prompt updated: *"You must cite every factual claim with `[Passage N]` before the claim."*
- User prompt includes: *"You have these passages: [list]. Cite them as [Passage 1], [Passage 2], etc."*

**API response** (`scripts/run_server.py::QueryResponse`):
```python
citations: List[{
    "marker": "[Passage 1]",
    "passage_id": int,
    "text": str,
    "score": float
}]
```

### 2.4 Groundedness Guardrail

**File:** `src/generation/groundedness.py`

- **Function:** `score_groundedness(answer, citations, llm_client)` → float (0–1)
- **Logic:**
  - Compile cited passages into a context window
  - LLM call: *"Rate how grounded this answer is in these passages. Return a 0–1 score."*
  - Parse response as float
  - Cost: ~0.3¢ per answer
- **Config:** `[guardrail] groundedness_enabled = false, groundedness_threshold = 0.5` (default)
- **Action:** Return score in API response; flag if below threshold (do NOT block or retry; keep latency bounded)

**API response** (`scripts/run_server.py::QueryResponse`):
```python
groundedness_score: float
groundedness_flagged: bool  # True if score < threshold
```

---

## 3. Task Breakdown

### Task 1: Chunking Module & Tests
**Scope:** `src/data/chunking.py` + tests  
**Output:** 
- `chunk_passages(corpus, threshold=500)` 
- `parent_id_of(chunk_id)` 
- Tests for sentence-boundary splits, parent-id mapping, edge cases (no boundaries, all short, all long)

### Task 2: Integrate Chunking into Corpus & Retrieval
**Scope:** Update corpus loader, BM25 indexing, dense indexing  
**Output:**
- Corpus loader returns chunks (not raw passages)
- BM25/dense indexing happens on chunks
- Eval harness maps chunk_id → parent_id before scoring gold labels
- Tests: verify eval scores still align with Phase 1 baseline (should be identical with default threshold=500, since most passages are <500 words)

### Task 3: Query Rewriting Module & Tests
**Scope:** `src/query/rewrite.py` + tests  
**Output:**
- `rewrite_query(question, llm_client)` 
- LLM prompt + response parsing
- Fallback on error
- Tests: mock LLM, verify hypothesis extraction, keyword parsing, error handling

### Task 4: Wire Query Rewriting into Pipeline
**Scope:** Update `src/pipeline/rag.py::rag_query()`  
**Output:**
- Call `rewrite_query()` if `[rewrite] enabled = true`
- Pass hypothesis to dense_search, expansions to BM25
- Config-gated (default off)
- Tests: verify rewrite is called/skipped per config, answers don't change when disabled

### Task 5: Citation Module & Tests
**Scope:** `src/generation/parser.py` (extend) + tests  
**Output:**
- `extract_and_validate_citations(answer, passage_ids)` 
- Regex for `[Passage N]` / `[P N]`
- Validation against retrieved set
- Stripped invalid citations + logging
- Tests: mock answers with valid/invalid/malformed citations, verify parsing and validation

### Task 6: Groundedness Guardrail & Tests
**Scope:** `src/generation/groundedness.py` + tests  
**Output:**
- `score_groundedness(answer, citations, llm_client)` → float
- LLM prompt for scoring
- Error handling (return 0.5 on LLM error, don't fail request)
- Tests: mock LLM responses, verify score parsing, threshold logic

### Task 7: Pipeline Integration & API Updates
**Scope:** `src/pipeline/rag.py::rag_query()`, `scripts/run_server.py`  
**Output:**
- Call citation extraction after LLM generation
- Call groundedness scoring if enabled
- Update `QueryResponse` model with citations + groundedness fields
- Config-gated (citation: default off, groundedness: default off)
- Tests: full pipeline integration, verify citations extracted and groundedness scored

### Task 8: Answer Evaluation Harness & Tests
**Scope:** `scripts/evaluate_answers.py` (new)  
**Output:**
- Load test QA split
- For each question:
  - Generate answer via full Phase 2 pipeline
  - LLM-as-judge: rate answer faithfulness against gold answer (0–1 score)
  - Record: question, gold_answer, generated_answer, faithfulness, groundedness_score
- Write to `artifacts/eval/answers_<timestamp>.json`
- Print summary: mean faithfulness, groundedness, per-sample results
- Tests: mock LLM judge responses, verify output format

---

## 4. Global Constraints

- **Backward compatible:** `[rewrite] enabled = false`, `[citation] enforce = false`, `[guardrail] groundedness_enabled = false` means Phase 2 adds zero overhead to Phase 1 pipeline when disabled (exact same behavior).
- **Chunk-to-parent mapping:** Every retrieval hit is a chunk. Before showing to user, scoring against eval gold, or parsing citations, map `chunk_id → parent_id` via `parent_id_of()`. This mapping lives in one place.
- **LLM calls are config-gated & graceful:** Query rewrite, citation, and groundedness all fail soft (return defaults) rather than crash requests.
- **All three ranking sources still work:** Chunks carry scores from BM25, dense, rerank; citation extraction works on any `answer`, not just ones with explicit citations; groundedness scores any answer regardless of citation presence.

---

## 5. Config Schema (Phase 2 additions)

```toml
[rewrite]
enabled = false               # HyDE + BM25 keyword expansion

[citation]
enforce = false               # require inline [Passage N] citations

[guardrail]
groundedness_enabled = false
groundedness_threshold = 0.5  # flag if score < threshold
```

No new env vars. Uses existing `GROQ_API_KEY` or `OPENAI_API_KEY` for LLM calls (rewrite, citation validation, groundedness, answer eval).

---

## 6. Testing Strategy

- **Unit tests:** Chunking sentence splits, citation regex, groundedness prompt parsing
- **Integration tests:** Full pipeline with each Phase 2 component enabled/disabled, verify backward compatibility
- **Answer eval:** Run `scripts/evaluate_answers.py` on test split, measure mean faithfulness before/after Phase 2 (should improve with citations + guardrail enabled)
- **No live LLM calls in CI:** Mock all LLM responses

---

## 7. Success Criteria

1. ✅ All Phase 2 modules implemented and tested (8 tasks, 34+ tests total)
2. ✅ Phase 1 baseline unchanged when Phase 2 disabled (backward compatibility verified)
3. ✅ Answer evaluation harness runs end-to-end on test split
4. ✅ Faithfulness score improves when citations + guardrail are enabled (vs. Phase 1 baseline)
5. ✅ Config schema supports all Phase 2 flags; defaults are off (opt-in)

---

## 8. Estimated Timeline

- Task 1: 2 hrs (chunking module + tests)
- Task 2: 3 hrs (corpus/retrieval integration + extensive testing for eval compatibility)
- Task 3: 2 hrs (query rewriting + tests)
- Task 4: 1 hr (pipeline wiring)
- Task 5: 2 hrs (citation extraction + tests)
- Task 6: 2 hrs (groundedness guardrail + tests)
- Task 7: 2 hrs (full pipeline integration + API updates)
- Task 8: 1.5 hrs (answer eval harness + tests)
- **Reviews & fixes:** 4 hrs (per-task + whole-branch review)
- **Total:** ~19.5 hrs (2.5 working days)

---

## 9. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Chunking breaks eval scoring | High | Task 2 explicitly verifies eval baseline unchanged |
| LLM call costs for rewrite + groundedness | Medium | Config-gated; defaults off; estimate <$0.50 per test run |
| Citation hallucinations not caught | Medium | Validation against retrieved set; log stripped citations |
| Groundedness score doesn't correlate with actual quality | Medium | Use LLM-as-judge for eval; measure against gold answers |

---

## 10. Next Steps

1. **Your review:** Read this plan, ask questions, suggest changes
2. **Approval:** Confirm scope, timeline, constraints
3. **Execution:** Dispatch Task 1 implementer via SDD
4. **Iterative:** Each task reviewed before next begins; final whole-branch review before merge

---

**Ready to proceed?**
