"""Regressions for four defects that the rest of the suite could not catch.

Each test states the observed failure it locks out, because in every case the
buggy code path was silent: it raised in a layer with no test, or swallowed the
error and returned a plausible-looking fallback.
"""
import os


class TestChunkedIdsSurviveTheApiLayer:
    """Bug 1: /query returned 500 for any question hitting a chunked passage."""

    def test_passage_out_accepts_chunk_id(self):
        from scripts.run_server import PassageOut

        out = PassageOut(passage_id="8559285::0", passage="text", score=0.5)
        assert out.passage_id == "8559285::0"

    def test_passage_out_keeps_int_ids_as_int(self):
        from scripts.run_server import PassageOut

        assert PassageOut(passage_id=30485523, passage="t", score=0.1).passage_id == 30485523

    def test_citation_out_accepts_chunk_id(self):
        from scripts.run_server import CitationOut

        out = CitationOut(marker="[Passage 1]", passage_id="8559285::0", text="t", score=0.5)
        assert out.passage_id == "8559285::0"

    def test_query_response_serializes_a_mixed_id_result(self):
        from scripts.run_server import PassageOut, QueryResponse

        # Exactly the id mix a real BM25 search returns once chunking is on.
        ids = ["8559285::0", 30485523, 23937304]
        resp = QueryResponse(
            question="q?",
            answer="a",
            passages=[PassageOut(passage_id=i, passage="t", score=0.1) for i in ids],
        )
        assert [p.passage_id for p in resp.passages] == ids


class TestRewriteUsesAConfiguredModel:
    """Bug 4: rewrite_query sent the literal model id "default", so every HyDE
    call 400'd and was swallowed by the fallback."""

    def _client(self, response="Hypothesis: h\nExpansions: a, b"):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = response
        return client

    def test_explicit_model_is_forwarded(self):
        from src.query.rewrite import rewrite_query

        client = self._client()
        rewrite_query("What is X?", client, model="llama-3.3-70b-versatile")
        assert client.chat.completions.create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_never_sends_the_literal_default(self, monkeypatch):
        from src.query.rewrite import rewrite_query

        monkeypatch.delenv("GROQ_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        client = self._client()
        rewrite_query("What is X?", client)
        assert client.chat.completions.create.call_args.kwargs["model"] != "default"

    def test_falls_back_to_env_model(self, monkeypatch):
        from src.query.rewrite import rewrite_query

        monkeypatch.setenv("GROQ_MODEL", "groq/compound")
        client = self._client()
        rewrite_query("What is X?", client)
        assert client.chat.completions.create.call_args.kwargs["model"] == "groq/compound"

    def test_pipeline_passes_the_config_model_through(self, monkeypatch, tmp_path):
        import src.pipeline.rag as rag

        seen = {}

        def fake_rewrite(question, client, model=None):
            seen["model"] = model
            return question, ""

        monkeypatch.setattr(rag, "rewrite_query", fake_rewrite)
        monkeypatch.setattr(rag, "_get_client", lambda: object())
        monkeypatch.setattr(rag, "retrieve_candidates", lambda *a, **k: [(1, "text", 1.0)])
        monkeypatch.setattr(rag, "generate", lambda *a, **k: "answer")

        cfg = rag._load_config(tmp_path / "missing.toml")
        cfg["rewrite"]["enabled"] = True
        cfg["llm"]["model"] = "openai/gpt-oss-120b"
        rag.rag_query_full("q?", config=cfg)

        assert seen["model"] == "openai/gpt-oss-120b"


class TestGroundednessUsesAConfiguredModel:
    """Same defect class as bug 4: a hardcoded "gpt-4o-mini" fallback that does
    not exist on Groq."""

    def test_no_hardcoded_openai_model(self, monkeypatch):
        from unittest.mock import MagicMock

        from src.generation.groundedness import score_groundedness

        monkeypatch.delenv("GROQ_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "0.8"
        score_groundedness("answer", [{"text": "some source passage"}], client)
        assert client.chat.completions.create.call_args.kwargs["model"] != "gpt-4o-mini"


class TestDotenvIsActuallyLoaded:
    """Bug 8: load_env() was a `pass` stub, so the .env file holding the API keys
    was never read despite python-dotenv being a declared dependency."""

    def test_load_env_reads_a_dotenv_file(self, monkeypatch, tmp_path):
        import src.config.env as env

        (tmp_path / ".env").write_text("RAG_TEST_TOKEN=from-file\n")
        monkeypatch.setattr(env, "_project_root", lambda: tmp_path)
        monkeypatch.delenv("RAG_TEST_TOKEN", raising=False)

        env.load_env(force=True)
        assert os.getenv("RAG_TEST_TOKEN") == "from-file"

    def test_exported_variable_beats_the_file(self, monkeypatch, tmp_path):
        import src.config.env as env

        (tmp_path / ".env").write_text("RAG_TEST_TOKEN=from-file\n")
        monkeypatch.setattr(env, "_project_root", lambda: tmp_path)
        monkeypatch.setenv("RAG_TEST_TOKEN", "from-shell")

        env.load_env(force=True)
        assert os.getenv("RAG_TEST_TOKEN") == "from-shell"

    def test_missing_file_is_not_an_error(self, monkeypatch, tmp_path):
        import src.config.env as env

        monkeypatch.setattr(env, "_project_root", lambda: tmp_path)
        env.load_env(force=True)


class TestCitationEnforceIsWired:
    """Bug 2 (second half): [citation] enforce had no consumer, so the model was
    never asked to cite anything and extraction always found zero markers."""

    def test_prompt_asks_for_citations_when_enforced(self):
        from src.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(context="[Passage 1]\nfoo", question="q?", require_citations=True)
        assert "[Passage 2]" in prompt  # the instruction's worked example
        assert "citation" in prompt.lower()

    def test_prompt_unchanged_by_default(self):
        from src.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(context="[Passage 1]\nfoo", question="q?")
        assert "Support every factual claim" not in prompt

    def test_pipeline_forwards_the_enforce_flag(self, monkeypatch, tmp_path):
        import src.pipeline.rag as rag

        seen = {}
        monkeypatch.setattr(rag, "retrieve_candidates", lambda *a, **k: [(1, "text", 1.0)])
        monkeypatch.setattr(rag, "generate", lambda *a, **k: "answer")

        real = rag.build_rag_prompt

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(rag, "build_rag_prompt", spy)

        cfg = rag._load_config(tmp_path / "missing.toml")
        cfg["citation"]["enforce"] = True
        rag.rag_query_full("q?", config=cfg)

        assert seen["require_citations"] is True


class TestGroundednessSeesItsEvidence:
    """Bug 3: the judge was shown only citation markers, so it was asked whether
    claims were supported by passages it had never read."""

    def test_pipeline_passes_passage_text_to_the_judge(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        import src.pipeline.rag as rag

        seen = {}

        def fake_score(answer, evidence, client, model=None):
            seen["evidence"] = evidence
            return 0.9

        monkeypatch.setattr(rag, "retrieve_candidates",
                            lambda *a, **k: [(23179372, "Real passage text.", 1.0)])
        monkeypatch.setattr(rag, "generate", lambda *a, **k: "An answer.")
        monkeypatch.setattr(rag, "_get_client", lambda: MagicMock())
        monkeypatch.setattr(rag, "score_groundedness", fake_score)

        cfg = rag._load_config(tmp_path / "missing.toml")
        cfg["guardrail"]["groundedness_enabled"] = True
        rag.rag_query_full("q?", config=cfg)

        assert seen["evidence"][0]["text"] == "Real passage text."

    def test_judge_prompt_carries_the_text(self):
        from unittest.mock import MagicMock

        from src.generation.groundedness import score_groundedness

        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = "0.8"
        score_groundedness("claim", [{"marker": "[Passage 1]", "passage_id": 7,
                                      "text": "Distinctive source sentence."}], client)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "Distinctive source sentence." in prompt


class TestEmbeddingThrottleSurvivesAMinuteBoundary:
    """Bug 5: the rate-limit window was reset only inside the sleep branch, so
    once 60s elapsed naturally `elapsed < 60` was false forever and the guard
    could never fire again.

    A flat workload does not expose this: if the token rate exceeds the ceiling,
    the margin is crossed inside the first minute and the guard self-corrects.
    It bites when a slow opening minute disables the guard and a later burst
    then runs unprotected -- which is realistic, because per-batch token volume
    varies by an order of magnitude across the corpus's passage-size long tail.
    """

    @staticmethod
    def _run(monkeypatch, batches=30, slow_batches=9, slow=400, fast=2000, latency=8.0):
        from unittest.mock import MagicMock

        import src.retrieval.dense as dense

        clock = [0.0]
        sent = []
        throttles = []
        state = {"n": 0}

        def post(*a, **k):
            tokens = slow if state["n"] < slow_batches else fast
            sent.append((clock[0], tokens))
            state["n"] += 1
            clock[0] += latency
            r = MagicMock()
            r.raise_for_status.return_value = None
            r.json.return_value = {"data": [{"embedding": [0.1]}] * 10,
                                   "usage": {"prompt_tokens": tokens}}
            return r

        def sleep(seconds):
            if seconds > 2:
                throttles.append(clock[0])
            clock[0] += seconds

        client = MagicMock()
        client.post.side_effect = post
        monkeypatch.setattr(dense.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(dense.time, "sleep", sleep)
        dense.embed_texts(["t"] * (batches * 10), client=client)
        return sent, throttles, slow_batches

    def test_throttles_during_a_burst_after_a_slow_first_minute(self, monkeypatch):
        sent, throttles, slow_batches = self._run(monkeypatch)
        burst_start = sent[slow_batches][0]
        assert [t for t in throttles if t >= burst_start], (
            "no throttle during the burst: the TPM window was never reset, so the "
            "guard stayed disabled after the first minute elapsed"
        )

    def test_light_traffic_is_never_throttled(self, monkeypatch):
        """A rate comfortably under the ceiling must not sleep at all."""
        _, throttles, _ = self._run(
            monkeypatch, batches=20, slow_batches=20, slow=100, fast=100, latency=30.0)
        assert not throttles


class TestRerankTransport:
    """Bugs 6 and 12: a chat payload went to the text-completions endpoint, and a
    malformed 200 raised out of the client instead of degrading."""

    def test_posts_to_the_chat_endpoint(self):
        import src.retrieval.rerank as rr

        assert rr.OPENROUTER_RERANK_URL.endswith("/chat/completions")

    def test_malformed_200_falls_back_to_candidate_order(self):
        from unittest.mock import MagicMock

        from src.retrieval.rerank import rerank

        client = MagicMock()
        client.post.return_value.raise_for_status.return_value = None
        client.post.return_value.json.return_value = {"unexpected": "shape"}
        candidates = [(1, "a", 0.9), (2, "b", 0.5)]
        out = rerank("q", candidates, top_n=2, api_key="k", client=client)
        assert [pid for pid, _, _ in out] == [1, 2]

    def test_pipeline_survives_a_malformed_rerank_response(self, monkeypatch, tmp_path):
        import src.pipeline.rag as rag

        def exploding_rerank(*a, **k):
            raise KeyError("choices")

        monkeypatch.setattr(rag, "rerank_candidates", exploding_rerank)
        monkeypatch.setattr(rag, "_ensure_loaded", lambda mode: None)
        monkeypatch.setattr(rag, "_bm25_state", ("idx", [], {}))
        monkeypatch.setattr(rag, "bm25_search",
                            lambda *a, **k: [(1, "a", 0.9), (2, "b", 0.5)])

        cfg = rag._load_config(tmp_path / "missing.toml")
        cfg["retrieval"]["rerank"] = True
        out = rag.retrieve_candidates("q?", cfg["retrieval"])
        assert [pid for pid, _, _ in out] == [1, 2]


class TestMetricsRespectK:
    """Bug 9: the *_at_k helpers never truncated, so passing more than k ids
    silently computed the metric at a different k than the name claimed."""

    def test_recall_truncates_to_k(self):
        from src.eval.metrics import recall_at_k

        # The gold id sits at rank 6, outside the top 5.
        retrieved = [9, 9, 9, 9, 9, 2]
        assert recall_at_k(retrieved, [2], k=5) == 0.0
        assert recall_at_k(retrieved, [2]) == 1.0  # unbounded, as before

    def test_mrr_and_ndcg_truncate_to_k(self):
        from src.eval.metrics import mrr_at_k, ndcg_at_k

        retrieved = [9, 9, 9, 2]
        assert mrr_at_k(retrieved, [2], k=3) == 0.0
        assert ndcg_at_k(retrieved, [2], k=3) == 0.0

    def test_max_recall_reports_the_attainable_ceiling(self):
        from src.eval.metrics import max_recall_at_k

        assert max_recall_at_k([1, 2, 3, 4, 5, 6, 7, 8], k=5) == 5 / 8
        assert max_recall_at_k([1, 2], k=5) == 1.0
        assert max_recall_at_k([], k=5) == 0.0


class TestChunkingDoesNotDuplicateOrOverflow:
    """Bug 11: a sentence longer than the budget was carried whole and then
    re-emitted as overlap, producing oversized, duplicated chunks."""

    def test_oversized_sentence_is_split_not_carried(self):
        from src.data.chunking import chunk_passages

        text = " ".join(["word"] * 60) + ". " + " ".join(["tail"] * 30) + "."
        chunks = chunk_passages([{"id": 1, "text": text}], threshold=20)
        assert all(len(c["text"].split()) <= 20 for c in chunks)

    def test_chunk_count_stays_linear(self):
        """Overlap >= threshold used to drive the step size to 1 word."""
        from src.data.chunking import chunk_passages

        chunks = chunk_passages([{"id": 1, "text": " ".join(["w"] * 200)}], threshold=20)
        assert len(chunks) < 30, f"expected ~20 chunks, got {len(chunks)}"

    def test_no_chunk_contains_a_sibling_whole(self):
        from src.data.chunking import chunk_passages

        text = " ".join(f"s{i} word word word word." for i in range(40))
        chunks = [c["text"] for c in chunk_passages([{"id": 1, "text": text}], threshold=30)]
        for a, b in zip(chunks, chunks[1:]):  # noqa: B905 - offset slices differ by 1
            assert a not in b and b not in a

    def test_ids_stay_unique(self):
        from src.data.chunking import chunk_passages

        text = " ".join(["word"] * 300) + ". " + " ".join(["tail"] * 40) + "."
        chunks = chunk_passages([{"id": 7, "text": text}], threshold=25)
        assert len({c["id"] for c in chunks}) == len(chunks)
        assert all(c["parent_id"] == 7 for c in chunks)


class TestContextBudgetIsRespected:
    """The truncation guard appended the untrimmed line when the budget was
    already spent, so max_chars could be overshot by a whole passage."""

    def test_max_chars_is_never_exceeded(self):
        from src.prompts.formatter import format_passages_as_context

        passages = [(1, "A" * 100, 0.9), (2, "B" * 5900, 0.8), (3, "C" * 9000, 0.7)]
        out = format_passages_as_context(passages, max_chars=6000)
        assert len(out) <= 6000, f"context is {len(out)} chars"

    def test_first_oversized_passage_is_trimmed(self):
        from src.prompts.formatter import format_passages_as_context

        out = format_passages_as_context([(1, "X" * 20000, 0.9)], max_chars=1000)
        assert len(out) <= 1000
        assert "...[truncated]" in out


class TestProviderErrorsAreNotInternalErrors:
    """A 413/429 from the provider surfaced as a bare 500 with a traceback."""

    def _client_raising(self, status):
        import httpx
        from openai import APIStatusError

        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        response = httpx.Response(status, request=request, json={"error": {"message": "x"}})
        return APIStatusError("boom", response=response, body={"error": {"message": "x"}})

    def test_413_maps_to_413_with_actionable_detail(self, monkeypatch):
        from fastapi.testclient import TestClient

        import scripts.run_server as srv

        monkeypatch.setattr(srv, "rag_query_full",
                            lambda q: (_ for _ in ()).throw(self._client_raising(413)))
        r = TestClient(srv.app).post("/query", json={"question": "q?"})
        assert r.status_code == 413
        assert "top_k" in r.json()["detail"]

    def test_429_maps_to_429(self, monkeypatch):
        from fastapi.testclient import TestClient

        import scripts.run_server as srv

        monkeypatch.setattr(srv, "rag_query_full",
                            lambda q: (_ for _ in ()).throw(self._client_raising(429)))
        r = TestClient(srv.app).post("/query", json={"question": "q?"})
        assert r.status_code == 429

    def test_other_provider_errors_map_to_502(self, monkeypatch):
        from fastapi.testclient import TestClient

        import scripts.run_server as srv

        monkeypatch.setattr(srv, "rag_query_full",
                            lambda q: (_ for _ in ()).throw(self._client_raising(500)))
        r = TestClient(srv.app).post("/query", json={"question": "q?"})
        assert r.status_code == 502
