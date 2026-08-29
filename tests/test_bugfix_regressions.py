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
        score_groundedness("answer", [], client)
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
