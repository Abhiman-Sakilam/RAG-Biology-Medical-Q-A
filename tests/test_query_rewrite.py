import pytest
import src.query.rewrite as rewrite_module
import src.pipeline.rag as rag_module


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str):
        self.response = response
        self.chat = self

    def completions_create(self, model, messages, max_tokens, temperature):
        class Choice:
            class Message:
                def __init__(self, content):
                    self.content = content

            def __init__(self, content):
                self.message = self.Message(content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(self.response)

    def create(self, model, messages, max_tokens, temperature):
        return self.completions_create(model, messages, max_tokens, temperature)

    class Completions:
        def __init__(self, parent):
            self.parent = parent

        def create(self, model, messages, max_tokens, temperature):
            return self.parent.completions_create(model, messages, max_tokens, temperature)

    @property
    def completions(self):
        return self.Completions(self)


def test_rewrite_query_returns_tuple_with_hypothesis_and_expansions():
    """Test that rewrite_query returns (hypothesis_passage, keyword_expansions)."""
    llm_response = "Hypothesis: COVID-19 is a respiratory illness caused by SARS-CoV-2.\nExpansions: coronavirus, pandemic, infectious disease"
    client = MockLLMClient(llm_response)

    hypothesis, expansions = rewrite_module.rewrite_query("What is COVID-19?", client)

    assert isinstance(hypothesis, str)
    assert isinstance(expansions, str)
    assert len(hypothesis) > 0
    assert len(expansions) > 0


def test_rewrite_query_parses_hypothesis_from_response():
    """Test that rewrite_query correctly parses the hypothesis from LLM response."""
    llm_response = "Hypothesis: Diabetes is a metabolic disorder affecting blood glucose.\nExpansions: blood sugar, glucose metabolism, hyperglycemia"
    client = MockLLMClient(llm_response)

    hypothesis, _ = rewrite_module.rewrite_query("What is diabetes?", client)

    assert "metabolic disorder" in hypothesis.lower()
    assert "glucose" in hypothesis.lower()


def test_rewrite_query_parses_expansions_from_response():
    """Test that rewrite_query correctly parses keyword expansions from LLM response."""
    llm_response = "Hypothesis: Heart disease involves problems with the cardiovascular system.\nExpansions: cardiac disease, heart attack, coronary artery disease"
    client = MockLLMClient(llm_response)

    _, expansions = rewrite_module.rewrite_query("What is heart disease?", client)

    assert "cardiac" in expansions.lower() or "heart" in expansions.lower()
    assert "coronary" in expansions.lower() or "artery" in expansions.lower()


def test_rewrite_query_fallback_on_error():
    """Test that rewrite_query returns (question, '') on error."""
    class BrokenClient:
        def chat_completions_create(self, **kwargs):
            raise Exception("API error")

    client = BrokenClient()
    question = "What is pneumonia?"

    hypothesis, expansions = rewrite_module.rewrite_query(question, client)

    assert hypothesis == question
    assert expansions == ""


def test_rewrite_query_called_when_enabled_in_config(monkeypatch):
    """Test that rewrite_query is called when [rewrite] enabled = true."""
    rag_module._bm25_state = ("fake-bm25", [{"id": 1, "passage": "test"}], {1: "test"})
    rag_module._dense_state = None

    rewrite_called = {}

    def fake_rewrite_query(question, client, model=None):
        rewrite_called["q"] = question
        return "hypothesis about the question", "keyword expansions"

    # Mock rewrite_query in the rag_module namespace
    monkeypatch.setattr(rag_module, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(rag_module, "bm25_search", lambda q, bm25, corpus, lut, k: [(1, "test", 5.0)])
    monkeypatch.setattr(rag_module, "generate", lambda prompt, model, max_tokens, temperature: "answer")
    # Mock the LLM client getter so it doesn't try to connect to real API
    mock_client = MockLLMClient("Hypothesis: test\nExpansions: test")
    monkeypatch.setattr(rag_module, "_get_client", lambda: mock_client)

    config = {
        "retrieval": {"mode": "bm25", "top_k": 1},
        "llm": {"model": "test", "max_tokens": 10, "temperature": 0.0},
        "rewrite": {"enabled": True},
    }

    rag_module.rag_query("What is this?", config=config)

    assert "q" in rewrite_called
    assert rewrite_called["q"] == "What is this?"


def test_rewrite_query_skipped_when_disabled_in_config(monkeypatch):
    """Test that rewrite_query is NOT called when [rewrite] enabled = false (default)."""
    rag_module._bm25_state = ("fake-bm25", [{"id": 1, "passage": "test"}], {1: "test"})
    rag_module._dense_state = None

    rewrite_called = {"count": 0}

    def fake_rewrite_query(question, client, model=None):
        rewrite_called["count"] += 1
        return "hypothesis", "expansions"

    monkeypatch.setattr(rewrite_module, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(rag_module, "bm25_search", lambda q, bm25, corpus, lut, k: [(1, "test", 5.0)])
    monkeypatch.setattr(rag_module, "generate", lambda prompt, model, max_tokens, temperature: "answer")

    # Test with rewrite disabled (default)
    config = {
        "retrieval": {"mode": "bm25", "top_k": 1},
        "llm": {"model": "test", "max_tokens": 10, "temperature": 0.0},
        "rewrite": {"enabled": False},
    }

    rag_module.rag_query("What is this?", config=config)

    assert rewrite_called["count"] == 0
