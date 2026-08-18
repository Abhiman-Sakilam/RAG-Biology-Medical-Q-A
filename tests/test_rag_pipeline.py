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


def test_hybrid_mode_loads_dense_index_when_only_bm25_was_loaded(monkeypatch):
    # Simulate a process that already loaded state in bm25 mode (e.g. an
    # earlier request), then a later call asks for hybrid mode. The dense
    # index must be loaded on demand rather than silently degrading to
    # sparse-only results.
    rag_module._bm25_state = _fixture_corpus_state()
    rag_module._dense_state = None

    called = {}

    def fake_load_indices(mode):
        called["mode"] = mode
        corpus, id_to_passage = _fixture_corpus_state()[1], _fixture_corpus_state()[2]
        rag_module._dense_state = ("fake-dense-index", corpus, id_to_passage)

    monkeypatch.setattr(rag_module, "load_indices", fake_load_indices)
    monkeypatch.setattr(rag_module, "bm25_search", lambda q, bm25, corpus, lut, k: [(1, "alpha", 5.0)])
    monkeypatch.setattr(rag_module, "dense_search", lambda q, idx, corpus, lut, k: [(2, "beta", 0.9)])
    monkeypatch.setattr(rag_module, "generate", lambda prompt, model, max_tokens, temperature: "answer")

    passages, _ = rag_module.rag_query("q?", config=_base_config(mode="hybrid", top_k=2))

    assert called["mode"] == "hybrid"
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
