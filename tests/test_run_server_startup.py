import asyncio

import scripts.run_server as run_server


def test_lifespan_loads_indices_with_configured_mode(monkeypatch):
    called = {}

    def fake_load_indices(mode):
        called["mode"] = mode

    monkeypatch.setattr(run_server, "load_indices", fake_load_indices)
    monkeypatch.setattr(run_server, "_cfg", {"retrieval": {"mode": "hybrid"}})

    async def exercise():
        async with run_server.lifespan(run_server.app):
            pass

    asyncio.run(exercise())
    assert called["mode"] == "hybrid"


def test_lifespan_defaults_to_bm25(monkeypatch):
    called = {}
    monkeypatch.setattr(run_server, "load_indices", lambda mode: called.setdefault("mode", mode))
    monkeypatch.setattr(run_server, "_cfg", {})

    async def exercise():
        async with run_server.lifespan(run_server.app):
            pass

    asyncio.run(exercise())
    assert called["mode"] == "bm25"
