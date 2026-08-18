import scripts.run_server as run_server


def test_startup_loads_indices_with_configured_mode(monkeypatch):
    called = {}

    def fake_load_indices(mode):
        called["mode"] = mode

    monkeypatch.setattr(run_server, "load_indices", fake_load_indices)
    monkeypatch.setattr(run_server, "_cfg", {"retrieval": {"mode": "hybrid"}})
    run_server._load_indices_on_startup()
    assert called["mode"] == "hybrid"
