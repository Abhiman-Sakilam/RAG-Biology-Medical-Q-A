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
