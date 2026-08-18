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
