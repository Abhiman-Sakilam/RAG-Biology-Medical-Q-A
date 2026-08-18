from src.eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def test_recall_at_k_counts_gold_hits():
    assert recall_at_k([1, 2, 3], [2, 4]) == 0.5


def test_recall_at_k_no_relevant_returns_zero():
    assert recall_at_k([1, 2], []) == 0.0


def test_mrr_at_k_uses_first_hit_rank():
    assert mrr_at_k([5, 6, 2], [2]) == 1.0 / 3


def test_mrr_at_k_no_hit_returns_zero():
    assert mrr_at_k([5, 6], [2]) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2], [1, 2]) == 1.0


def test_ndcg_at_k_lower_when_relevant_ranked_lower():
    high = ndcg_at_k([1, 3], [1])
    low = ndcg_at_k([3, 1], [1])
    assert high == 1.0
    assert low < high
