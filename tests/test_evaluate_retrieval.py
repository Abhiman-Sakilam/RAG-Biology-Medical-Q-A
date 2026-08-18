from scripts.evaluate_retrieval import evaluate_mode


def test_evaluate_mode_averages_metrics_across_questions():
    qa_pairs = [
        {"question": "q1", "relevant_passage_ids": [10]},
        {"question": "q2", "relevant_passage_ids": [20]},
    ]

    def fake_retrieve(question):
        return [10] if question == "q1" else [99]

    result = evaluate_mode(qa_pairs, fake_retrieve)
    assert result["recall_at_k"] == 0.5
    assert result["mrr_at_k"] == 0.5
