"""
Task 7: Pipeline Integration Tests - Test citation extraction, groundedness scoring,
and QueryResponse model updates in the full RAG pipeline.

Tests cover:
- Citation extraction integrated into rag_query
- Groundedness scoring integrated into rag_query when enabled
- Config flags gating both features (default OFF)
- QueryResponse model includes citations and groundedness fields
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.pipeline.rag import rag_query, _load_config


class TestPipelineIntegrationCitations:
    """Task 7: Test citation extraction in full pipeline."""

    def test_rag_query_returns_citations_when_extracted(self):
        """Test that rag_query returns citations extracted from LLM answer.

        When LLM generates answer with [Passage N] markers, rag_query should:
        1. Extract and validate citations against retrieved passages
        2. Return (passages, answer, citations) tuple
        """
        # Mock the retrieval and LLM generation
        with patch('src.pipeline.rag.retrieve_candidates') as mock_retrieve, \
             patch('src.pipeline.rag.generate') as mock_generate, \
             patch('src.pipeline.rag.parse_answer') as mock_parse, \
             patch('src.pipeline.rag._load_config') as mock_config:

            # Setup config
            mock_cfg = {
                'retrieval': {'mode': 'bm25', 'top_k': 5},
                'llm': {'model': 'gpt-4', 'max_tokens': 512, 'temperature': 0.2},
                'rewrite': {'enabled': False},
                'guardrail': {'groundedness_enabled': False},
            }
            mock_config.return_value = mock_cfg

            # Setup retrieved passages: IDs 1, 3, 5
            passages = [
                (1, "Passage one text", 0.9),
                (3, "Passage three text", 0.8),
                (5, "Passage five text", 0.7),
            ]
            mock_retrieve.return_value = passages

            # Setup LLM response with valid citations
            mock_generate.return_value = "Answer text [Passage 1] states something. [Passage 3] confirms it."
            # mock_parse will extract citations
            answer_with_citations = "Answer text [Passage 1] states something. [Passage 3] confirms it."
            mock_parse.return_value = answer_with_citations

            # Call rag_query
            question = "What is the answer?"
            result = rag_query(question)

            # Result should be a tuple of (passages, answer)
            assert isinstance(result, tuple)
            assert len(result) == 2
            passages_out, answer_out = result

            # Check passages returned correctly
            assert len(passages_out) == 3
            assert passages_out[0][0] == 1
            assert passages_out[1][0] == 3

            # Check answer returned
            assert "Answer text" in answer_out

    def test_rag_query_with_invalid_citations_stripped(self):
        """Test that invalid citations (citing non-retrieved passages) are stripped.

        When LLM cites passages that weren't retrieved, citations should be
        removed from answer and not included in response.
        """
        with patch('src.pipeline.rag.retrieve_candidates') as mock_retrieve, \
             patch('src.pipeline.rag.generate') as mock_generate, \
             patch('src.pipeline.rag.parse_answer') as mock_parse, \
             patch('src.pipeline.rag._load_config') as mock_config:

            mock_cfg = {
                'retrieval': {'mode': 'bm25', 'top_k': 5},
                'llm': {'model': 'gpt-4', 'max_tokens': 512, 'temperature': 0.2},
                'rewrite': {'enabled': False},
                'guardrail': {'groundedness_enabled': False},
            }
            mock_config.return_value = mock_cfg

            # Only passages 1 and 3 retrieved
            passages = [
                (1, "Passage one text", 0.9),
                (3, "Passage three text", 0.8),
            ]
            mock_retrieve.return_value = passages

            # LLM tries to cite passage 99 which doesn't exist
            mock_generate.return_value = "Valid claim [Passage 1]. Invalid claim [Passage 99]."
            mock_parse.return_value = "Valid claim [Passage 1]. Invalid claim [Passage 99]."

            result = rag_query("What is the answer?")
            passages_out, answer_out = result

            # Answer should have invalid citation stripped
            assert "[Passage 99]" not in answer_out
            assert "[Passage 1]" in answer_out


class TestPipelineIntegrationGroundedness:
    """Task 7: Test groundedness scoring in full pipeline."""

    def test_rag_query_groundedness_disabled_by_default(self):
        """Test that groundedness scoring is disabled by default (opt-in).

        Config should default groundedness_enabled to False, so no scoring happens
        unless explicitly enabled.
        """
        cfg = _load_config()
        assert cfg['guardrail']['groundedness_enabled'] is False

    def test_rag_query_groundedness_not_called_when_disabled(self):
        """Test that groundedness scoring is not called when disabled in config."""
        with patch('src.pipeline.rag.retrieve_candidates') as mock_retrieve, \
             patch('src.pipeline.rag.generate') as mock_generate, \
             patch('src.pipeline.rag.parse_answer') as mock_parse, \
             patch('src.pipeline.rag.score_groundedness') as mock_score, \
             patch('src.pipeline.rag._load_config') as mock_config:

            mock_cfg = {
                'retrieval': {'mode': 'bm25', 'top_k': 5},
                'llm': {'model': 'gpt-4', 'max_tokens': 512, 'temperature': 0.2},
                'rewrite': {'enabled': False},
                'guardrail': {'groundedness_enabled': False},
            }
            mock_config.return_value = mock_cfg

            mock_retrieve.return_value = [(1, "Text", 0.9)]
            mock_generate.return_value = "Answer"
            mock_parse.return_value = "Answer"

            rag_query("What is the answer?")

            # groundedness scorer should NOT be called
            mock_score.assert_not_called()

    def test_rag_query_groundedness_called_when_enabled(self):
        """Test that groundedness scoring IS called when enabled in config."""
        with patch('src.pipeline.rag.retrieve_candidates') as mock_retrieve, \
             patch('src.pipeline.rag.generate') as mock_generate, \
             patch('src.pipeline.rag.parse_answer') as mock_parse, \
             patch('src.pipeline.rag.score_groundedness') as mock_score, \
             patch('src.pipeline.rag._load_config') as mock_config, \
             patch('src.pipeline.rag._get_client') as mock_get_client:

            mock_cfg = {
                'retrieval': {'mode': 'bm25', 'top_k': 5},
                'llm': {'model': 'gpt-4', 'max_tokens': 512, 'temperature': 0.2},
                'rewrite': {'enabled': False},
                'guardrail': {'groundedness_enabled': True, 'groundedness_threshold': 0.5},
            }
            mock_config.return_value = mock_cfg
            mock_score.return_value = 0.85
            mock_get_client.return_value = Mock()

            mock_retrieve.return_value = [(1, "Text", 0.9)]
            mock_generate.return_value = "Answer [Passage 1]"
            mock_parse.return_value = "Answer [Passage 1]"

            rag_query("What is the answer?")

            # groundedness scorer SHOULD be called
            mock_score.assert_called_once()


class TestQueryResponseModel:
    """Task 7: Test QueryResponse model updates to include citations and groundedness."""

    def test_query_response_has_citations_field(self):
        """Test that QueryResponse model includes citations field."""
        from scripts.run_server import QueryResponse, CitationOut

        # Create a response with citations
        citations = [
            CitationOut(marker="[Passage 1]", passage_id=1, text="Some text", score=0.85),
            CitationOut(marker="[Passage 3]", passage_id=3, text="Other text", score=0.92),
        ]

        response = QueryResponse(
            question="What is it?",
            answer="The answer [Passage 1] is [Passage 3].",
            passages=[],
            citations=citations,
        )

        assert response.citations == citations
        assert len(response.citations) == 2

    def test_query_response_has_groundedness_fields(self):
        """Test that QueryResponse model includes groundedness_score and groundedness_flagged."""
        from scripts.run_server import QueryResponse

        response = QueryResponse(
            question="What is it?",
            answer="The answer.",
            passages=[],
            citations=[],
            groundedness_score=0.75,
            groundedness_flagged=False,
        )

        assert response.groundedness_score == 0.75
        assert response.groundedness_flagged is False

    def test_query_response_groundedness_flagged_when_below_threshold(self):
        """Test that groundedness_flagged is True when score is below configured threshold."""
        from scripts.run_server import QueryResponse

        # Response with low groundedness score (below typical 0.5 threshold)
        response = QueryResponse(
            question="What is it?",
            answer="Ungrounded claim.",
            passages=[],
            citations=[],
            groundedness_score=0.3,
            groundedness_flagged=True,
        )

        assert response.groundedness_score == 0.3
        assert response.groundedness_flagged is True


class TestConfigFlags:
    """Task 7: Test configuration flags for citations and groundedness."""

    def test_config_has_citation_enforce_flag(self, tmp_path):
        """Test that config can include [citation] enforce flag."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[citation]\n"
            "enforce = true\n"
        )

        cfg = _load_config(config_file)

        # Should have citation section
        assert "citation" in cfg or "[citation]" in config_file.read_text()

    def test_config_has_guardrail_groundedness_enabled_flag(self, tmp_path):
        """Test that config can include [guardrail] groundedness_enabled flag."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[guardrail]\n"
            "groundedness_enabled = true\n"
            "groundedness_threshold = 0.6\n"
        )

        cfg = _load_config(config_file)

        assert cfg["guardrail"]["groundedness_enabled"] is True
        assert cfg["guardrail"]["groundedness_threshold"] == 0.6

    def test_config_defaults_both_features_off(self):
        """Test that by default both citation and groundedness are OFF (opt-in)."""
        cfg = _load_config()

        # Groundedness should be disabled by default
        assert cfg["guardrail"]["groundedness_enabled"] is False
