import pytest
from unittest.mock import Mock, patch
from src.generation.parser import extract_and_validate_citations
from src.generation.groundedness import score_groundedness


class TestCitationExtraction:
    """Test citation regex parsing and validation."""

    def test_extract_valid_passage_citations(self):
        """Test extraction of [Passage N] format citations."""
        answer = "According to research [Passage 1], the finding is significant. [Passage 3] confirms this."
        retrieved_ids = {1, 3, 5}
        cleaned, citations = extract_and_validate_citations(answer, retrieved_ids)

        assert len(citations) == 2
        assert citations[0] == ("[Passage 1]", 1)
        assert citations[1] == ("[Passage 3]", 3)
        assert "[Passage 1]" in cleaned
        assert "[Passage 3]" in cleaned

    def test_extract_shorthand_p_citations(self):
        """Test extraction of [P N] format citations."""
        answer = "Some evidence [P 2] shows that [P 4] is true."
        retrieved_ids = {2, 4, 6}
        cleaned, citations = extract_and_validate_citations(answer, retrieved_ids)

        assert len(citations) == 2
        assert citations[0] == ("[P 2]", 2)
        assert citations[1] == ("[P 4]", 4)

    def test_invalid_citation_stripped(self):
        """Test that citations with invalid passage IDs are stripped."""
        answer = "Claim one [Passage 1] is valid. Claim two [Passage 99] is invalid."
        retrieved_ids = {1, 3, 5}
        cleaned, citations = extract_and_validate_citations(answer, retrieved_ids)

        assert len(citations) == 1
        assert citations[0] == ("[Passage 1]", 1)
        assert "[Passage 99]" not in cleaned
        assert "[Passage 1]" in cleaned

    def test_malformed_citations_ignored(self):
        """Test that malformed citations are ignored."""
        answer = "Valid [Passage 1] and malformed [Passage] and [P] are present."
        retrieved_ids = {1, 2}
        cleaned, citations = extract_and_validate_citations(answer, retrieved_ids)

        assert len(citations) == 1
        assert citations[0] == ("[Passage 1]", 1)

    def test_no_citations(self):
        """Test handling of answers with no citations."""
        answer = "This answer has no citations at all."
        retrieved_ids = {1, 2, 3}
        cleaned, citations = extract_and_validate_citations(answer, retrieved_ids)

        assert len(citations) == 0
        assert cleaned == answer

    def test_duplicate_citations(self):
        """Test handling of duplicate citations in same answer."""
        answer = "Study [Passage 1] and [Passage 1] both support this."
        retrieved_ids = {1, 2}
        cleaned, citations = extract_and_validate_citations(answer, retrieved_ids)

        # Should preserve duplicates as they appear
        assert len(citations) == 2
        assert all(c == ("[Passage 1]", 1) for c in citations)


class TestGroundednessScoring:
    """Test groundedness score computation."""

    def test_score_groundedness_valid_response(self):
        """Test groundedness scoring with valid LLM response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Groundedness score: 0.85"
        mock_client.chat.completions.create.return_value = mock_response

        answer = "The sky is blue [Passage 1]."
        citations = [("[Passage 1]", 1)]
        score = score_groundedness(answer, citations, mock_client)

        assert score == 0.85
        # Verify LLM was called
        assert mock_client.chat.completions.create.called

    def test_score_groundedness_parsing_integer(self):
        """Test groundedness score parsing with integer response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Score: 1"
        mock_client.chat.completions.create.return_value = mock_response

        answer = "Valid claim [Passage 1]."
        citations = [("[Passage 1]", 1)]
        score = score_groundedness(answer, citations, mock_client)

        assert score == 1.0

    def test_score_groundedness_error_fallback(self):
        """Test fallback to 0.5 when LLM call fails."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        answer = "Some answer [Passage 1]."
        citations = [("[Passage 1]", 1)]
        score = score_groundedness(answer, citations, mock_client)

        assert score == 0.5

    def test_score_groundedness_no_citations(self):
        """Test groundedness scoring with no citations (ungrounded)."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Groundedness: 0.0"
        mock_client.chat.completions.create.return_value = mock_response

        answer = "Unsubstantiated claim with no citations."
        citations = []
        score = score_groundedness(answer, citations, mock_client)

        # Should still call LLM and return parsed score
        assert score == 0.0
        assert mock_client.chat.completions.create.called

    def test_score_groundedness_with_multiple_citations(self):
        """Test groundedness scoring with multiple citations."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "0.75"
        mock_client.chat.completions.create.return_value = mock_response

        answer = "Fact one [P 1] and fact two [P 2] are grounded."
        citations = [("[P 1]", 1), ("[P 2]", 2)]
        score = score_groundedness(answer, citations, mock_client)

        assert score == 0.75


class TestGroundednessConfig:
    """Test configuration gating of groundedness guardrail."""

    def test_groundedness_config_defaults(self):
        """Test that groundedness config defaults are properly set."""
        from src.pipeline.rag import _load_config

        cfg = _load_config()

        assert "guardrail" in cfg
        assert cfg["guardrail"]["groundedness_enabled"] is False
        assert cfg["guardrail"]["groundedness_threshold"] == 0.5

    def test_groundedness_config_can_be_overridden(self, tmp_path):
        """Test that groundedness config can be overridden via config.toml."""
        from src.pipeline.rag import _load_config

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[guardrail]\n"
            "groundedness_enabled = true\n"
            "groundedness_threshold = 0.7\n"
        )
        cfg = _load_config(config_file)

        assert cfg["guardrail"]["groundedness_enabled"] is True
        assert cfg["guardrail"]["groundedness_threshold"] == 0.7
