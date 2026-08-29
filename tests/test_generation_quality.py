from unittest.mock import Mock
from src.generation.parser import extract_and_validate_citations
from src.generation.groundedness import score_groundedness


class TestCitationExtraction:
    """Test citation regex parsing and ordinal-to-passage-id resolution.

    The number in a [Passage N] marker is the passage's 1-based position in the
    prompt context, not its corpus id, so these tests use realistic corpus ids
    (large PubMed-style ints and chunked "parent::idx" strings) that are
    deliberately different from their ordinals.
    """

    ORDERED_IDS = [23179372, 19270706, "9797::1"]

    def test_extract_valid_passage_citations(self):
        """[Passage N] resolves to the id at position N of the retrieved list."""
        answer = "According to research [Passage 1], the finding is significant. [Passage 3] confirms this."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert citations == [("[Passage 1]", 23179372), ("[Passage 3]", "9797::1")]
        assert "[Passage 1]" in cleaned
        assert "[Passage 3]" in cleaned

    def test_extract_shorthand_p_citations(self):
        """Test extraction of [P N] format citations."""
        answer = "Some evidence [P 2] shows that [P 3] is true."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert citations == [("[P 2]", 19270706), ("[P 3]", "9797::1")]

    def test_chunked_string_id_resolves(self):
        """Regression: a chunk id is a string and must survive resolution intact."""
        _, citations = extract_and_validate_citations("Claim [Passage 3].", self.ORDERED_IDS)
        assert citations == [("[Passage 3]", "9797::1")]

    def test_ordinal_is_not_compared_against_corpus_id(self):
        """Regression: ordinals must not be validated against the id values.

        Previously the marker's number was tested for membership in the set of
        retrieved ids, so every citation over a real corpus was judged invalid
        and silently stripped from the answer.
        """
        answer = "Histone methylation is altered [Passage 1] and affects proliferation [Passage 2]."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert len(citations) == 2
        assert "[Passage 1]" in cleaned and "[Passage 2]" in cleaned

    def test_out_of_range_citation_stripped(self):
        """Test that citations pointing past the retrieved list are stripped."""
        answer = "Claim one [Passage 1] is valid. Claim two [Passage 99] is invalid."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert citations == [("[Passage 1]", 23179372)]
        assert "[Passage 99]" not in cleaned
        assert "[Passage 1]" in cleaned

    def test_stripping_leaves_no_whitespace_scar(self):
        """Removing a marker must not leave a dangling space before punctuation."""
        answer = "Alpha is true [Passage 42]. Beta follows [Passage 1]."
        cleaned, _ = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert cleaned == "Alpha is true. Beta follows [Passage 1]."

    def test_malformed_citations_ignored(self):
        """Test that malformed citations are ignored."""
        answer = "Valid [Passage 1] and malformed [Passage] and [P] are present."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert citations == [("[Passage 1]", 23179372)]

    def test_no_citations(self):
        """Test handling of answers with no citations."""
        answer = "This answer has no citations at all."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert len(citations) == 0
        assert cleaned == answer

    def test_duplicate_citations(self):
        """Test handling of duplicate citations in same answer."""
        answer = "Study [Passage 1] and [Passage 1] both support this."
        cleaned, citations = extract_and_validate_citations(answer, self.ORDERED_IDS)

        assert len(citations) == 2
        assert all(c == ("[Passage 1]", 23179372) for c in citations)

    def test_empty_retrieved_list_strips_everything(self):
        """With nothing retrieved, no marker can be valid."""
        cleaned, citations = extract_and_validate_citations("Claim [Passage 1].", [])
        assert citations == []
        assert "[Passage 1]" not in cleaned


class TestGroundednessScoring:
    """Test groundedness score computation.

    The judge must be shown the text of each evidence passage: a grader that
    only sees citation markers has nothing to compare the answer against.
    """

    EVIDENCE = [
        {"marker": "[Passage 1]", "passage_id": 23179372,
         "text": "The sky appears blue because of Rayleigh scattering."},
    ]

    def _client(self, content):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_score_groundedness_valid_response(self):
        client = self._client("Groundedness score: 0.85")
        score = score_groundedness("The sky is blue [Passage 1].", self.EVIDENCE, client)
        assert score == 0.85
        assert client.chat.completions.create.called

    def test_prompt_contains_the_passage_text(self):
        """Regression: the judge used to receive only markers and ids."""
        client = self._client("0.9")
        score_groundedness("The sky is blue [Passage 1].", self.EVIDENCE, client)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "Rayleigh scattering" in prompt
        assert "[Passage 1]" in prompt

    def test_long_passages_are_truncated_in_the_prompt(self):
        client = self._client("0.9")
        evidence = [{"marker": "[Passage 1]", "passage_id": 1, "text": "x" * 5000}]
        score_groundedness("claim", evidence, client)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "...[truncated]" in prompt
        assert len(prompt) < 4000

    def test_score_groundedness_parsing_integer(self):
        client = self._client("Score: 1")
        assert score_groundedness("Valid claim [Passage 1].", self.EVIDENCE, client) == 1.0

    def test_score_groundedness_error_fallback(self):
        client = Mock()
        client.chat.completions.create.side_effect = Exception("API error")
        assert score_groundedness("Some answer.", self.EVIDENCE, client) == 0.5

    def test_no_evidence_scores_zero_without_calling_the_llm(self):
        """Nothing to be grounded in means ungrounded; no need to ask a judge."""
        client = self._client("0.9")
        assert score_groundedness("Unsubstantiated claim.", [], client) == 0.0
        assert not client.chat.completions.create.called

    def test_score_groundedness_with_multiple_citations(self):
        client = self._client("0.75")
        evidence = [
            {"marker": "[P 1]", "passage_id": 1, "text": "Fact one is documented."},
            {"marker": "[P 2]", "passage_id": 2, "text": "Fact two is documented."},
        ]
        score = score_groundedness("Fact one [P 1] and fact two [P 2].", evidence, client)
        assert score == 0.75
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "Fact one is documented." in prompt
        assert "Fact two is documented." in prompt


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
