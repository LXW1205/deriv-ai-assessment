"""Unit tests for grounding.py — citation grounding validation."""

import pytest

from grounding import extract_keywords, check_grounding


# ===================================================================
# extract_keywords tests
# ===================================================================

class TestExtractKeywords:
    """Tests for the extract_keywords function."""

    def test_extract_keywords_removes_stop_words(self):
        """extract_keywords should filter out common stop words."""
        text = "The quick brown fox is jumping over the lazy dog"
        keywords = extract_keywords(text)

        # Stop words should be removed
        assert "the" not in keywords
        assert "is" not in keywords

        # Content words should remain
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        assert "jumping" in keywords
        assert "lazy" in keywords
        assert "dog" in keywords
        # "over" is NOT in the stop words list, so it remains
        assert "over" in keywords

    def test_extract_keywords_with_test_29_stop_words(self):
        """Test 32: extract_keywords handles all listed stop words."""
        # Test a broad set of stop words
        stop_text = (
            "the a an is are was were be been being have has had do does did "
            "will would could should may might shall can need must to of in "
            "for on with at by from as into through during before after above "
            "below between under again further then once and but or nor not "
            "so yet both either neither each every all any few more most other "
            "some such no only own same than too very just it its this that "
            "these those i me my we our you your he him his she her they them "
            "their what which who whom how when where why"
        )
        keywords = extract_keywords(stop_text)
        # All stop words should be filtered out (and short words < 3 chars)
        assert len(keywords) == 0

    def test_extract_keywords_short_words_filtered(self):
        """Words shorter than 3 characters should be filtered."""
        text = "an is be do go to in at"
        keywords = extract_keywords(text)
        assert len(keywords) == 0

    def test_extract_keywords_preserves_meaningful_words(self):
        """Meaningful words should be preserved."""
        text = "bank withdrawal processing takes three business days"
        keywords = extract_keywords(text)

        assert "bank" in keywords
        assert "withdrawal" in keywords
        assert "processing" in keywords
        assert "takes" in keywords
        assert "three" in keywords
        assert "business" in keywords
        assert "days" in keywords

    def test_extract_keywords_lowercase(self):
        """extract_keywords should return lowercase keywords."""
        text = "The QUICK Brown FOX"
        keywords = extract_keywords(text)

        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        # No uppercase should remain
        for kw in keywords:
            assert kw == kw.lower()

    def test_extract_keywords_handles_punctuation(self):
        """extract_keywords should strip punctuation."""
        text = "Hello, world! This is a test-case."
        keywords = extract_keywords(text)

        assert "hello" in keywords
        assert "world" in keywords
        assert "test" in keywords  # "test-case" splits at non-alpha
        assert "case" in keywords

    def test_extract_keywords_empty_string(self):
        """extract_keywords with empty string returns empty set."""
        assert extract_keywords("") == set()

    def test_extract_keywords_returns_set(self):
        """extract_keywords should return a set."""
        result = extract_keywords("hello world")
        assert isinstance(result, set)

    def test_extract_keywords_no_duplicates(self):
        """extract_keywords should not have duplicate keywords."""
        text = "bank bank bank withdrawal withdrawal"
        keywords = extract_keywords(text)

        assert keywords == {"bank", "withdrawal"}


# ===================================================================
# check_grounding tests
# ===================================================================

class TestCheckGrounding:
    """Tests for the check_grounding function."""

    def test_valid_citations(self, sample_grounding_retrieval, sample_grounding_answers):
        """Test 28: check_grounding with valid citations returns all valid."""
        results = check_grounding(sample_grounding_answers, sample_grounding_retrieval)

        assert len(results) == 1
        assert results[0]["query_id"] == "Q1"
        assert results[0]["all_citations_valid"] is True
        assert len(results[0]["citation_checks"]) == 1
        assert results[0]["citation_checks"][0]["valid"] is True
        assert results[0]["citation_checks"][0]["chunk_exists_in_retrieval"] is True
        assert results[0]["citation_checks"][0]["text_supports_answer"] is True

    def test_invalid_chunk_id(self):
        """Test 29: check_grounding with invalid chunk_id marks citation invalid."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_valid_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": "Some supporting text here.",
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": "Some answer [Doc A §chunk_nonexistent_0].",
                "answer_label": "grounded_answer",
                "citations": ["[Doc A §chunk_nonexistent_0]"],
                "used_chunk_ids": ["chunk_nonexistent_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is False
        assert results[0]["citation_checks"][0]["chunk_exists_in_retrieval"] is False
        assert results[0]["citation_checks"][0]["valid"] is False
        assert "not found" in results[0]["citation_checks"][0]["explanation"]

    def test_insufficient_context_answer(self):
        """Test 30: check_grounding with insufficient_context skips citation checks."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": "I cannot answer this question.",
                "answer_label": "insufficient_context",
                "citations": [],
                "used_chunk_ids": [],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is True
        assert results[0]["citation_checks"] == []
        assert "No citations required" in results[0]["explanation"]

    def test_no_keyword_overlap(self):
        """Test 31: check_grounding with no keyword overlap marks citation invalid."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_unrelated_0",
                        "doc_title": "Unrelated Doc",
                        "score": 0.9,
                        "chunk_text": "This text is about cooking recipes and kitchen appliances.",
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": (
                    "Bank withdrawals take three business days to process "
                    "after approval [Unrelated Doc §chunk_unrelated_0]."
                ),
                "answer_label": "grounded_answer",
                "citations": ["[Unrelated Doc §chunk_unrelated_0]"],
                "used_chunk_ids": ["chunk_unrelated_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is False
        assert results[0]["citation_checks"][0]["text_supports_answer"] is False
        assert results[0]["citation_checks"][0]["valid"] is False

    def test_citation_parse_failure(self):
        """check_grounding with unparseable citation marks it invalid."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": "Supporting text.",
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": "Some answer with bad citation format.",
                "answer_label": "grounded_answer",
                "citations": ["This is not a proper citation"],
                "used_chunk_ids": [],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is False
        assert results[0]["citation_checks"][0]["valid"] is False
        assert "Could not parse" in results[0]["citation_checks"][0]["explanation"]

    def test_multiple_citations_mixed_validity(self):
        """check_grounding with mixed valid/invalid citations."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_valid_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": (
                            "Bank withdrawals take three business days "
                            "after approval and funds transfer."
                        ),
                    },
                    {
                        "rank": 2,
                        "chunk_id": "chunk_also_valid_0",
                        "doc_title": "Doc B",
                        "score": 0.8,
                        "chunk_text": "Processing times vary by institution.",
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": (
                    "Bank withdrawals take three business days "
                    "[Doc A §chunk_valid_0]. "
                    "Times vary [Doc B §chunk_also_valid_0]."
                ),
                "answer_label": "grounded_answer",
                "citations": [
                    "[Doc A §chunk_valid_0]",
                    "[Doc B §chunk_also_valid_0]",
                ],
                "used_chunk_ids": ["chunk_valid_0", "chunk_also_valid_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is True
        assert len(results[0]["citation_checks"]) == 2
        assert all(c["valid"] for c in results[0]["citation_checks"])

    def test_query_not_in_retrieval(self):
        """check_grounding with query not in retrieval marks citations invalid."""
        retrieval = [
            {
                "query_id": "Q2",
                "question": "Different query",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": "Some text.",
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",  # Not in retrieval
                "answer": "Answer [Doc A §chunk_0].",
                "answer_label": "grounded_answer",
                "citations": ["[Doc A §chunk_0]"],
                "used_chunk_ids": ["chunk_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is False
        assert results[0]["citation_checks"][0]["chunk_exists_in_retrieval"] is False

    def test_keyword_overlap_ratio_reported(self):
        """check_grounding should report keyword overlap ratio."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": (
                            "Bank withdrawals take three business days "
                            "after approval processing."
                        ),
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": "Bank withdrawals processing [Doc A §chunk_0].",
                "answer_label": "grounded_answer",
                "citations": ["[Doc A §chunk_0]"],
                "used_chunk_ids": ["chunk_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        check = results[0]["citation_checks"][0]
        assert "keyword_overlap_ratio" in check
        assert "phrase_support_ratio" in check
        assert isinstance(check["keyword_overlap_ratio"], float)
        assert isinstance(check["phrase_support_ratio"], float)

    def test_empty_citations_list(self):
        """check_grounding with empty citations list should still produce result."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": "Some answer without citations.",
                "answer_label": "grounded_answer",
                "citations": [],
                "used_chunk_ids": [],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        assert results[0]["all_citations_valid"] is True  # No citations to invalidate
        assert results[0]["citation_checks"] == []

    def test_conflicting_context_answer(self):
        """check_grounding with conflicting_context label should check citations."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": "Bank withdrawals take three days.",
                    },
                ],
            },
        ]
        answers = [
            {
                "query_id": "Q1",
                "answer": "Conflicting info [Doc A §chunk_0].",
                "answer_label": "conflicting_context",
                "citations": ["[Doc A §chunk_0]"],
                "used_chunk_ids": ["chunk_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        # conflicting_context is not insufficient_context, so citations are checked
        assert results[0]["citation_checks"]  # Should have citation checks

    def test_no_answer_keywords_overlap_ratio_defaults_to_one(self):
        """When answer has no keywords after filtering, overlap_ratio defaults to 1.0."""
        retrieval = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Doc A",
                        "score": 0.9,
                        "chunk_text": "Some supporting text here with meaningful words.",
                    },
                ],
            },
        ]
        # Answer with only stop words and short words — after filtering, no keywords remain
        # "it", "is", "to", "be", "of", "the" are all stop words or < 3 chars
        answers = [
            {
                "query_id": "Q1",
                "answer": "It is to be of the [Doc A §chunk_0].",
                "answer_label": "grounded_answer",
                "citations": ["[Doc A §chunk_0]"],
                "used_chunk_ids": ["chunk_0"],
            },
        ]

        results = check_grounding(answers, retrieval)

        assert len(results) == 1
        # With no answer keywords, overlap_ratio defaults to 1.0
        check = results[0]["citation_checks"][0]
        assert check["keyword_overlap_ratio"] == 1.0
        assert check["text_supports_answer"] is True
