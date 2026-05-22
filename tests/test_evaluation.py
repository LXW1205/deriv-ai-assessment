"""Unit tests for evaluation.py — retrieval evaluation against ground truth."""

import json
import os
import tempfile

import pytest

from evaluation import evaluate_retrieval, save_evaluation, load_evaluation, ALLOWED_RETRIEVAL_STATUSES


# ===================================================================
# evaluate_retrieval tests
# ===================================================================

class TestEvaluateRetrieval:
    """Tests for the evaluate_retrieval function."""

    def test_perfect_match_hit(self):
        """Test 23: evaluate_retrieval with perfect match returns 'hit'."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test query",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Expected Doc",
                        "score": 0.9,
                        "chunk_text": "Some text",
                    },
                    {
                        "rank": 2,
                        "chunk_id": "chunk_1",
                        "doc_title": "Other Doc",
                        "score": 0.5,
                        "chunk_text": "Other text",
                    },
                ],
            },
        ]
        queries = [
            {
                "query_id": "Q1",
                "question": "Test query",
                "expected_doc_titles": ["Expected Doc"],
            },
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        assert len(evaluations) == 1
        assert evaluations[0]["retrieval_status"] == "hit"
        assert evaluations[0]["matched_expected_title"] is True
        assert summary["hits"] == 1
        assert summary["partial_hits"] == 0
        assert summary["misses"] == 0

    def test_partial_match_partial_hit(self):
        """Test 24: evaluate_retrieval with partial match returns 'partial_hit'."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test query",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Expected Doc A",
                        "score": 0.9,
                        "chunk_text": "Some text",
                    },
                    {
                        "rank": 2,
                        "chunk_id": "chunk_1",
                        "doc_title": "Other Doc",
                        "score": 0.5,
                        "chunk_text": "Other text",
                    },
                    {
                        "rank": 3,
                        "chunk_id": "chunk_2",
                        "doc_title": "Yet Another",
                        "score": 0.3,
                        "chunk_text": "More text",
                    },
                ],
            },
        ]
        queries = [
            {
                "query_id": "Q1",
                "question": "Test query",
                "expected_doc_titles": ["Expected Doc A", "Expected Doc B"],
            },
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        assert len(evaluations) == 1
        assert evaluations[0]["retrieval_status"] == "partial_hit"
        assert evaluations[0]["matched_expected_title"] is True
        assert summary["hits"] == 0
        assert summary["partial_hits"] == 1
        assert summary["misses"] == 0
        assert "1/2" in evaluations[0]["explanation"]

    def test_no_match_miss(self):
        """Test 25: evaluate_retrieval with no match returns 'miss'."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test query",
                "top_k": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk_0",
                        "doc_title": "Unrelated Doc",
                        "score": 0.1,
                        "chunk_text": "Some text",
                    },
                ],
            },
        ]
        queries = [
            {
                "query_id": "Q1",
                "question": "Test query",
                "expected_doc_titles": ["Expected Doc"],
            },
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        assert len(evaluations) == 1
        assert evaluations[0]["retrieval_status"] == "miss"
        assert evaluations[0]["matched_expected_title"] is False
        assert summary["hits"] == 0
        assert summary["partial_hits"] == 0
        assert summary["misses"] == 1

    def test_aggregate_summary_correctness(self):
        """Test 26: evaluate_retrieval aggregate summary is correctly computed."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Hit query",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Doc A", "score": 0.9, "chunk_text": "t"},
                ],
            },
            {
                "query_id": "Q2",
                "question": "Partial query",
                "top_k": [
                    {"rank": 1, "chunk_id": "c1", "doc_title": "Doc B", "score": 0.9, "chunk_text": "t"},
                    {"rank": 2, "chunk_id": "c2", "doc_title": "Doc X", "score": 0.5, "chunk_text": "t"},
                    {"rank": 3, "chunk_id": "c3", "doc_title": "Doc Y", "score": 0.3, "chunk_text": "t"},
                ],
            },
            {
                "query_id": "Q3",
                "question": "Miss query",
                "top_k": [
                    {"rank": 1, "chunk_id": "c4", "doc_title": "Doc Z", "score": 0.1, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Hit query", "expected_doc_titles": ["Doc A"]},
            {"query_id": "Q2", "question": "Partial query", "expected_doc_titles": ["Doc B", "Doc C"]},
            {"query_id": "Q3", "question": "Miss query", "expected_doc_titles": ["Doc D"]},
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        assert summary["total_queries"] == 3
        assert summary["hits"] == 1
        assert summary["partial_hits"] == 1
        assert summary["misses"] == 1
        # hit_rate = 1/3 ≈ 0.3333
        assert summary["top3_hit_rate"] == pytest.approx(0.3333, abs=0.0001)

    def test_missing_query_in_results(self):
        """Test 27: evaluate_retrieval skips queries not in retrieval results."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Present query",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Doc A", "score": 0.9, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Present query", "expected_doc_titles": ["Doc A"]},
            {"query_id": "Q2", "question": "Missing query", "expected_doc_titles": ["Doc B"]},
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        # Only Q1 should be evaluated; Q2 is skipped
        assert len(evaluations) == 1
        assert evaluations[0]["query_id"] == "Q1"
        assert summary["total_queries"] == 1

    def test_empty_expected_titles(self):
        """Query with empty expected_doc_titles should result in miss."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Doc A", "score": 0.9, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Test", "expected_doc_titles": []},
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        assert len(evaluations) == 1
        # With empty expected_titles, matched_count == len(expected_titles) == 0
        # But the condition requires len(expected_titles) > 0 for "hit"
        assert evaluations[0]["retrieval_status"] == "miss"

    def test_multiple_expected_titles_all_matched(self):
        """All expected titles matched should be a hit."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Doc A", "score": 0.9, "chunk_text": "t"},
                    {"rank": 2, "chunk_id": "c1", "doc_title": "Doc B", "score": 0.8, "chunk_text": "t"},
                    {"rank": 3, "chunk_id": "c2", "doc_title": "Doc C", "score": 0.7, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Test", "expected_doc_titles": ["Doc A", "Doc B"]},
        ]

        evaluations, summary = evaluate_retrieval(retrieval_results, queries)

        assert evaluations[0]["retrieval_status"] == "hit"
        assert summary["hits"] == 1

    def test_hit_rank_detection(self):
        """Hit evaluation should detect the rank of the first matched title."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Other Doc", "score": 0.9, "chunk_text": "t"},
                    {"rank": 2, "chunk_id": "c1", "doc_title": "Expected Doc", "score": 0.8, "chunk_text": "t"},
                    {"rank": 3, "chunk_id": "c2", "doc_title": "Another", "score": 0.7, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Test", "expected_doc_titles": ["Expected Doc"]},
        ]

        evaluations, _ = evaluate_retrieval(retrieval_results, queries)

        assert evaluations[0]["retrieval_status"] == "hit"
        assert "rank" in evaluations[0]["explanation"].lower()

    def test_top_k_threshold_respected(self):
        """top_k_threshold should limit which results are checked."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Other", "score": 0.9, "chunk_text": "t"},
                    {"rank": 2, "chunk_id": "c1", "doc_title": "Other2", "score": 0.8, "chunk_text": "t"},
                    {"rank": 3, "chunk_id": "c2", "doc_title": "Other3", "score": 0.7, "chunk_text": "t"},
                    {"rank": 4, "chunk_id": "c3", "doc_title": "Expected Doc", "score": 0.6, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Test", "expected_doc_titles": ["Expected Doc"]},
        ]

        # With threshold=3, Expected Doc at rank 4 should not be found
        evaluations, _ = evaluate_retrieval(retrieval_results, queries, top_k_threshold=3)
        assert evaluations[0]["retrieval_status"] == "miss"

        # With threshold=4, Expected Doc at rank 4 should be found
        evaluations, _ = evaluate_retrieval(retrieval_results, queries, top_k_threshold=4)
        assert evaluations[0]["retrieval_status"] == "hit"

    def test_empty_retrieval_results(self):
        """Empty retrieval results should return empty evaluations."""
        evaluations, summary = evaluate_retrieval([], [])

        assert evaluations == []
        assert summary["total_queries"] == 0
        assert summary["hits"] == 0
        assert summary["top3_hit_rate"] == 0.0

    def test_all_hit_rate_calculation(self):
        """top3_hit_rate should be 1.0 when all queries are hits."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Doc A", "score": 0.9, "chunk_text": "t"},
                    {"rank": 2, "chunk_id": "c1", "doc_title": "Doc B", "score": 0.8, "chunk_text": "t"},
                    {"rank": 3, "chunk_id": "c2", "doc_title": "Doc C", "score": 0.7, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Test", "expected_doc_titles": ["Doc A"]},
        ]

        _, summary = evaluate_retrieval(retrieval_results, queries)
        assert summary["top3_hit_rate"] == 1.0

    def test_retrieved_doc_titles_top3_list(self):
        """retrieved_doc_titles_top3 should contain only top-k titles."""
        retrieval_results = [
            {
                "query_id": "Q1",
                "question": "Test",
                "top_k": [
                    {"rank": 1, "chunk_id": "c0", "doc_title": "Doc A", "score": 0.9, "chunk_text": "t"},
                    {"rank": 2, "chunk_id": "c1", "doc_title": "Doc B", "score": 0.8, "chunk_text": "t"},
                    {"rank": 3, "chunk_id": "c2", "doc_title": "Doc C", "score": 0.7, "chunk_text": "t"},
                    {"rank": 4, "chunk_id": "c3", "doc_title": "Doc D", "score": 0.6, "chunk_text": "t"},
                ],
            },
        ]
        queries = [
            {"query_id": "Q1", "question": "Test", "expected_doc_titles": ["Doc A"]},
        ]

        evaluations, _ = evaluate_retrieval(retrieval_results, queries, top_k_threshold=3)

        assert evaluations[0]["retrieved_doc_titles_top3"] == ["Doc A", "Doc B", "Doc C"]


# ===================================================================
# ALLOWED_RETRIEVAL_STATUSES constant tests
# ===================================================================

class TestAllowedRetrievalStatuses:
    """Tests for the ALLOWED_RETRIEVAL_STATUSES constant."""

    def test_contains_expected_statuses(self):
        """ALLOWED_RETRIEVAL_STATUSES should contain hit, partial_hit, miss."""
        assert "hit" in ALLOWED_RETRIEVAL_STATUSES
        assert "partial_hit" in ALLOWED_RETRIEVAL_STATUSES
        assert "miss" in ALLOWED_RETRIEVAL_STATUSES
        assert len(ALLOWED_RETRIEVAL_STATUSES) == 3


# ===================================================================
# save_evaluation / load_evaluation tests
# ===================================================================

class TestSaveLoadEvaluation:
    """Tests for save_evaluation() and load_evaluation()."""

    def test_save_and_load_round_trip(self):
        """save_evaluation and load_evaluation round-trip preserves data."""
        evaluations = [
            {
                "query_id": "Q1",
                "expected_doc_titles": ["Doc A"],
                "retrieved_doc_titles_top3": ["Doc A"],
                "retrieval_status": "hit",
                "matched_expected_title": True,
                "explanation": "Found at rank 1",
            },
        ]
        summary = {
            "top3_hit_rate": 1.0,
            "total_queries": 1,
            "hits": 1,
            "partial_hits": 0,
            "misses": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "eval_output", "eval.json")
            save_evaluation(evaluations, summary, path)

            loaded = load_evaluation(path)

            assert loaded["evaluations"] == evaluations
            assert loaded["summary"] == summary

    def test_save_creates_directory(self):
        """save_evaluation should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "nested", "eval.json")
            save_evaluation([], {"top3_hit_rate": 0.0, "total_queries": 0, "hits": 0, "partial_hits": 0, "misses": 0}, path)
            assert os.path.exists(path)
