"""Unit tests for validate.py — ValidationResult class and validation checks."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from validate import (
    ValidationResult,
    ALLOWED_ANSWER_LABELS,
    ALLOWED_RETRIEVAL_STATUSES,
    REQUIRED_ARTIFACTS,
    OPTIONAL_ARTIFACTS,
)


# ===================================================================
# ValidationResult tests
# ===================================================================

class TestValidationResult:
    """Tests for the ValidationResult class."""

    def test_initial_state(self):
        """ValidationResult should start with zero counts."""
        result = ValidationResult()
        assert result.passed == 0
        assert result.failed == 0
        assert result.errors == []

    def test_check_pass(self):
        """Test 33a: ValidationResult check with True increments passed."""
        result = ValidationResult()
        result.check(True, "Test passed")

        assert result.passed == 1
        assert result.failed == 0
        assert result.errors == []

    def test_check_fail(self):
        """Test 33b: ValidationResult check with False increments failed."""
        result = ValidationResult()
        result.check(False, "Test failed")

        assert result.passed == 0
        assert result.failed == 1
        assert "Test failed" in result.errors

    def test_check_multiple_passes(self):
        """Multiple passing checks should accumulate."""
        result = ValidationResult()
        result.check(True, "First pass")
        result.check(True, "Second pass")
        result.check(True, "Third pass")

        assert result.passed == 3
        assert result.failed == 0

    def test_check_multiple_failures(self):
        """Multiple failing checks should accumulate errors."""
        result = ValidationResult()
        result.check(False, "First fail")
        result.check(False, "Second fail")

        assert result.passed == 0
        assert result.failed == 2
        assert "First fail" in result.errors
        assert "Second fail" in result.errors

    def test_check_mixed_results(self):
        """Mixed pass/fail should track both correctly."""
        result = ValidationResult()
        result.check(True, "Pass 1")
        result.check(False, "Fail 1")
        result.check(True, "Pass 2")
        result.check(False, "Fail 2")
        result.check(True, "Pass 3")

        assert result.passed == 3
        assert result.failed == 2
        assert len(result.errors) == 2

    def test_summary_all_pass(self, capsys):
        """Test 34a: summary returns True when all checks pass."""
        result = ValidationResult()
        result.check(True, "Check 1")
        result.check(True, "Check 2")

        overall = result.summary()

        assert overall is True
        captured = capsys.readouterr()
        assert "2/2 checks passed" in captured.out

    def test_summary_some_fail(self, capsys):
        """Test 34b: summary returns False when any check fails."""
        result = ValidationResult()
        result.check(True, "Check 1")
        result.check(False, "Check 2")
        result.check(True, "Check 3")

        overall = result.summary()

        assert overall is False
        captured = capsys.readouterr()
        assert "2/3 checks passed" in captured.out
        assert "Failed checks:" in captured.out
        assert "Check 2" in captured.out

    def test_summary_no_checks(self, capsys):
        """summary with no checks should return True (no failures)."""
        result = ValidationResult()
        overall = result.summary()

        assert overall is True
        captured = capsys.readouterr()
        assert "0/0 checks passed" in captured.out

    def test_summary_prints_separator(self, capsys):
        """summary should print separator lines."""
        result = ValidationResult()
        result.check(True, "Test")
        result.summary()

        captured = capsys.readouterr()
        assert "=" * 60 in captured.out

    def test_check_prints_pass(self, capsys):
        """check with True should print PASS message."""
        result = ValidationResult()
        result.check(True, "My test message")

        captured = capsys.readouterr()
        assert "PASS: My test message" in captured.out

    def test_check_prints_fail(self, capsys):
        """check with False should print FAIL message."""
        result = ValidationResult()
        result.check(False, "My failure message")

        captured = capsys.readouterr()
        assert "FAIL: My failure message" in captured.out


# ===================================================================
# Controlled vocabulary tests
# ===================================================================

class TestControlledVocabularies:
    """Tests for the controlled vocabulary constants."""

    def test_allowed_answer_labels(self):
        """ALLOWED_ANSWER_LABELS should contain expected values."""
        assert "grounded_answer" in ALLOWED_ANSWER_LABELS
        assert "insufficient_context" in ALLOWED_ANSWER_LABELS
        assert "conflicting_context" in ALLOWED_ANSWER_LABELS
        assert len(ALLOWED_ANSWER_LABELS) == 3

    def test_allowed_retrieval_statuses(self):
        """ALLOWED_RETRIEVAL_STATUSES should contain expected values."""
        assert "hit" in ALLOWED_RETRIEVAL_STATUSES
        assert "partial_hit" in ALLOWED_RETRIEVAL_STATUSES
        assert "miss" in ALLOWED_RETRIEVAL_STATUSES
        assert len(ALLOWED_RETRIEVAL_STATUSES) == 3

    def test_required_artifacts(self):
        """REQUIRED_ARTIFACTS should list expected artifact paths."""
        assert "artifacts/chunks.json" in REQUIRED_ARTIFACTS
        assert "artifacts/retrieval.json" in REQUIRED_ARTIFACTS
        assert "artifacts/answers.json" in REQUIRED_ARTIFACTS
        assert "artifacts/eval.json" in REQUIRED_ARTIFACTS
        assert len(REQUIRED_ARTIFACTS) == 4

    def test_optional_artifacts(self):
        """OPTIONAL_ARTIFACTS should list expected optional paths."""
        assert "artifacts/grounding_check.json" in OPTIONAL_ARTIFACTS
        assert "artifacts/chunking_comparison.json" in OPTIONAL_ARTIFACTS
        assert len(OPTIONAL_ARTIFACTS) == 2


# ===================================================================
# Validation function tests (with mocked file system)
# ===================================================================

class TestValidationFunctions:
    """Tests for validation functions with mocked artifacts."""

    @pytest.fixture
    def mock_artifact_dir(self):
        """Create a temporary directory with mock artifact files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create required artifacts
            artifacts_dir = os.path.join(tmpdir, "artifacts")
            os.makedirs(artifacts_dir)

            # chunks.json
            with open(os.path.join(artifacts_dir, "chunks.json"), "w") as f:
                json.dump([
                    {
                        "chunk_id": "chunk_test_0",
                        "doc_title": "Test Doc",
                        "section": "Test",
                        "text": "Sample chunk text.",
                        "start_char": 0,
                        "end_char": 18,
                    },
                ], f)

            # retrieval.json
            with open(os.path.join(artifacts_dir, "retrieval.json"), "w") as f:
                json.dump([
                    {
                        "query_id": "Q1",
                        "question": "Test question?",
                        "top_k": [
                            {
                                "rank": 1,
                                "chunk_id": "chunk_test_0",
                                "doc_title": "Test Doc",
                                "score": 0.85,
                                "chunk_text": "Sample chunk text.",
                            },
                            {
                                "rank": 2,
                                "chunk_id": "chunk_test_1",
                                "doc_title": "Other Doc",
                                "score": 0.50,
                                "chunk_text": "Other text.",
                            },
                            {
                                "rank": 3,
                                "chunk_id": "chunk_test_2",
                                "doc_title": "Another Doc",
                                "score": 0.30,
                                "chunk_text": "Another text.",
                            },
                        ],
                    },
                ], f)

            # answers.json
            with open(os.path.join(artifacts_dir, "answers.json"), "w") as f:
                json.dump([
                    {
                        "query_id": "Q1",
                        "answer": "Sample answer [Test Doc §chunk_test_0].",
                        "answer_label": "grounded_answer",
                        "citations": ["[Test Doc §chunk_test_0]"],
                        "used_chunk_ids": ["chunk_test_0"],
                    },
                ], f)

            # eval.json
            with open(os.path.join(artifacts_dir, "eval.json"), "w") as f:
                json.dump({
                    "evaluations": [
                        {
                            "query_id": "Q1",
                            "expected_doc_titles": ["Test Doc"],
                            "retrieved_doc_titles_top3": ["Test Doc", "Other Doc", "Another Doc"],
                            "retrieval_status": "hit",
                            "matched_expected_title": True,
                            "explanation": "Expected title found at rank 1",
                        },
                    ],
                    "summary": {
                        "top3_hit_rate": 1.0,
                        "total_queries": 1,
                        "hits": 1,
                        "partial_hits": 0,
                        "misses": 0,
                    },
                }, f)

            # queries.json (at project root)
            with open(os.path.join(tmpdir, "queries.json"), "w") as f:
                json.dump([
                    {
                        "query_id": "Q1",
                        "question": "Test question?",
                        "expected_doc_titles": ["Test Doc"],
                    },
                ], f)

            yield tmpdir

    def test_validate_all_pass(self, mock_artifact_dir, capsys):
        """validate() should return True when all artifacts are valid."""
        from validate import validate

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is True
        finally:
            os.chdir(original_cwd)

    def test_validate_missing_artifact(self, mock_artifact_dir, capsys):
        """validate() crashes when a required artifact is missing (defect in validate.py).

        Note: validate_evaluation_statuses() calls load_json() without checking
        if the file exists first. This is a defect — it should handle missing
        files gracefully and record a failure instead of crashing.
        """
        from validate import validate

        # Remove a required artifact
        os.remove(os.path.join(mock_artifact_dir, "artifacts", "eval.json"))

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            # validate() raises FileNotFoundError instead of returning False
            with pytest.raises(FileNotFoundError):
                validate()
        finally:
            os.chdir(original_cwd)

    def test_validate_invalid_json(self, mock_artifact_dir, capsys):
        """validate() crashes when a JSON file is invalid (defect in validate.py).

        Note: validate_evaluation_statuses() calls load_json() without catching
        JSONDecodeError. This is a defect — it should handle invalid JSON
        gracefully and record a failure instead of crashing.
        """
        from validate import validate

        # Corrupt a JSON file
        with open(os.path.join(mock_artifact_dir, "artifacts", "eval.json"), "w") as f:
            f.write("{invalid json content")

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            # validate() raises JSONDecodeError instead of returning False
            import json
            with pytest.raises(json.JSONDecodeError):
                validate()
        finally:
            os.chdir(original_cwd)

    def test_validate_insufficient_chunks(self, mock_artifact_dir, capsys):
        """validate() should fail when a query has fewer than 3 chunks."""
        from validate import validate

        # Replace retrieval.json with only 2 chunks
        with open(os.path.join(mock_artifact_dir, "artifacts", "retrieval.json"), "w") as f:
            json.dump([
                {
                    "query_id": "Q1",
                    "question": "Test question?",
                    "top_k": [
                        {
                            "rank": 1,
                            "chunk_id": "chunk_test_0",
                            "doc_title": "Test Doc",
                            "score": 0.85,
                            "chunk_text": "Sample chunk text.",
                        },
                        {
                            "rank": 2,
                            "chunk_id": "chunk_test_1",
                            "doc_title": "Other Doc",
                            "score": 0.50,
                            "chunk_text": "Other text.",
                        },
                    ],
                },
            ], f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_non_numeric_score(self, mock_artifact_dir, capsys):
        """validate() should fail when a retrieval score is not numeric."""
        from validate import validate

        # Replace retrieval.json with non-numeric score
        with open(os.path.join(mock_artifact_dir, "artifacts", "retrieval.json"), "w") as f:
            json.dump([
                {
                    "query_id": "Q1",
                    "question": "Test question?",
                    "top_k": [
                        {
                            "rank": 1,
                            "chunk_id": "chunk_test_0",
                            "doc_title": "Test Doc",
                            "score": "not_a_number",
                            "chunk_text": "Sample chunk text.",
                        },
                        {
                            "rank": 2,
                            "chunk_id": "chunk_test_1",
                            "doc_title": "Other Doc",
                            "score": 0.50,
                            "chunk_text": "Other text.",
                        },
                        {
                            "rank": 3,
                            "chunk_id": "chunk_test_2",
                            "doc_title": "Another Doc",
                            "score": 0.30,
                            "chunk_text": "Another text.",
                        },
                    ],
                },
            ], f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_invalid_answer_label(self, mock_artifact_dir, capsys):
        """validate() should fail when an answer label is not in controlled vocabulary."""
        from validate import validate

        # Replace answers.json with invalid label
        with open(os.path.join(mock_artifact_dir, "artifacts", "answers.json"), "w") as f:
            json.dump([
                {
                    "query_id": "Q1",
                    "answer": "Sample answer.",
                    "answer_label": "invalid_label",
                    "citations": [],
                    "used_chunk_ids": [],
                },
            ], f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_invalid_retrieval_status(self, mock_artifact_dir, capsys):
        """validate() should fail when a retrieval status is not in controlled vocabulary."""
        from validate import validate

        # Replace eval.json with invalid status
        with open(os.path.join(mock_artifact_dir, "artifacts", "eval.json"), "w") as f:
            json.dump({
                "evaluations": [
                    {
                        "query_id": "Q1",
                        "expected_doc_titles": ["Test Doc"],
                        "retrieved_doc_titles_top3": ["Test Doc"],
                        "retrieval_status": "invalid_status",
                        "matched_expected_title": True,
                        "explanation": "Test",
                    },
                ],
                "summary": {
                    "top3_hit_rate": 1.0,
                    "total_queries": 1,
                    "hits": 1,
                    "partial_hits": 0,
                    "misses": 0,
                },
            }, f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_missing_evaluation_summary(self, mock_artifact_dir, capsys):
        """validate() should fail when evaluation summary is missing."""
        from validate import validate

        # Replace eval.json without summary
        with open(os.path.join(mock_artifact_dir, "artifacts", "eval.json"), "w") as f:
            json.dump({
                "evaluations": [],
            }, f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_grounded_answer_without_citations(self, mock_artifact_dir, capsys):
        """validate() should fail when grounded_answer has no citations."""
        from validate import validate

        # Replace answers.json with grounded_answer but no citations
        with open(os.path.join(mock_artifact_dir, "artifacts", "answers.json"), "w") as f:
            json.dump([
                {
                    "query_id": "Q1",
                    "answer": "Sample answer without citations.",
                    "answer_label": "grounded_answer",
                    "citations": [],
                    "used_chunk_ids": [],
                },
            ], f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_cited_chunk_not_retrieved(self, mock_artifact_dir, capsys):
        """validate() should fail when cited chunk was not retrieved."""
        from validate import validate

        # Replace answers.json with a chunk_id not in retrieval
        with open(os.path.join(mock_artifact_dir, "artifacts", "answers.json"), "w") as f:
            json.dump([
                {
                    "query_id": "Q1",
                    "answer": "Sample answer [Test Doc §chunk_not_retrieved].",
                    "answer_label": "grounded_answer",
                    "citations": ["[Test Doc §chunk_not_retrieved]"],
                    "used_chunk_ids": ["chunk_not_retrieved"],
                },
            ], f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_unmatched_query_ids(self, mock_artifact_dir, capsys):
        """validate() should fail when query IDs don't match between files."""
        from validate import validate

        # Replace answers.json with different query_id
        with open(os.path.join(mock_artifact_dir, "artifacts", "answers.json"), "w") as f:
            json.dump([
                {
                    "query_id": "Q99",  # Doesn't match queries.json
                    "answer": "Sample answer.",
                    "answer_label": "insufficient_context",
                    "citations": [],
                    "used_chunk_ids": [],
                },
            ], f)

        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            assert result is False
        finally:
            os.chdir(original_cwd)

    def test_validate_optional_artifacts_skipped(self, mock_artifact_dir, capsys):
        """validate() should skip optional artifacts that don't exist."""
        from validate import validate

        # Don't create optional artifacts — they should be skipped
        original_cwd = os.getcwd()
        try:
            os.chdir(mock_artifact_dir)
            result = validate()
            # Should still pass (optional artifacts are optional)
            assert result is True
        finally:
            os.chdir(original_cwd)
