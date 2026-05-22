"""Shared fixtures for the mini-RAG pipeline test suite."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path so modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Ingestion fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_document():
    """A well-formed raw document string with Title, Section, and body."""
    return (
        "Title: Cash withdrawal processing\n"
        "Section: Banking\n"
        "Bank withdrawals may take 1 to 3 business days after approval.\n"
        "Once approved, funds are transferred to your nominated account.\n"
        "Processing times may vary depending on your bank."
    )


@pytest.fixture
def sample_parsed_document():
    """A parsed document dict ready for chunking."""
    return {
        "source": "/fake/path/article_01.txt",
        "title": "Cash withdrawal processing",
        "section": "Banking",
        "body": (
            "Bank withdrawals may take 1 to 3 business days after approval. "
            "Once approved, funds are transferred to your nominated account. "
            "Processing times may vary depending on your bank."
        ),
    }


@pytest.fixture
def sample_documents():
    """Multiple parsed documents for bulk operations."""
    return [
        {
            "source": "/fake/path/article_01.txt",
            "title": "Cash withdrawal processing",
            "section": "Banking",
            "body": (
                "Bank withdrawals may take 1 to 3 business days after approval. "
                "Once approved, funds are transferred to your nominated account. "
                "Processing times may vary depending on your bank."
            ),
        },
        {
            "source": "/fake/path/article_02.txt",
            "title": "Password reset and account recovery",
            "section": "Account Management",
            "body": (
                "Reset links expire after 30 minutes. "
                "Support agents cannot see or manually reveal user passwords. "
                "If you need help, use the forgot password link on the login page."
            ),
        },
    ]


@pytest.fixture
def tmp_kb_dir():
    """Create a temporary knowledge base directory with sample .txt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sample KB files
        Path(tmpdir, "article_01.txt").write_text(
            "Title: Cash withdrawal processing\n"
            "Section: Banking\n"
            "Bank withdrawals may take 1 to 3 business days after approval.\n"
            "Once approved, funds are transferred to your nominated account.\n",
            encoding="utf-8",
        )
        Path(tmpdir, "article_02.txt").write_text(
            "Title: Password reset and account recovery\n"
            "Section: Account Management\n"
            "Reset links expire after 30 minutes.\n"
            "Support agents cannot see or manually reveal user passwords.\n",
            encoding="utf-8",
        )
        yield tmpdir


@pytest.fixture
def empty_kb_dir():
    """Create a temporary empty knowledge base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tmp_json_path():
    """Provide a temporary file path for JSON save/load round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_output", "data.json")


# ---------------------------------------------------------------------------
# Retrieval fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    """Sample chunk dicts for retrieval testing."""
    return [
        {
            "chunk_id": "chunk_cash_withdrawal_processing_0",
            "doc_title": "Cash withdrawal processing",
            "section": "Banking",
            "text": (
                "Bank withdrawals may take 1 to 3 business days after approval. "
                "Once approved, funds are transferred to your nominated account."
            ),
            "start_char": 0,
            "end_char": 130,
        },
        {
            "chunk_id": "chunk_password_reset_and_account_recovery_0",
            "doc_title": "Password reset and account recovery",
            "section": "Account Management",
            "text": (
                "Reset links expire after 30 minutes. "
                "Support agents cannot see or manually reveal user passwords."
            ),
            "start_char": 0,
            "end_char": 120,
        },
        {
            "chunk_id": "chunk_document_verification_requirements_0",
            "doc_title": "Document verification requirements",
            "section": "Compliance",
            "text": (
                "Utility bills older than 6 months are not accepted. "
                "Proof of address must be recent and clearly show your name."
            ),
            "start_char": 0,
            "end_char": 130,
        },
    ]


@pytest.fixture
def mock_embeddings():
    """Return deterministic fake embedding vectors."""
    # 3 chunks → 3 embeddings, each a list of 4 floats
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3],
    ]


@pytest.fixture
def mock_query_embedding():
    """A deterministic fake query embedding."""
    return [0.15, 0.25, 0.35, 0.45]


@pytest.fixture
def patched_retrieval(mock_embeddings, mock_query_embedding):
    """
    Patch get_gemini_embeddings so HybridRetriever can be tested without
    calling the real Gemini API.
    """
    def fake_get_embeddings(texts):
        # Return pre-built embeddings for chunks, query embedding for single text
        if len(texts) == 1:
            return [mock_query_embedding]
        return mock_embeddings[:len(texts)]

    with patch("retrieval.get_gemini_embeddings", side_effect=fake_get_embeddings):
        yield


# ---------------------------------------------------------------------------
# Evaluation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_queries():
    """Sample query dicts with expected_doc_titles."""
    return [
        {
            "query_id": "Q1",
            "question": "How long for bank withdrawal?",
            "expected_doc_titles": ["Cash withdrawal processing"],
        },
        {
            "query_id": "Q2",
            "question": "Reset link expired?",
            "expected_doc_titles": [
                "Password reset and account recovery",
                "Account security",
            ],
        },
        {
            "query_id": "Q3",
            "question": "Can I withdraw demo profit?",
            "expected_doc_titles": ["Demo account behaviour"],
        },
    ]


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for evaluation testing."""
    return [
        {
            "query_id": "Q1",
            "question": "How long for bank withdrawal?",
            "top_k": [
                {
                    "rank": 1,
                    "chunk_id": "chunk_cash_withdrawal_processing_0",
                    "doc_title": "Cash withdrawal processing",
                    "score": 0.85,
                    "chunk_text": "Bank withdrawals take 1 to 3 business days.",
                },
                {
                    "rank": 2,
                    "chunk_id": "chunk_other_0",
                    "doc_title": "Other doc",
                    "score": 0.30,
                    "chunk_text": "Some other text.",
                },
            ],
        },
        {
            "query_id": "Q2",
            "question": "Reset link expired?",
            "top_k": [
                {
                    "rank": 1,
                    "chunk_id": "chunk_password_reset_0",
                    "doc_title": "Password reset and account recovery",
                    "score": 0.90,
                    "chunk_text": "Reset links expire after 30 minutes.",
                },
                {
                    "rank": 2,
                    "chunk_id": "chunk_other_1",
                    "doc_title": "Other doc",
                    "score": 0.20,
                    "chunk_text": "Some other text.",
                },
            ],
        },
        {
            "query_id": "Q3",
            "question": "Can I withdraw demo profit?",
            "top_k": [
                {
                    "rank": 1,
                    "chunk_id": "chunk_unrelated_0",
                    "doc_title": "Unrelated doc",
                    "score": 0.10,
                    "chunk_text": "Completely unrelated text.",
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Grounding fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_grounding_retrieval():
    """Retrieval results for grounding checks."""
    return [
        {
            "query_id": "Q1",
            "question": "How long for bank withdrawal?",
            "top_k": [
                {
                    "rank": 1,
                    "chunk_id": "chunk_cash_withdrawal_processing_0",
                    "doc_title": "Cash withdrawal processing",
                    "score": 0.85,
                    "chunk_text": (
                        "Bank withdrawals may take 1 to 3 business days "
                        "after approval. Funds are transferred to your account."
                    ),
                },
            ],
        },
    ]


@pytest.fixture
def sample_grounding_answers():
    """Answers for grounding checks."""
    return [
        {
            "query_id": "Q1",
            "answer": (
                "Bank withdrawals take 1 to 3 business days after approval "
                "[Cash withdrawal processing §chunk_cash_withdrawal_processing_0]."
            ),
            "answer_label": "grounded_answer",
            "citations": [
                "[Cash withdrawal processing §chunk_cash_withdrawal_processing_0]"
            ],
            "used_chunk_ids": ["chunk_cash_withdrawal_processing_0"],
        },
    ]
