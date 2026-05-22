"""Unit tests for ingestion.py — document loading, parsing, and chunking."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion import (
    parse_document,
    load_documents,
    estimate_tokens,
    chunk_fixed_token,
    chunk_paragraph,
    chunk_documents,
    save_chunks,
    load_chunks,
    CHUNKING_STRATEGIES,
)


# ===================================================================
# parse_document tests
# ===================================================================

class TestParseDocument:
    """Tests for parse_document()."""

    def test_parse_document_valid_input(self, sample_raw_document):
        """Test 1: parse_document with valid input extracts Title, Section, body."""
        result = parse_document(sample_raw_document, source="/fake/path.txt")

        assert result["source"] == "/fake/path.txt"
        assert result["title"] == "Cash withdrawal processing"
        assert result["section"] == "Banking"
        assert "Bank withdrawals may take 1 to 3 business days" in result["body"]
        assert "Once approved, funds are transferred" in result["body"]
        assert "Processing times may vary" in result["body"]

    def test_parse_document_missing_title(self):
        """Test 2a: parse_document with missing Title returns empty title."""
        raw = (
            "Section: Banking\n"
            "This document has no title line.\n"
            "Just some body text here."
        )
        result = parse_document(raw, source="/fake/path.txt")

        assert result["title"] == ""
        assert result["section"] == "Banking"
        assert "This document has no title line." in result["body"]

    def test_parse_document_missing_section(self):
        """Test 2b: parse_document with missing Section returns empty section."""
        raw = (
            "Title: Some Title\n"
            "This document has no section line.\n"
            "Just some body text here."
        )
        result = parse_document(raw, source="/fake/path.txt")

        assert result["title"] == "Some Title"
        assert result["section"] == ""
        assert "This document has no section line." in result["body"]

    def test_parse_document_missing_both_title_and_section(self):
        """Test 2c: parse_document with neither Title nor Section."""
        raw = "Just plain body text without any headers."
        result = parse_document(raw, source="/fake/path.txt")

        assert result["title"] == ""
        assert result["section"] == ""
        assert result["body"] == "Just plain body text without any headers."

    def test_parse_document_empty_body(self):
        """Test 3: parse_document with only headers and no body content."""
        raw = "Title: Only Title\nSection: Only Section\n"
        result = parse_document(raw, source="/fake/path.txt")

        assert result["title"] == "Only Title"
        assert result["section"] == "Only Section"
        assert result["body"] == ""

    def test_parse_document_case_insensitive_headers(self):
        """Headers should be case-insensitive (title: vs Title:)."""
        raw = (
            "title: lowercase title\n"
            "SECTION: uppercase section\n"
            "Body text here."
        )
        result = parse_document(raw, source="/fake/path.txt")

        assert result["title"] == "lowercase title"
        assert result["section"] == "uppercase section"

    def test_parse_document_default_source(self):
        """Source defaults to empty string when not provided."""
        result = parse_document("Title: Test\nSection: Sec\nBody text.")

        assert result["source"] == ""


# ===================================================================
# estimate_tokens tests
# ===================================================================

class TestEstimateTokens:
    """Tests for estimate_tokens()."""

    def test_estimate_tokens_normal_text(self):
        """Normal text should return ~1.3 * word_count."""
        text = "one two three four five"  # 5 words
        result = estimate_tokens(text)
        assert result == max(1, int(5 * 1.3))  # int(6.5) = 6

    def test_estimate_tokens_empty_string(self):
        """Empty string should return 1 (minimum)."""
        assert estimate_tokens("") == 1

    def test_estimate_tokens_single_word(self):
        """Single word should return at least 1."""
        assert estimate_tokens("hello") == max(1, int(1 * 1.3))  # 1

    def test_estimate_tokens_whitespace_only(self):
        """Whitespace-only string should return 1."""
        assert estimate_tokens("   \t\n  ") == 1

    def test_estimate_tokens_large_text(self):
        """Large text should scale linearly."""
        words = ["word"] * 1000
        text = " ".join(words)
        result = estimate_tokens(text)
        assert result == int(1000 * 1.3)  # 1300


# ===================================================================
# chunk_fixed_token tests
# ===================================================================

class TestChunkFixedToken:
    """Tests for chunk_fixed_token()."""

    def test_chunk_fixed_token_normal_document(self, sample_parsed_document):
        """Test 4: chunk_fixed_token with normal document produces chunks."""
        chunks = chunk_fixed_token(sample_parsed_document)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "doc_title" in chunk
            assert "section" in chunk
            assert "text" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert chunk["doc_title"] == "Cash withdrawal processing"
            assert chunk["section"] == "Banking"

    def test_chunk_fixed_token_short_document(self):
        """Test 5: chunk_fixed_token with very short document (< 512 tokens) produces single chunk."""
        doc = {
            "source": "",
            "title": "Short Doc",
            "section": "Test",
            "body": "This is a very short document. It has only two sentences.",
        }
        chunks = chunk_fixed_token(doc)

        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "chunk_short_doc_0"
        assert "This is a very short document." in chunks[0]["text"]

    def test_chunk_fixed_token_long_document(self):
        """Test 6: chunk_fixed_token with very long document produces multiple chunks."""
        # Create a document with many sentences to exceed 512 tokens
        sentences = []
        for i in range(200):
            sentences.append(
                f"This is sentence number {i} in a very long document "
                f"that should definitely exceed the maximum token limit "
                f"of five hundred and twelve tokens per chunk boundary."
            )
        body = " ".join(sentences)

        doc = {
            "source": "",
            "title": "Long Document",
            "section": "Test",
            "body": body,
        }
        chunks = chunk_fixed_token(doc, max_tokens=512, overlap_pct=0.15)

        assert len(chunks) > 1
        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"chunk_long_document_{i}"

    def test_chunk_fixed_token_overlap_calculation(self):
        """Test 7: chunk_fixed_token overlap is correctly calculated."""
        # Create text that will definitely produce multiple chunks
        sentences = []
        for i in range(100):
            sentences.append(
                f"Sentence {i} contains enough words to fill up the chunk "
                f"buffer and force a split at the five hundred twelve token mark."
            )
        body = " ".join(sentences)

        doc = {
            "source": "",
            "title": "Overlap Test",
            "section": "Test",
            "body": body,
        }

        # With overlap, the last sentence(s) of chunk N should appear
        # at the beginning of chunk N+1
        chunks = chunk_fixed_token(doc, max_tokens=512, overlap_pct=0.15)

        assert len(chunks) > 1

        # Check that overlap exists between consecutive chunks
        # The last few words of chunk 0 should appear in chunk 1
        chunk0_text = chunks[0]["text"]
        chunk1_text = chunks[1]["text"]

        # Extract last sentence from chunk0
        last_sent_chunk0 = chunk0_text.split(". ")[-1] if ". " in chunk0_text else ""
        if last_sent_chunk0:
            # The overlap sentence should appear in chunk1
            assert last_sent_chunk0 in chunk1_text or last_sent_chunk0.rstrip(".") in chunk1_text

    def test_chunk_fixed_token_empty_body(self):
        """chunk_fixed_token with empty body returns empty list."""
        doc = {
            "source": "",
            "title": "Empty",
            "section": "Test",
            "body": "",
        }
        assert chunk_fixed_token(doc) == []

    def test_chunk_fixed_token_whitespace_only_body(self):
        """chunk_fixed_token with whitespace-only body returns empty list."""
        doc = {
            "source": "",
            "title": "Empty",
            "section": "Test",
            "body": "   \n\t  ",
        }
        assert chunk_fixed_token(doc) == []

    def test_chunk_fixed_token_custom_max_tokens(self):
        """Custom max_tokens parameter should be respected."""
        # Very short max to force multiple chunks
        doc = {
            "source": "",
            "title": "Custom",
            "section": "Test",
            "body": (
                "First sentence here. Second sentence here. "
                "Third sentence here. Fourth sentence here. "
                "Fifth sentence here. Sixth sentence here."
            ),
        }
        chunks = chunk_fixed_token(doc, max_tokens=10, overlap_pct=0.0)
        assert len(chunks) > 1

    def test_chunk_fixed_token_chunk_positions(self, sample_parsed_document):
        """Chunk start_char and end_char should be valid positions."""
        chunks = chunk_fixed_token(sample_parsed_document)

        for chunk in chunks:
            body = sample_parsed_document["body"]
            assert chunk["start_char"] >= 0
            assert chunk["end_char"] > chunk["start_char"]
            # The text should match the slice
            assert body[chunk["start_char"]:chunk["end_char"]] == chunk["text"]


# ===================================================================
# chunk_paragraph tests
# ===================================================================

class TestChunkParagraph:
    """Tests for chunk_paragraph()."""

    def test_chunk_paragraph_single_paragraph(self):
        """Test 8: chunk_paragraph with single paragraph produces one chunk."""
        doc = {
            "source": "",
            "title": "Single Para",
            "section": "Test",
            "body": "This is a single paragraph with no line breaks.",
        }
        chunks = chunk_paragraph(doc)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a single paragraph with no line breaks."
        assert chunks[0]["chunk_id"] == "chunk_single_para_0"

    def test_chunk_paragraph_multiple_paragraphs(self):
        """Test 9: chunk_paragraph with multiple paragraphs produces multiple chunks."""
        doc = {
            "source": "",
            "title": "Multi Para",
            "section": "Test",
            "body": (
                "First paragraph here.\n\n"
                "Second paragraph here.\n\n"
                "Third paragraph here."
            ),
        }
        chunks = chunk_paragraph(doc)

        assert len(chunks) == 3
        assert "First paragraph" in chunks[0]["text"]
        assert "Second paragraph" in chunks[1]["text"]
        assert "Third paragraph" in chunks[2]["text"]

    def test_chunk_paragraph_very_long_paragraph(self):
        """Test 10: chunk_paragraph with very long paragraph splits at sentences."""
        # Create a single paragraph longer than 1000 chars
        sentences = []
        for i in range(50):
            sentences.append(
                f"This is a long sentence number {i} that adds significant "
                f"character count to the paragraph so it exceeds the soft limit."
            )
        long_para = " ".join(sentences)

        doc = {
            "source": "",
            "title": "Long Para",
            "section": "Test",
            "body": long_para,
        }
        chunks = chunk_paragraph(doc)

        assert len(chunks) > 1
        # Each chunk should be under the soft limit (approximately)
        for chunk in chunks:
            assert len(chunk["text"]) <= 1100  # Allow some margin

    def test_chunk_paragraph_empty_body(self):
        """chunk_paragraph with empty body returns empty list."""
        doc = {
            "source": "",
            "title": "Empty",
            "section": "Test",
            "body": "",
        }
        assert chunk_paragraph(doc) == []

    def test_chunk_paragraph_paragraphs_with_extra_whitespace(self):
        """Paragraphs separated by varying whitespace should be handled."""
        doc = {
            "source": "",
            "title": "Whitespace",
            "section": "Test",
            "body": (
                "Paragraph one.\n\n\n"
                "Paragraph two.\n  \n"
                "Paragraph three."
            ),
        }
        chunks = chunk_paragraph(doc)

        assert len(chunks) == 3

    def test_chunk_paragraph_chunk_positions(self):
        """Chunk positions should be valid."""
        doc = {
            "source": "",
            "title": "Position Test",
            "section": "Test",
            "body": "First paragraph.\n\nSecond paragraph.",
        }
        chunks = chunk_paragraph(doc)

        for chunk in chunks:
            body = doc["body"]
            assert chunk["start_char"] >= 0
            assert chunk["end_char"] > chunk["start_char"]
            assert body[chunk["start_char"]:chunk["end_char"]] == chunk["text"]


# ===================================================================
# chunk_documents tests
# ===================================================================

class TestChunkDocuments:
    """Tests for chunk_documents()."""

    def test_chunk_documents_fixed_token_strategy(self, sample_documents):
        """chunk_documents with fixed_token strategy works."""
        chunks = chunk_documents(sample_documents, strategy="fixed_token")
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk

    def test_chunk_documents_paragraph_strategy(self, sample_documents):
        """chunk_documents with paragraph strategy works."""
        chunks = chunk_documents(sample_documents, strategy="paragraph")
        assert len(chunks) > 0

    def test_chunk_documents_unknown_strategy(self, sample_documents):
        """Test 11: chunk_documents with unknown strategy raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            chunk_documents(sample_documents, strategy="unknown_strategy")

        assert "Unknown chunking strategy" in str(excinfo.value)
        assert "unknown_strategy" in str(excinfo.value)
        assert "fixed_token" in str(excinfo.value)
        assert "paragraph" in str(excinfo.value)

    def test_chunk_documents_empty_list(self):
        """chunk_documents with empty document list returns empty list."""
        assert chunk_documents([], strategy="fixed_token") == []
        assert chunk_documents([], strategy="paragraph") == []

    def test_chunk_documents_passes_kwargs(self):
        """Additional kwargs should be passed to the chunker."""
        # Create enough sentences that with max_tokens=5 they must split
        sentences = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        doc = {
            "source": "",
            "title": "Kwargs Test",
            "section": "Test",
            "body": sentences,
        }
        # Very small max_tokens forces multiple chunks
        chunks = chunk_documents([doc], strategy="fixed_token", max_tokens=4, overlap_pct=0.0)
        assert len(chunks) > 1


# ===================================================================
# save_chunks / load_chunks tests
# ===================================================================

class TestSaveLoadChunks:
    """Tests for save_chunks() and load_chunks()."""

    def test_save_and_load_round_trip(self, tmp_json_path):
        """Test 12: save_chunks and load_chunks round-trip preserves data."""
        chunks = [
            {
                "chunk_id": "chunk_test_0",
                "doc_title": "Test Doc",
                "section": "Test",
                "text": "Hello world.",
                "start_char": 0,
                "end_char": 12,
            },
            {
                "chunk_id": "chunk_test_1",
                "doc_title": "Test Doc",
                "section": "Test",
                "text": "Goodbye world.",
                "start_char": 13,
                "end_char": 27,
            },
        ]

        save_chunks(chunks, tmp_json_path)
        loaded = load_chunks(tmp_json_path)

        assert loaded == chunks

    def test_save_creates_directory(self):
        """save_chunks should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "nested", "dir", "chunks.json")
            chunks = [{"chunk_id": "test_0", "text": "hello"}]

            save_chunks(chunks, path)
            assert os.path.exists(path)

            loaded = load_chunks(path)
            assert len(loaded) == 1

    def test_load_nonexistent_file(self):
        """load_chunks should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_chunks("/nonexistent/path/chunks.json")

    def test_save_empty_list(self, tmp_json_path):
        """save_chunks should handle empty list."""
        save_chunks([], tmp_json_path)
        loaded = load_chunks(tmp_json_path)
        assert loaded == []


# ===================================================================
# load_documents tests
# ===================================================================

class TestLoadDocuments:
    """Tests for load_documents()."""

    def test_load_documents_from_directory(self, tmp_kb_dir):
        """load_documents should load all .txt files from a directory."""
        docs = load_documents(tmp_kb_dir)

        assert len(docs) == 2
        titles = {d["title"] for d in docs}
        assert "Cash withdrawal processing" in titles
        assert "Password reset and account recovery" in titles

    def test_load_documents_empty_directory(self, empty_kb_dir):
        """Test 13: load_documents with empty kb directory returns empty list."""
        docs = load_documents(empty_kb_dir)
        assert docs == []

    def test_load_documents_nonexistent_directory(self):
        """load_documents should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError) as excinfo:
            load_documents("/nonexistent/kb/dir")

        assert "Knowledge base directory not found" in str(excinfo.value)

    def test_load_documents_source_paths(self, tmp_kb_dir):
        """load_documents should set source to the file path."""
        docs = load_documents(tmp_kb_dir)

        for doc in docs:
            assert doc["source"].endswith(".txt")
            assert os.path.exists(doc["source"])

    def test_load_documents_sorted_order(self, tmp_kb_dir):
        """load_documents should return documents in sorted filename order."""
        docs = load_documents(tmp_kb_dir)
        sources = [d["source"] for d in docs]
        assert sources == sorted(sources)


# ===================================================================
# CHUNKING_STRATEGIES registry tests
# ===================================================================

class TestChunkingStrategiesRegistry:
    """Tests for the CHUNKING_STRATEGIES registry."""

    def test_registry_contains_expected_strategies(self):
        """CHUNKING_STRATEGIES should contain fixed_token and paragraph."""
        assert "fixed_token" in CHUNKING_STRATEGIES
        assert "paragraph" in CHUNKING_STRATEGIES
        assert len(CHUNKING_STRATEGIES) == 2

    def test_registry_values_are_callable(self):
        """All registered strategies should be callable functions."""
        for name, func in CHUNKING_STRATEGIES.items():
            assert callable(func)


# ===================================================================
# Edge case coverage for start_char fallback paths
# ===================================================================

class TestChunkingEdgeCases:
    """Tests for edge cases in chunking that trigger fallback paths."""

    def test_chunk_fixed_token_start_char_fallback(self):
        """When body.find returns -1, start_char should fall back to 0.

        This happens when the reconstructed chunk text doesn't match
        the original body (e.g., due to sentence splitting/rejoining).
        """
        # Create a document where sentence rejoining might cause
        # the chunk text to not match body.find()
        doc = {
            "source": "",
            "title": "Fallback Test",
            "section": "Test",
            "body": "First sentence here. Second sentence here. Third sentence here.",
        }
        chunks = chunk_fixed_token(doc, max_tokens=5, overlap_pct=0.0)

        # Even if fallback triggers, start_char should be >= 0
        for chunk in chunks:
            assert chunk["start_char"] >= 0

    def test_chunk_paragraph_start_char_fallback(self):
        """When body.find returns -1 in paragraph chunking, start_char falls back to 0."""
        doc = {
            "source": "",
            "title": "Para Fallback",
            "section": "Test",
            "body": "Short paragraph.\n\nAnother short paragraph.",
        }
        chunks = chunk_paragraph(doc)

        for chunk in chunks:
            assert chunk["start_char"] >= 0
            assert chunk["end_char"] > chunk["start_char"]
