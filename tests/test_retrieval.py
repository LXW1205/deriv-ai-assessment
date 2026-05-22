"""Unit tests for retrieval.py — BM25 index, cosine similarity, hybrid retriever."""

import math
from unittest.mock import patch, MagicMock

import pytest

from retrieval import BM25Index, cosine_similarity, HybridRetriever


# ===================================================================
# BM25Index tests
# ===================================================================

class TestBM25Index:
    """Tests for the BM25Index class."""

    def test_add_document_and_build(self):
        """Test 15: BM25Index add_document and build work correctly."""
        index = BM25Index()

        index.add_document(0, "The quick brown fox jumps over the lazy dog")
        index.add_document(1, "The lazy dog sleeps all day long")
        index.build()

        assert len(index.documents) == 2
        assert index._built is True
        assert index.avg_doc_length > 0
        assert len(index.idf) > 0

    def test_add_document_tracks_frequencies(self):
        """add_document should track document frequencies correctly."""
        index = BM25Index()

        index.add_document(0, "apple banana cherry")
        index.add_document(1, "apple banana date")
        index.add_document(2, "apple elderberry fig")
        index.build()

        # "apple" appears in all 3 docs
        assert index.doc_freqs["apple"] == 3
        # "banana" appears in 2 docs
        assert index.doc_freqs["banana"] == 2
        # "cherry" appears in 1 doc
        assert index.doc_freqs["cherry"] == 1

    def test_score_with_matching_query(self):
        """Test 16: BM25Index score returns higher scores for matching docs."""
        index = BM25Index()

        index.add_document(0, "Python programming language tutorial")
        index.add_document(1, "Java programming language tutorial")
        index.add_document(2, "Cooking recipes for beginners")
        index.build()

        scores = index.score("Python programming")

        # Should return all docs sorted by score
        assert len(scores) == 3
        # Doc 0 (Python) should rank highest
        assert scores[0][0] == 0
        # Scores should be in descending order
        assert scores[0][1] >= scores[1][1] >= scores[2][1]

    def test_score_with_no_matching_terms(self):
        """Test 17: BM25Index score with no matching terms returns zero scores."""
        index = BM25Index()

        index.add_document(0, "Python programming language")
        index.add_document(1, "Java programming language")
        index.build()

        scores = index.score("quantum physics relativity")

        # All scores should be 0.0 since no terms match
        for doc_idx, score in scores:
            assert score == 0.0

    def test_score_before_build_raises_runtime_error(self):
        """Test 18: BM25Index score before build raises RuntimeError."""
        index = BM25Index()
        index.add_document(0, "Some text here")

        with pytest.raises(RuntimeError) as excinfo:
            index.score("Some query")

        assert "Index not built" in str(excinfo.value)
        assert "build()" in str(excinfo.value)

    def test_score_returns_sorted_results(self):
        """score should return results sorted by score descending."""
        index = BM25Index()

        index.add_document(0, "machine learning neural networks deep learning")
        index.add_document(1, "machine learning basics introduction")
        index.add_document(2, "cooking recipes food kitchen")
        index.build()

        scores = index.score("machine learning neural networks")

        # Verify descending order
        for i in range(len(scores) - 1):
            assert scores[i][1] >= scores[i + 1][1]

    def test_build_empty_index(self):
        """build() on an empty index should not crash."""
        index = BM25Index()
        index.build()

        # Should not be marked as built (no documents)
        assert index._built is False
        assert index.avg_doc_length == 0.0

    def test_tokenize_lowercase_and_alphanumeric(self):
        """_tokenize should lowercase and split on non-alphanumeric."""
        index = BM25Index()

        tokens = index._tokenize("Hello, World! 123 test-case")
        assert tokens == ["hello", "world", "123", "test", "case"]

    def test_bm25_default_parameters(self):
        """BM25Index should use default k1=1.5 and b=0.75."""
        index = BM25Index()
        assert index.k1 == 1.5
        assert index.b == 0.75

    def test_bm25_custom_parameters(self):
        """BM25Index should accept custom k1 and b parameters."""
        index = BM25Index(k1=2.0, b=0.5)
        assert index.k1 == 2.0
        assert index.b == 0.5


# ===================================================================
# cosine_similarity tests
# ===================================================================

class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors(self):
        """Test 19: cosine_similarity with identical vectors returns 1.0."""
        vec = [1.0, 2.0, 3.0, 4.0]
        result = cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test 20: cosine_similarity with orthogonal vectors returns 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        result = cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(0.0)

    def test_zero_vector(self):
        """Test 21: cosine_similarity with zero vector returns 0.0."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(0.0)

    def test_both_zero_vectors(self):
        """cosine_similarity with both zero vectors returns 0.0."""
        result = cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert result == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """cosine_similarity with opposite vectors returns -1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        result = cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(-1.0)

    def test_partial_similarity(self):
        """cosine_similarity should return correct partial similarity."""
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 1.0]
        result = cosine_similarity(vec_a, vec_b)
        # cos(45°) = 1/sqrt(2) ≈ 0.7071
        assert result == pytest.approx(1.0 / math.sqrt(2), rel=1e-6)

    def test_single_element_vectors(self):
        """cosine_similarity with single-element vectors."""
        result = cosine_similarity([5.0], [3.0])
        assert result == pytest.approx(1.0)

    def test_negative_values(self):
        """cosine_similarity should handle negative values correctly."""
        vec_a = [-1.0, -1.0]
        vec_b = [-1.0, -1.0]
        result = cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(1.0)


# ===================================================================
# HybridRetriever tests
# ===================================================================

class TestHybridRetriever:
    """Tests for the HybridRetriever class."""

    def test_build_index_and_retrieve(self, sample_chunks, patched_retrieval):
        """Test 22: HybridRetriever build_index and retrieve work correctly."""
        retriever = HybridRetriever()
        retriever.build_index(sample_chunks)

        assert retriever._built is True
        assert len(retriever.chunks) == 3
        assert len(retriever.embeddings) == 3

        results = retriever.retrieve("bank withdrawal", top_k=2)

        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2
        assert "chunk_id" in results[0]
        assert "doc_title" in results[0]
        assert "score" in results[0]
        assert "chunk_text" in results[0]

    def test_build_index_stores_chunks(self, sample_chunks, patched_retrieval):
        """build_index should store chunks and build BM25 index."""
        retriever = HybridRetriever()
        retriever.build_index(sample_chunks)

        assert retriever.bm25_index._built is True
        assert len(retriever.bm25_index.documents) == 3

    def test_retrieve_before_build_raises_runtime_error(self):
        """retrieve() before build_index should raise RuntimeError."""
        retriever = HybridRetriever()

        with pytest.raises(RuntimeError) as excinfo:
            retriever.retrieve("some query")

        assert "Index not built" in str(excinfo.value)
        assert "build_index()" in str(excinfo.value)

    def test_retrieve_top_k_limits_results(self, sample_chunks, patched_retrieval):
        """retrieve should respect top_k parameter."""
        retriever = HybridRetriever()
        retriever.build_index(sample_chunks)

        results = retriever.retrieve("test query", top_k=1)
        assert len(results) == 1

        results = retriever.retrieve("test query", top_k=5)
        assert len(results) == 3  # Only 3 chunks available

    def test_retrieve_scores_are_normalized(self, sample_chunks, patched_retrieval):
        """retrieve should return normalized combined scores."""
        retriever = HybridRetriever()
        retriever.build_index(sample_chunks)

        results = retriever.retrieve("test query", top_k=3)

        for result in results:
            assert 0.0 <= result["score"] <= 1.0

    def test_retrieve_default_weights(self):
        """HybridRetriever should use default 0.5/0.5 weights."""
        retriever = HybridRetriever()
        assert retriever.bm25_weight == 0.5
        assert retriever.embedding_weight == 0.5

    def test_retrieve_custom_weights(self):
        """HybridRetriever should accept custom weights."""
        retriever = HybridRetriever(bm25_weight=0.7, embedding_weight=0.3)
        assert retriever.bm25_weight == 0.7
        assert retriever.embedding_weight == 0.3

    def test_retrieve_rounds_scores(self, sample_chunks, patched_retrieval):
        """retrieve should round scores to 4 decimal places."""
        retriever = HybridRetriever()
        retriever.build_index(sample_chunks)

        results = retriever.retrieve("test query", top_k=1)
        score = results[0]["score"]

        # Score should have at most 4 decimal places
        assert round(score, 4) == score

    def test_build_index_empty_chunks(self, patched_retrieval):
        """build_index with empty chunks list should not crash."""
        retriever = HybridRetriever()
        retriever.build_index([])

        assert retriever._built is True
        assert len(retriever.chunks) == 0
        # Note: BM25Index._built remains False when built with 0 docs,
        # so retrieve() on an empty index will raise RuntimeError.
        # This is expected behavior from the source code.

    def test_retrieve_zero_embedding_scores(self, sample_chunks):
        """When all embedding similarities are 0, max_embed fallback to 1.0."""
        # Mock embeddings that produce zero similarity with any query
        zero_embeddings = [[0.0, 0.0, 0.0, 0.0] for _ in sample_chunks]
        zero_query = [0.0, 0.0, 0.0, 0.0]

        def fake_get_embeddings(texts):
            if len(texts) == 1:
                return [zero_query]
            return zero_embeddings[:len(texts)]

        with patch("retrieval.get_gemini_embeddings", side_effect=fake_get_embeddings):
            retriever = HybridRetriever()
            retriever.build_index(sample_chunks)

            # Should not crash even with all-zero embeddings
            results = retriever.retrieve("test query", top_k=2)
            assert len(results) == 2
            # Scores should still be valid (from BM25 component)
            for r in results:
                assert isinstance(r["score"], float)
