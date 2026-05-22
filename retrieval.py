"""
Hybrid retrieval pipeline.

Combines BM25 keyword scoring with Gemini embedding cosine similarity
for robust chunk retrieval.
"""

import json
import math
import os
from collections import Counter
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# BM25 implementation (pure Python, no external deps)
# ---------------------------------------------------------------------------

class BM25Index:
    """BM25 keyword-based retrieval index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self.doc_freqs: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        import re
        return re.findall(r'[a-z0-9]+', text.lower())

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the index."""
        tokens = self._tokenize(text)
        self.documents.append(text)
        self.doc_lengths.append(len(tokens))

        for token in set(tokens):
            self.doc_freqs[token] += 1

    def build(self) -> None:
        """Finalise the index, computing IDF values."""
        n_docs = len(self.documents)
        if n_docs == 0:
            return

        self.avg_doc_length = sum(self.doc_lengths) / n_docs

        for term, freq in self.doc_freqs.items():
            # IDF with smoothing
            self.idf[term] = math.log(
                (n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

        self._built = True

    def score(self, query: str) -> list[tuple[int, float]]:
        """Score all documents against a query.

        Returns list of (doc_index, score) sorted by score descending.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        query_tokens = self._tokenize(query)
        scores = []

        for doc_idx, doc_text in enumerate(self.documents):
            doc_tokens = self._tokenize(doc_text)
            doc_len = self.doc_lengths[doc_idx]
            doc_counter = Counter(doc_tokens)

            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = doc_counter.get(token, 0)
                idf = self.idf[token]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avg_doc_length)
                )
                score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Gemini embeddings
# ---------------------------------------------------------------------------

def get_gemini_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Gemini API for a list of texts.

    Returns a list of embedding vectors.
    """
    import google.genai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    # Process in batches to avoid rate limits
    batch_size = 50
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Gemini embeddings API
        results = client.models.embed_content(
            model="gemini-embedding-2",
            contents=batch,
        )

        for embedding in results.embeddings:
            all_embeddings.append(embedding.values)

    return all_embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retriever combining BM25 and embedding similarity."""

    def __init__(self, bm25_weight: float = 0.5, embedding_weight: float = 0.5):
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.bm25_index = BM25Index()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []
        self._built = False

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build the hybrid index from chunks.

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', etc.
        """
        self.chunks = chunks

        # Build BM25 index
        for idx, chunk in enumerate(chunks):
            self.bm25_index.add_document(idx, chunk["text"])
        self.bm25_index.build()

        # Build embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        self.embeddings = get_gemini_embeddings(chunk_texts)

        self._built = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks for a query.

        Returns list of result dicts with rank, chunk_id, doc_title,
        score, and chunk_text.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # BM25 scores
        bm25_scores = self.bm25_index.score(query)
        bm25_map = {idx: score for idx, score in bm25_scores}

        # Embedding scores
        query_embedding = get_gemini_embeddings([query])[0]
        embedding_scores = []
        for idx, chunk_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, chunk_embedding)
            embedding_scores.append((idx, sim))
        embedding_scores.sort(key=lambda x: x[1], reverse=True)
        embedding_map = {idx: score for idx, score in embedding_scores}

        # Normalize scores to [0, 1] range
        max_bm25 = max((s for _, s in bm25_scores), default=1.0)
        max_embed = max((s for _, s in embedding_scores), default=1.0)

        if max_bm25 == 0:
            max_bm25 = 1.0
        if max_embed == 0:
            max_embed = 1.0

        # Combine scores
        combined_scores = []
        for idx in range(len(self.chunks)):
            bm25_norm = bm25_map.get(idx, 0.0) / max_bm25
            embed_norm = embedding_map.get(idx, 0.0) / max_embed
            combined = (
                self.bm25_weight * bm25_norm +
                self.embedding_weight * embed_norm
            )
            combined_scores.append((idx, combined))

        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(combined_scores[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "score": round(score, 4),
                "chunk_text": chunk["text"],
            })

        return results


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def run_retrieval(
    retriever: HybridRetriever,
    queries: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run retrieval for all queries.

    Args:
        retriever: Built HybridRetriever instance.
        queries: List of query dicts from queries.json.
        top_k: Number of top chunks to retrieve per query.
    """
    results = []

    for query in queries:
        retrieval_result = retriever.retrieve(query["question"], top_k=top_k)
        results.append({
            "query_id": query["query_id"],
            "question": query["question"],
            "top_k": retrieval_result,
        })

    return results


def save_retrieval_results(
    results: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Save retrieval results to a JSON file."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_retrieval_results(input_path: str) -> list[dict[str, Any]]:
    """Load retrieval results from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
