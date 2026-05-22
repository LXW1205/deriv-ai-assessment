"""
Document ingestion and chunking module.

Supports two chunking strategies:
  1. fixed_token — 512-token chunks with 15% overlap
  2. paragraph   — paragraph/sentence-based chunks
"""

import json
import os
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(kb_dir: str) -> list[dict[str, Any]]:
    """Load all .txt files from the knowledge base directory.

    Parses Title: and Section: headers from each file.
    Works regardless of filename or file ordering.
    """
    documents = []
    kb_path = Path(kb_dir)

    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    txt_files = sorted(kb_path.glob("*.txt"))

    for filepath in txt_files:
        raw = filepath.read_text(encoding="utf-8").strip()
        doc = parse_document(raw, source=str(filepath))
        documents.append(doc)

    return documents


def parse_document(raw: str, source: str = "") -> dict[str, Any]:
    """Parse a single document's Title, Section, and body text."""
    title = ""
    section = ""
    body_lines = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            title = stripped[len("title:"):].strip()
        elif stripped.lower().startswith("section:"):
            section = stripped[len("section:"):].strip()
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    return {
        "source": source,
        "title": title,
        "section": section,
        "body": body,
    }


# ---------------------------------------------------------------------------
# Token estimation (simple word-based approximation)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple word-based heuristic.

    English text averages ~1.3 tokens per word. We use this as a
    lightweight approximation without requiring a tokenizer library.
    """
    words = text.split()
    return max(1, int(len(words) * 1.3))


# ---------------------------------------------------------------------------
# Chunking strategy 1: Fixed-token with overlap
# ---------------------------------------------------------------------------

def chunk_fixed_token(
    document: dict[str, Any],
    max_tokens: int = 512,
    overlap_pct: float = 0.15,
) -> list[dict[str, Any]]:
    """Chunk a document into fixed-size token windows with overlap.

    Args:
        document: Parsed document dict.
        max_tokens: Maximum tokens per chunk.
        overlap_pct: Fraction of tokens to overlap between chunks.
    """
    body = document["body"]
    if not body.strip():
        return []

    # Split body into sentences for clean boundaries
    sentences = re.split(r'(?<=[.!?])\s+', body)

    chunks = []
    current_sentences = []
    current_tokens = 0
    overlap_tokens = int(max_tokens * overlap_pct)
    chunk_counter = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            # Save current chunk
            chunk_text = " ".join(current_sentences)
            start_char = body.find(chunk_text)
            if start_char == -1:
                start_char = 0
            end_char = start_char + len(chunk_text)

            chunks.append({
                "chunk_id": f"chunk_{document['title'].lower().replace(' ', '_')}_{chunk_counter}",
                "doc_title": document["title"],
                "section": document["section"],
                "text": chunk_text,
                "start_char": start_char,
                "end_char": end_char,
            })
            chunk_counter += 1

            # Keep overlap sentences
            overlap_count = max(1, int(len(current_sentences) * overlap_pct))
            current_sentences = current_sentences[-overlap_count:]
            current_tokens = sum(estimate_tokens(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        start_char = body.find(chunk_text)
        if start_char == -1:
            start_char = 0
        end_char = start_char + len(chunk_text)

        chunks.append({
            "chunk_id": f"chunk_{document['title'].lower().replace(' ', '_')}_{chunk_counter}",
            "doc_title": document["title"],
            "section": document["section"],
            "text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
        })

    return chunks


# ---------------------------------------------------------------------------
# Chunking strategy 2: Paragraph-based
# ---------------------------------------------------------------------------

def chunk_paragraph(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk a document by paragraph boundaries.

    Each paragraph becomes a chunk. If a paragraph is very long,
    it is split at sentence boundaries to keep chunks manageable.
    """
    body = document["body"]
    if not body.strip():
        return []

    # Split into paragraphs (double newline)
    paragraphs = re.split(r'\n\s*\n', body)

    chunks = []
    chunk_counter = 0
    max_chunk_chars = 1000  # Soft limit for paragraph chunks

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph is too long, split at sentences
        if len(para) > max_chunk_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = []
            current_len = 0

            for sent in sentences:
                if current_len + len(sent) > max_chunk_chars and current:
                    chunk_text = " ".join(current)
                    start_char = body.find(chunk_text)
                    if start_char == -1:
                        start_char = 0
                    end_char = start_char + len(chunk_text)

                    chunks.append({
                        "chunk_id": f"chunk_{document['title'].lower().replace(' ', '_')}_{chunk_counter}",
                        "doc_title": document["title"],
                        "section": document["section"],
                        "text": chunk_text,
                        "start_char": start_char,
                        "end_char": end_char,
                    })
                    chunk_counter += 1
                    current = [sent]
                    current_len = len(sent)
                else:
                    current.append(sent)
                    current_len += len(sent)

            if current:
                chunk_text = " ".join(current)
                start_char = body.find(chunk_text)
                if start_char == -1:
                    start_char = 0
                end_char = start_char + len(chunk_text)

                chunks.append({
                    "chunk_id": f"chunk_{document['title'].lower().replace(' ', '_')}_{chunk_counter}",
                    "doc_title": document["title"],
                    "section": document["section"],
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                })
                chunk_counter += 1
        else:
            start_char = body.find(para)
            if start_char == -1:
                start_char = 0
            end_char = start_char + len(para)

            chunks.append({
                "chunk_id": f"chunk_{document['title'].lower().replace(' ', '_')}_{chunk_counter}",
                "doc_title": document["title"],
                "section": document["section"],
                "text": para,
                "start_char": start_char,
                "end_char": end_char,
            })
            chunk_counter += 1

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CHUNKING_STRATEGIES = {
    "fixed_token": chunk_fixed_token,
    "paragraph": chunk_paragraph,
}


def chunk_documents(
    documents: list[dict[str, Any]],
    strategy: str = "fixed_token",
    **kwargs,
) -> list[dict[str, Any]]:
    """Chunk all documents using the specified strategy.

    Args:
        documents: List of parsed document dicts.
        strategy: One of 'fixed_token' or 'paragraph'.
        **kwargs: Additional strategy-specific parameters.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                         f"Available: {list(CHUNKING_STRATEGIES.keys())}")

    chunker = CHUNKING_STRATEGIES[strategy]
    all_chunks = []

    for doc in documents:
        chunks = chunker(doc, **kwargs)
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: list[dict[str, Any]], output_path: str) -> None:
    """Save chunks to a JSON file."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def load_chunks(input_path: str) -> list[dict[str, Any]]:
    """Load chunks from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
