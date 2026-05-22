"""
FastAPI server for the mini-RAG pipeline.

Exposes a POST /answer endpoint that accepts a question and returns
a citation-strict answer grounded in the knowledge base.
"""

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from ingestion import load_documents, chunk_documents
from retrieval import HybridRetriever


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Mini-RAG Pipeline API", version="1.0.0")

# Global retriever (lazy-loaded)
_retriever: HybridRetriever | None = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer_label: str
    answer: str
    citations: list[str]


# ---------------------------------------------------------------------------
# Lazy index loading
# ---------------------------------------------------------------------------

def get_retriever() -> HybridRetriever:
    """Get or build the global retriever instance."""
    global _retriever

    if _retriever is not None:
        return _retriever

    kb_dir = os.getenv("KB_DIR", "kb")
    chunks_path = "artifacts/chunks.json"

    # Try to load pre-built chunks first
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        # Build chunks from documents
        documents = load_documents(kb_dir)
        chunks = chunk_documents(documents, strategy="fixed_token", max_tokens=512, overlap_pct=0.15)

    # Build retriever
    _retriever = HybridRetriever()
    _retriever.build_index(chunks)

    return _retriever


# ---------------------------------------------------------------------------
# Deterministic answer generation (no LLM for API)
# ---------------------------------------------------------------------------

def generate_deterministic_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate an answer using deterministic extraction from retrieved chunks.

    This is used for the API endpoint to avoid LLM latency and cost.
    """
    import re

    if not retrieved_chunks:
        return {
            "answer_label": "insufficient_context",
            "answer": "insufficient_context",
            "citations": [],
        }

    # Tokenize question
    question_words = set(re.findall(r'[a-z]+', question.lower()))
    # Remove common words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need",
        "must", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further",
        "then", "once", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no", "only",
        "own", "same", "than", "too", "very", "just", "it", "its",
        "this", "that", "these", "those", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her",
        "they", "them", "their", "what", "which", "who", "whom",
        "how", "when", "where", "why",
    }
    question_words -= stop_words

    if not question_words:
        return {
            "answer_label": "insufficient_context",
            "answer": "insufficient_context",
            "citations": [],
        }

    # Score each chunk by keyword overlap with question
    best_chunk = None
    best_score = 0

    for chunk in retrieved_chunks:
        chunk_text = chunk.get("chunk_text", "")
        chunk_words = set(re.findall(r'[a-z]+', chunk_text.lower()))
        overlap = question_words & chunk_words
        score = len(overlap) / len(question_words) if question_words else 0

        if score > best_score:
            best_score = score
            best_chunk = chunk

    # If no good match, return insufficient_context
    if best_score < 0.2 or best_chunk is None:
        return {
            "answer_label": "insufficient_context",
            "answer": "insufficient_context",
            "citations": [],
        }

    # Extract the most relevant sentence from the best chunk
    chunk_text = best_chunk["chunk_text"]
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)

    best_sentence = None
    best_sent_score = 0

    for sentence in sentences:
        sent_words = set(re.findall(r'[a-z]+', sentence.lower()))
        overlap = question_words & sent_words
        score = len(overlap) / len(question_words) if question_words else 0

        if score > best_sent_score:
            best_sent_score = score
            best_sentence = sentence

    if best_sentence is None:
        best_sentence = chunk_text

    # Build citation
    citation = f"[{best_chunk['doc_title']} §{best_chunk['chunk_id']}]"

    return {
        "answer_label": "grounded_answer",
        "answer": f"{best_sentence.strip()} {citation}",
        "citations": [citation],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Answer a question using the RAG pipeline.

    Input:
        {"question": "string"}

    Output:
        {
            "answer_label": "grounded_answer" | "insufficient_context" | "conflicting_context",
            "answer": "string",
            "citations": ["[doc_title §chunk_id]", ...]
        }
    """
    try:
        retriever = get_retriever()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialise retriever: {e}")

    # Retrieve relevant chunks
    try:
        retrieved = retriever.retrieve(request.question, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # Generate answer
    result = generate_deterministic_answer(request.question, retrieved)

    return AnswerResponse(**result)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
