"""
Citation-strict answer generation using Gemini API.

Generates answers grounded ONLY in retrieved chunks.
If context is insufficient, returns 'insufficient_context'.
All factual claims must include citations.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ALLOWED_ANSWER_LABELS = {"grounded_answer", "insufficient_context", "conflicting_context"}
CITATION_FORMAT = "[{doc_title} §{chunk_id}]"


# ---------------------------------------------------------------------------
# LLM call logging
# ---------------------------------------------------------------------------

LLM_CALLS_PATH = "llm_calls.jsonl"


def log_llm_call(
    stage: str,
    query_id: str | None,
    provider: str,
    model: str,
    prompt: str,
    input_artifacts: list[str],
    output_artifact: str,
) -> None:
    """Log an LLM call to the JSONL file."""
    record = {
        "stage": stage,
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_artifacts": input_artifacts,
        "output_artifact": output_artifact,
    }

    os.makedirs(os.path.dirname(LLM_CALLS_PATH) if os.path.dirname(LLM_CALLS_PATH) else ".", exist_ok=True)
    with open(LLM_CALLS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """Build a prompt for grounded answer generation.

    The prompt instructs the LLM to:
    1. Answer ONLY using the provided context
    2. Include citations for every factual claim
    3. Say 'insufficient_context' if the context doesn't support an answer
    """
    context_parts = []
    for chunk in retrieved_chunks:
        citation = CITATION_FORMAT.format(
            doc_title=chunk["doc_title"],
            chunk_id=chunk["chunk_id"],
        )
        context_parts.append(f"{citation}\n{chunk['chunk_text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a customer support assistant answering questions based ONLY on the provided knowledge base context.

QUESTION: {question}

CONTEXT:
{context_block}

INSTRUCTIONS:
1. Answer the question using ONLY the information in the CONTEXT above.
2. If the CONTEXT does not contain enough information to answer, respond with exactly: "insufficient_context"
3. If you can answer, provide a concise answer grounded in the context.
4. Every factual statement must include a citation in the format: [doc_title §chunk_id]
5. Use the exact doc_title and chunk_id from the CONTEXT.
6. Do not invent facts, speculate, or use outside knowledge.
7. Do not cite chunks that were not provided in the CONTEXT.

ANSWER:"""

    return prompt


def generate_answer_with_gemini(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    query_id: str | None = None,
) -> dict[str, Any]:
    """Generate a citation-strict answer using Gemini API.

    Returns a dict with:
        - query_id
        - answer_label
        - answer
        - citations
        - used_chunk_ids
    """
    import google.genai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(question, retrieved_chunks)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.1},
        )
        answer_text = response.text.strip()
    except Exception as e:
        # Fallback: if API fails, return insufficient_context
        answer_text = "insufficient_context"

    # Parse the response
    return parse_answer_response(
        query_id=query_id or "unknown",
        answer_text=answer_text,
        retrieved_chunks=retrieved_chunks,
    )


def parse_answer_response(
    query_id: str,
    answer_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse the LLM response into a structured answer record.

    Extracts citations and validates against retrieved chunks.
    """
    # Check for insufficient context
    if answer_text.lower() == "insufficient_context":
        return {
            "query_id": query_id,
            "answer_label": "insufficient_context",
            "answer": "insufficient_context",
            "citations": [],
            "used_chunk_ids": [],
        }

    # Extract citations from the answer text
    import re
    citation_pattern = r'\[([^\]]+)\s§([^\]]+)\]'
    citations = re.findall(citation_pattern, answer_text)

    # Build citation strings and used chunk IDs
    citation_strings = []
    used_chunk_ids = []
    valid_chunk_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}

    for doc_title, chunk_id in citations:
        chunk_id = chunk_id.strip()
        if chunk_id in valid_chunk_ids:
            citation_str = f"[{doc_title.strip()} §{chunk_id}]"
            citation_strings.append(citation_str)
            if chunk_id not in used_chunk_ids:
                used_chunk_ids.append(chunk_id)

    # If no valid citations found but answer is not insufficient_context,
    # label it as insufficient_context (ungrounded)
    if not citation_strings and answer_text.lower() != "insufficient_context":
        return {
            "query_id": query_id,
            "answer_label": "insufficient_context",
            "answer": answer_text,
            "citations": [],
            "used_chunk_ids": [],
        }

    return {
        "query_id": query_id,
        "answer_label": "grounded_answer",
        "answer": answer_text,
        "citations": citation_strings,
        "used_chunk_ids": used_chunk_ids,
    }


def generate_answers(
    retrieval_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate answers for all queries.

    Args:
        retrieval_results: List of retrieval result dicts.
    """
    answers = []

    for result in retrieval_results:
        query_id = result["query_id"]
        question = result["question"]
        retrieved_chunks = result["top_k"]

        answer_record = generate_answer_with_gemini(
            question=question,
            retrieved_chunks=retrieved_chunks,
            query_id=query_id,
        )

        # Log the LLM call
        prompt = build_prompt(question, retrieved_chunks)
        log_llm_call(
            stage="answer_generation",
            query_id=query_id,
            provider="google",
            model="gemini-2.5-flash",
            prompt=prompt,
            input_artifacts=["artifacts/retrieval.json"],
            output_artifact="artifacts/answers.json",
        )

        answers.append(answer_record)

    return answers


def save_answers(answers: list[dict[str, Any]], output_path: str) -> None:
    """Save answers to a JSON file."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)


def load_answers(input_path: str) -> list[dict[str, Any]]:
    """Load answers from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
