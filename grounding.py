"""
Grounding check validator.

Verifies that each citation in answers.json:
1. Exists in retrieval.json
2. The cited chunk text contains supporting wording for the answer

Uses heuristic phrase overlap and keyword matching.
"""

import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# Grounding validation
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text."""
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "need", "must", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "not",
        "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just",
        "it", "its", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "they", "them", "their", "what", "which", "who",
        "whom", "how", "when", "where", "why",
    }
    words = re.findall(r'[a-z]+', text.lower())
    return {w for w in words if w not in stop_words and len(w) > 2}


def check_grounding(
    answers: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Check grounding of each answer against retrieved chunks.

    Returns a list of grounding check results.
    """
    # Build lookup: query_id -> retrieved chunks
    retrieval_map = {}
    for result in retrieval_results:
        query_id = result["query_id"]
        retrieval_map[query_id] = {
            chunk["chunk_id"]: chunk for chunk in result.get("top_k", [])
        }

    grounding_checks = []

    for answer in answers:
        query_id = answer["query_id"]
        citations = answer.get("citations", [])
        used_chunk_ids = answer.get("used_chunk_ids", [])
        answer_text = answer.get("answer", "")
        answer_label = answer.get("answer_label", "")

        # Skip insufficient_context answers
        if answer_label == "insufficient_context":
            grounding_checks.append({
                "query_id": query_id,
                "answer_label": answer_label,
                "all_citations_valid": True,
                "citation_checks": [],
                "explanation": "No citations required for insufficient_context",
            })
            continue

        retrieved_chunks = retrieval_map.get(query_id, {})
        citation_checks = []
        all_valid = True

        for citation in citations:
            # Parse chunk_id from citation [doc_title §chunk_id]
            match = re.match(r'\[.*§(.+?)\]', citation)
            if not match:
                citation_checks.append({
                    "citation": citation,
                    "chunk_exists_in_retrieval": False,
                    "text_supports_answer": False,
                    "valid": False,
                    "explanation": "Could not parse chunk_id from citation",
                })
                all_valid = False
                continue

            chunk_id = match.group(1).strip()

            # Check 1: Does the chunk exist in retrieval?
            chunk_exists = chunk_id in retrieved_chunks

            if not chunk_exists:
                citation_checks.append({
                    "citation": citation,
                    "chunk_exists_in_retrieval": False,
                    "text_supports_answer": False,
                    "valid": False,
                    "explanation": f"Chunk '{chunk_id}' not found in retrieval results",
                })
                all_valid = False
                continue

            # Check 2: Does the chunk text support the answer?
            chunk = retrieved_chunks[chunk_id]
            chunk_text = chunk.get("chunk_text", "")

            # Extract keywords from answer (excluding citation parts)
            answer_clean = re.sub(r'\[.*?\]', '', answer_text)
            answer_keywords = extract_keywords(answer_clean)
            chunk_keywords = extract_keywords(chunk_text)

            # Check keyword overlap
            if answer_keywords:
                overlap = answer_keywords & chunk_keywords
                overlap_ratio = len(overlap) / len(answer_keywords)
            else:
                overlap_ratio = 1.0

            # Check if key phrases from answer appear in chunk
            # Extract meaningful phrases (3+ words) from answer
            answer_phrases = re.findall(r'\b\w+(?:\s+\w+){2,}\b', answer_clean)
            phrase_support = 0
            phrase_total = 0

            for phrase in answer_phrases:
                phrase_total += 1
                if phrase.lower() in chunk_text.lower():
                    phrase_support += 1

            phrase_ratio = phrase_support / phrase_total if phrase_total > 0 else 1.0

            # Determine if text supports answer
            text_supports = (
                overlap_ratio >= 0.3 or
                phrase_ratio >= 0.5 or
                len(answer_keywords & chunk_keywords) >= 2
            )

            if not text_supports:
                all_valid = False

            citation_checks.append({
                "citation": citation,
                "chunk_exists_in_retrieval": True,
                "text_supports_answer": text_supports,
                "keyword_overlap_ratio": round(overlap_ratio, 3),
                "phrase_support_ratio": round(phrase_ratio, 3),
                "valid": text_supports,
                "explanation": (
                    f"Keyword overlap: {overlap_ratio:.1%}, "
                    f"Phrase support: {phrase_ratio:.1%}"
                ),
            })

        grounding_checks.append({
            "query_id": query_id,
            "answer_label": answer_label,
            "all_citations_valid": all_valid,
            "citation_checks": citation_checks,
            "explanation": (
                "All citations valid and grounded" if all_valid
                else "Some citations are not properly grounded in retrieved chunks"
            ),
        })

    return grounding_checks


def save_grounding_check(
    grounding_checks: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Save grounding check results to a JSON file."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Compute summary
    total = len(grounding_checks)
    valid = sum(1 for g in grounding_checks if g["all_citations_valid"])

    summary = {
        "total_answers": total,
        "fully_grounded": valid,
        "partially_grounded": total - valid,
        "grounding_rate": round(valid / total, 4) if total > 0 else 0.0,
    }

    output = {
        "grounding_checks": grounding_checks,
        "summary": summary,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_grounding_check(input_path: str) -> dict[str, Any]:
    """Load grounding check results from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
