"""
Deterministic retrieval evaluation.

Evaluates retrieval quality using ground truth from queries.json.
Computes hit/partial_hit/miss status and aggregate metrics.
"""

import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ALLOWED_RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retrieval_results: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k_threshold: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate retrieval results against ground truth.

    Args:
        retrieval_results: List of retrieval result dicts.
        queries: Original queries with expected_doc_titles.
        top_k_threshold: Number of top results to check for hits.

    Returns:
        Tuple of (per-query evaluations, aggregate summary).
    """
    # Build query lookup
    query_map = {q["query_id"]: q for q in queries}

    evaluations = []
    hits = 0
    partial_hits = 0
    misses = 0

    for result in retrieval_results:
        query_id = result["query_id"]
        query_info = query_map.get(query_id)

        if not query_info:
            continue

        expected_titles = query_info.get("expected_doc_titles", [])
        top_k = result.get("top_k", [])

        # Get retrieved doc titles from top_k
        retrieved_titles_top3 = [
            chunk["doc_title"] for chunk in top_k[:top_k_threshold]
        ]

        # Check how many expected titles are found
        matched_count = 0
        matched_expected = []
        for expected in expected_titles:
            if expected in retrieved_titles_top3:
                matched_count += 1
                matched_expected.append(expected)

        # Determine retrieval status
        if matched_count == len(expected_titles) and len(expected_titles) > 0:
            status = "hit"
            hits += 1
            # Find the rank of the first matched title
            first_match_rank = None
            for expected in expected_titles:
                if expected in retrieved_titles_top3:
                    rank = retrieved_titles_top3.index(expected) + 1
                    if first_match_rank is None or rank < first_match_rank:
                        first_match_rank = rank
            explanation = f"Expected title(s) found at rank(s) {first_match_rank}"
        elif matched_count > 0:
            status = "partial_hit"
            partial_hits += 1
            matched_str = ", ".join(matched_expected)
            explanation = f"Found {matched_count}/{len(expected_titles)} expected title(s): {matched_str}"
        else:
            status = "miss"
            misses += 1
            explanation = f"None of the expected title(s) found in top {top_k_threshold}"

        evaluations.append({
            "query_id": query_id,
            "expected_doc_titles": expected_titles,
            "retrieved_doc_titles_top3": retrieved_titles_top3,
            "retrieval_status": status,
            "matched_expected_title": matched_count > 0,
            "explanation": explanation,
        })

    total = len(evaluations)
    summary = {
        "top3_hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        "total_queries": total,
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses,
    }

    return evaluations, summary


def save_evaluation(
    evaluations: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: str,
) -> None:
    """Save evaluation results to a JSON file."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    output = {
        "evaluations": evaluations,
        "summary": summary,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_evaluation(input_path: str) -> dict[str, Any]:
    """Load evaluation results from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
