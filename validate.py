"""
Validation script for the mini-RAG pipeline.

Checks that:
- Required artifacts exist
- JSON files are valid
- All queries were processed
- Each query has at least 3 retrieved chunks
- Retrieval scores are numeric
- Answer labels use only the controlled vocabulary
- Grounded answers include citations
- Citations refer only to retrieved chunks
- Evaluation statuses use only the controlled vocabulary
- Aggregate evaluation summary is present
"""

import json
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ALLOWED_ANSWER_LABELS = {"grounded_answer", "insufficient_context", "conflicting_context"}
ALLOWED_RETRIEVAL_STATUSES = {"hit", "partial_hit", "miss"}

REQUIRED_ARTIFACTS = [
    "artifacts/chunks.json",
    "artifacts/retrieval.json",
    "artifacts/answers.json",
    "artifacts/eval.json",
]

OPTIONAL_ARTIFACTS = [
    "artifacts/grounding_check.json",
    "artifacts/chunking_comparison.json",
]


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

class ValidationResult:
    """Tracks validation results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        """Record a validation check."""
        if condition:
            self.passed += 1
            print(f"  PASS: {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  FAIL: {message}")

    def summary(self) -> bool:
        """Print summary and return overall pass/fail."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Validation Results: {self.passed}/{total} checks passed")
        if self.errors:
            print(f"\nFailed checks:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*60}")
        return self.failed == 0


def load_json(path: str) -> Any:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_artifacts_exist(result: ValidationResult) -> None:
    """Check that all required artifacts exist."""
    print("\n--- Checking artifact existence ---")

    for artifact in REQUIRED_ARTIFACTS:
        result.check(
            os.path.exists(artifact),
            f"Required artifact exists: {artifact}"
        )

    for artifact in OPTIONAL_ARTIFACTS:
        if os.path.exists(artifact):
            result.check(True, f"Optional artifact exists: {artifact}")
        else:
            print(f"  SKIP: Optional artifact not found: {artifact}")


def validate_json_valid(result: ValidationResult) -> None:
    """Check that all JSON files are valid."""
    print("\n--- Checking JSON validity ---")

    all_artifacts = REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS
    for artifact in all_artifacts:
        if not os.path.exists(artifact):
            continue
        try:
            load_json(artifact)
            result.check(True, f"Valid JSON: {artifact}")
        except json.JSONDecodeError as e:
            result.check(False, f"Invalid JSON: {artifact} — {e}")


def validate_queries_processed(result: ValidationResult) -> None:
    """Check that all queries were processed."""
    print("\n--- Checking query processing ---")

    queries = load_json("queries.json")
    retrieval = load_json("artifacts/retrieval.json")
    answers = load_json("artifacts/answers.json")

    query_ids = {q["query_id"] for q in queries}
    retrieval_ids = {r["query_id"] for r in retrieval}
    answer_ids = {a["query_id"] for a in answers}

    result.check(
        query_ids == retrieval_ids,
        f"All queries have retrieval results (expected {len(query_ids)}, got {len(retrieval_ids)})"
    )
    result.check(
        query_ids == answer_ids,
        f"All queries have answers (expected {len(query_ids)}, got {len(answer_ids)})"
    )


def validate_retrieval_chunks(result: ValidationResult) -> None:
    """Check that each query has at least 3 retrieved chunks."""
    print("\n--- Checking retrieval chunk counts ---")

    retrieval = load_json("artifacts/retrieval.json")

    for r in retrieval:
        top_k = r.get("top_k", [])
        result.check(
            len(top_k) >= 3,
            f"Query {r['query_id']} has {len(top_k)} retrieved chunks (minimum 3)"
        )


def validate_retrieval_scores(result: ValidationResult) -> None:
    """Check that retrieval scores are numeric."""
    print("\n--- Checking retrieval scores ---")

    retrieval = load_json("artifacts/retrieval.json")

    for r in retrieval:
        for chunk in r.get("top_k", []):
            score = chunk.get("score")
            result.check(
                isinstance(score, (int, float)),
                f"Query {r['query_id']}, chunk {chunk.get('chunk_id')}: score is numeric ({score})"
            )


def validate_answer_labels(result: ValidationResult) -> None:
    """Check that answer labels use only the controlled vocabulary."""
    print("\n--- Checking answer labels ---")

    answers = load_json("artifacts/answers.json")

    for a in answers:
        label = a.get("answer_label")
        result.check(
            label in ALLOWED_ANSWER_LABELS,
            f"Query {a['query_id']}: answer_label '{label}' is in allowed vocabulary"
        )


def validate_citations(result: ValidationResult) -> None:
    """Check that grounded answers include citations."""
    print("\n--- Checking citations ---")

    answers = load_json("artifacts/answers.json")
    retrieval = load_json("artifacts/retrieval.json")

    # Build lookup: query_id -> retrieved chunk_ids
    retrieval_map = {}
    for r in retrieval:
        retrieval_map[r["query_id"]] = {
            chunk["chunk_id"] for chunk in r.get("top_k", [])
        }

    for a in answers:
        query_id = a["query_id"]
        label = a.get("answer_label")
        citations = a.get("citations", [])
        used_chunk_ids = a.get("used_chunk_ids", [])

        if label == "grounded_answer":
            result.check(
                len(citations) > 0,
                f"Query {query_id}: grounded_answer has citations"
            )

            # Check citations refer only to retrieved chunks
            retrieved_ids = retrieval_map.get(query_id, set())
            for chunk_id in used_chunk_ids:
                result.check(
                    chunk_id in retrieved_ids,
                    f"Query {query_id}: cited chunk '{chunk_id}' was retrieved"
                )
        else:
            result.check(
                len(citations) == 0,
                f"Query {query_id}: non-grounded answer has no citations"
            )


def validate_evaluation_statuses(result: ValidationResult) -> None:
    """Check that evaluation statuses use only the controlled vocabulary."""
    print("\n--- Checking evaluation statuses ---")

    eval_data = load_json("artifacts/eval.json")
    evaluations = eval_data.get("evaluations", [])

    for e in evaluations:
        status = e.get("retrieval_status")
        result.check(
            status in ALLOWED_RETRIEVAL_STATUSES,
            f"Query {e['query_id']}: retrieval_status '{status}' is in allowed vocabulary"
        )


def validate_evaluation_summary(result: ValidationResult) -> None:
    """Check that aggregate evaluation summary is present."""
    print("\n--- Checking evaluation summary ---")

    eval_data = load_json("artifacts/eval.json")
    summary = eval_data.get("summary")

    required_fields = ["top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"]

    if summary is None:
        result.check(False, "Evaluation summary is present")
        return

    result.check(True, "Evaluation summary is present")

    for field in required_fields:
        result.check(
            field in summary,
            f"Summary contains '{field}'"
        )


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate() -> bool:
    """Run all validation checks.

    Returns True if all checks pass, False otherwise.
    """
    result = ValidationResult()

    validate_artifacts_exist(result)
    validate_json_valid(result)
    validate_queries_processed(result)
    validate_retrieval_chunks(result)
    validate_retrieval_scores(result)
    validate_answer_labels(result)
    validate_citations(result)
    validate_evaluation_statuses(result)
    validate_evaluation_summary(result)

    return result.summary()


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
