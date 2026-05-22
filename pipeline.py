"""
Pipeline orchestrator.

Enforces the required stage sequence:
    INIT -> DOCUMENTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT
    -> RETRIEVAL_COMPLETE -> ANSWERS_GENERATED -> EVALUATION_COMPLETE
    -> VALIDATION_COMPLETE -> RESULTS_FINALISED
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from ingestion import load_documents, chunk_documents, save_chunks
from retrieval import HybridRetriever, run_retrieval, save_retrieval_results
from generation import generate_answers, save_answers
from evaluation import evaluate_retrieval, save_evaluation
from grounding import check_grounding, save_grounding_check
from chunking_comparison import compare_strategies


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KB_DIR = os.getenv("KB_DIR", "kb")
QUERIES_PATH = os.getenv("QUERIES_PATH", "queries.json")
ARTIFACTS_DIR = "artifacts"

CHUNKS_PATH = os.path.join(ARTIFACTS_DIR, "chunks.json")
RETRIEVAL_PATH = os.path.join(ARTIFACTS_DIR, "retrieval.json")
ANSWERS_PATH = os.path.join(ARTIFACTS_DIR, "answers.json")
EVAL_PATH = os.path.join(ARTIFACTS_DIR, "eval.json")
GROUNDING_PATH = os.path.join(ARTIFACTS_DIR, "grounding_check.json")
CHUNKING_COMPARISON_PATH = os.path.join(ARTIFACTS_DIR, "chunking_comparison.json")


# ---------------------------------------------------------------------------
# Stage tracking
# ---------------------------------------------------------------------------

STAGES = [
    "INIT",
    "DOCUMENTS_LOADED",
    "DOCUMENTS_CHUNKED",
    "INDEX_BUILT",
    "RETRIEVAL_COMPLETE",
    "ANSWERS_GENERATED",
    "EVALUATION_COMPLETE",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]


class PipelineState:
    """Tracks pipeline stage progression."""

    def __init__(self):
        self.current_stage = "INIT"
        self.stage_log: list[dict[str, Any]] = []
        self.start_time = time.time()

    def advance(self, stage: str, details: str = "") -> None:
        """Advance to the next stage."""
        expected_idx = STAGES.index(stage) if stage in STAGES else -1
        current_idx = STAGES.index(self.current_stage) if self.current_stage in STAGES else -1

        if expected_idx <= current_idx and stage != self.current_stage:
            raise ValueError(
                f"Invalid stage transition: {self.current_stage} -> {stage}. "
                f"Expected next stage: {STAGES[current_idx + 1] if current_idx + 1 < len(STAGES) else 'none'}"
            )

        elapsed = time.time() - self.start_time
        self.current_stage = stage
        self.stage_log.append({
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "details": details,
        })
        print(f"  [{stage}] {details}")

    def save_log(self, output_path: str) -> None:
        """Save stage log to a JSON file."""
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.stage_log, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(
    kb_dir: str = KB_DIR,
    queries_path: str = QUERIES_PATH,
    run_chunking_comparison: bool = True,
    run_grounding_check: bool = True,
) -> dict[str, Any]:
    """Execute the full RAG pipeline with stage enforcement.

    Args:
        kb_dir: Path to knowledge base directory.
        queries_path: Path to queries JSON file.
        run_chunking_comparison: Whether to run chunking comparison.
        run_grounding_check: Whether to run grounding check.
    """
    state = PipelineState()
    state.advance("INIT", "Pipeline starting")

    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Clear previous LLM calls log
    llm_calls_path = "llm_calls.jsonl"
    if os.path.exists(llm_calls_path):
        os.remove(llm_calls_path)

    # -------------------------------------------------------------------
    # Stage 1: Load documents
    # -------------------------------------------------------------------
    print("\n--- Loading documents ---")
    documents = load_documents(kb_dir)
    state.advance(
        "DOCUMENTS_LOADED",
        f"Loaded {len(documents)} documents from {kb_dir}"
    )

    # -------------------------------------------------------------------
    # Stage 2: Chunk documents
    # -------------------------------------------------------------------
    print("\n--- Chunking documents ---")
    chunks = chunk_documents(
        documents,
        strategy="fixed_token",
        max_tokens=512,
        overlap_pct=0.15,
    )
    save_chunks(chunks, CHUNKS_PATH)
    state.advance(
        "DOCUMENTS_CHUNKED",
        f"Created {len(chunks)} chunks (fixed_token, 512 tokens, 15% overlap). Saved to {CHUNKS_PATH}"
    )

    # -------------------------------------------------------------------
    # Stage 3: Build index
    # -------------------------------------------------------------------
    print("\n--- Building retrieval index ---")
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    state.advance("INDEX_BUILT", "Hybrid index built (BM25 + Gemini embeddings)")

    # -------------------------------------------------------------------
    # Stage 4: Load queries and run retrieval
    # -------------------------------------------------------------------
    print("\n--- Running retrieval ---")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    retrieval_results = run_retrieval(retriever, queries, top_k=5)
    save_retrieval_results(retrieval_results, RETRIEVAL_PATH)
    state.advance(
        "RETRIEVAL_COMPLETE",
        f"Retrieved top-5 chunks for {len(queries)} queries. Saved to {RETRIEVAL_PATH}"
    )

    # -------------------------------------------------------------------
    # Stage 5: Generate answers (must happen AFTER retrieval)
    # -------------------------------------------------------------------
    print("\n--- Generating answers ---")
    answers = generate_answers(retrieval_results)
    save_answers(answers, ANSWERS_PATH)
    state.advance(
        "ANSWERS_GENERATED",
        f"Generated {len(answers)} answers with citations. Saved to {ANSWERS_PATH}"
    )

    # -------------------------------------------------------------------
    # Stage 6: Evaluate retrieval
    # -------------------------------------------------------------------
    print("\n--- Evaluating retrieval ---")
    evaluations, summary = evaluate_retrieval(retrieval_results, queries)
    save_evaluation(evaluations, summary, EVAL_PATH)
    state.advance(
        "EVALUATION_COMPLETE",
        f"Evaluated {len(evaluations)} queries. "
        f"Hit rate: {summary['top3_hit_rate']:.1%} "
        f"({summary['hits']} hits, {summary['partial_hits']} partial, {summary['misses']} misses). "
        f"Saved to {EVAL_PATH}"
    )

    # -------------------------------------------------------------------
    # Stage 7: Grounding check (optional)
    # -------------------------------------------------------------------
    if run_grounding_check:
        print("\n--- Running grounding check ---")
        grounding_checks = check_grounding(answers, retrieval_results)
        save_grounding_check(grounding_checks, GROUNDING_PATH)
        state.advance(
            "VALIDATION_COMPLETE",
            f"Grounding check complete. Saved to {GROUNDING_PATH}"
        )
    else:
        state.advance(
            "VALIDATION_COMPLETE",
            "Grounding check skipped"
        )

    # -------------------------------------------------------------------
    # Stage 8: Chunking comparison (optional)
    # -------------------------------------------------------------------
    if run_chunking_comparison:
        print("\n--- Running chunking comparison ---")
        compare_strategies(kb_dir, queries, CHUNKING_COMPARISON_PATH)
        state.advance(
            "RESULTS_FINALISED",
            f"Chunking comparison complete. Saved to {CHUNKING_COMPARISON_PATH}"
        )
    else:
        state.advance(
            "RESULTS_FINALISED",
            "Chunking comparison skipped"
        )

    # -------------------------------------------------------------------
    # Save stage log
    # -------------------------------------------------------------------
    stage_log_path = os.path.join(ARTIFACTS_DIR, "pipeline_stages.json")
    state.save_log(stage_log_path)

    print(f"\n{'='*60}")
    print(f"Pipeline complete! All stages passed.")
    print(f"Total time: {time.time() - state.start_time:.1f}s")
    print(f"Artifacts saved to {ARTIFACTS_DIR}/")
    print(f"{'='*60}")

    return {
        "state": state,
        "documents": documents,
        "chunks": chunks,
        "retrieval_results": retrieval_results,
        "answers": answers,
        "evaluations": evaluations,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the pipeline from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Mini-RAG Pipeline")
    parser.add_argument("--kb-dir", default=KB_DIR, help="Knowledge base directory")
    parser.add_argument("--queries", default=QUERIES_PATH, help="Queries JSON file")
    parser.add_argument("--no-chunking-comparison", action="store_true", help="Skip chunking comparison")
    parser.add_argument("--no-grounding-check", action="store_true", help="Skip grounding check")

    args = parser.parse_args()

    run_pipeline(
        kb_dir=args.kb_dir,
        queries_path=args.queries,
        run_chunking_comparison=not args.no_chunking_comparison,
        run_grounding_check=not args.no_grounding_check,
    )


if __name__ == "__main__":
    main()
