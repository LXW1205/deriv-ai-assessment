"""
Chunking comparison module.

Runs the pipeline with two chunking strategies and compares
aggregate retrieval performance.
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from ingestion import load_documents, chunk_documents, save_chunks
from retrieval import HybridRetriever, run_retrieval, save_retrieval_results
from evaluation import evaluate_retrieval


def run_pipeline_with_strategy(
    kb_dir: str,
    queries: list[dict[str, Any]],
    strategy: str,
    top_k: int = 5,
    **chunk_kwargs,
) -> dict[str, Any]:
    """Run the full ingestion -> retrieval -> evaluation pipeline.

    Args:
        kb_dir: Path to knowledge base directory.
        queries: List of query dicts.
        strategy: Chunking strategy name.
        top_k: Number of top chunks to retrieve.
        **chunk_kwargs: Additional chunking parameters.
    """
    start_time = time.time()

    # Load documents
    documents = load_documents(kb_dir)

    # Chunk documents
    chunks = chunk_documents(documents, strategy=strategy, **chunk_kwargs)

    # Build retriever
    retriever = HybridRetriever()
    retriever.build_index(chunks)

    # Run retrieval
    retrieval_results = run_retrieval(retriever, queries, top_k=top_k)

    # Evaluate
    evaluations, summary = evaluate_retrieval(retrieval_results, queries)

    elapsed = time.time() - start_time

    return {
        "strategy": strategy,
        "chunk_kwargs": chunk_kwargs,
        "num_chunks": len(chunks),
        "num_documents": len(documents),
        "retrieval_results": retrieval_results,
        "evaluations": evaluations,
        "summary": summary,
        "elapsed_seconds": round(elapsed, 2),
    }


def compare_strategies(
    kb_dir: str,
    queries: list[dict[str, Any]],
    output_path: str = "artifacts/chunking_comparison.json",
) -> dict[str, Any]:
    """Compare two chunking strategies and save results.

    Strategies compared:
    1. fixed_token: 512 tokens, 15% overlap
    2. paragraph: paragraph-based chunking
    """
    print("\n=== Chunking Comparison ===")

    # Strategy 1: Fixed token (512 tokens, 15% overlap)
    print("\nRunning with fixed_token strategy (512 tokens, 15% overlap)...")
    result_fixed = run_pipeline_with_strategy(
        kb_dir=kb_dir,
        queries=queries,
        strategy="fixed_token",
        max_tokens=512,
        overlap_pct=0.15,
    )

    # Strategy 2: Paragraph-based
    print("Running with paragraph strategy...")
    result_paragraph = run_pipeline_with_strategy(
        kb_dir=kb_dir,
        queries=queries,
        strategy="paragraph",
    )

    # Compare results
    comparison = {
        "strategies": [
            {
                "name": result_fixed["strategy"],
                "config": result_fixed["chunk_kwargs"],
                "num_chunks": result_fixed["num_chunks"],
                "hit_rate": result_fixed["summary"]["top3_hit_rate"],
                "hits": result_fixed["summary"]["hits"],
                "partial_hits": result_fixed["summary"]["partial_hits"],
                "misses": result_fixed["summary"]["misses"],
                "elapsed_seconds": result_fixed["elapsed_seconds"],
            },
            {
                "name": result_paragraph["strategy"],
                "config": result_paragraph["chunk_kwargs"],
                "num_chunks": result_paragraph["num_chunks"],
                "hit_rate": result_paragraph["summary"]["top3_hit_rate"],
                "hits": result_paragraph["summary"]["hits"],
                "partial_hits": result_paragraph["summary"]["partial_hits"],
                "misses": result_paragraph["summary"]["misses"],
                "elapsed_seconds": result_paragraph["elapsed_seconds"],
            },
        ],
        "tradeoff_analysis": _analyze_tradeoff(result_fixed, result_paragraph),
    }

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nChunking comparison saved to {output_path}")
    return comparison


def _analyze_tradeoff(
    result_fixed: dict[str, Any],
    result_paragraph: dict[str, Any],
) -> str:
    """Generate a brief tradeoff analysis between strategies."""
    fixed_rate = result_fixed["summary"]["top3_hit_rate"]
    para_rate = result_paragraph["summary"]["top3_hit_rate"]

    fixed_chunks = result_fixed["num_chunks"]
    para_chunks = result_paragraph["num_chunks"]

    winner = "fixed_token" if fixed_rate >= para_rate else "paragraph"

    analysis = (
        f"Tradeoff Analysis:\n"
        f"- Fixed-token strategy (512 tokens, 15% overlap) produced {fixed_chunks} chunks "
        f"with a top-3 hit rate of {fixed_rate:.1%}.\n"
        f"- Paragraph-based strategy produced {para_chunks} chunks "
        f"with a top-3 hit rate of {para_rate:.1%}.\n"
        f"- The {winner} strategy performed better in retrieval quality.\n"
        f"\n"
        f"Tradeoffs:\n"
        f"- Fixed-token chunks provide consistent size and controlled overlap, "
        f"which helps ensure no information is lost at boundaries. "
        f"However, they may split coherent paragraphs.\n"
        f"- Paragraph-based chunks preserve semantic boundaries, keeping related "
        f"sentences together. However, chunk sizes vary significantly, "
        f"which can affect embedding quality and retrieval consistency.\n"
        f"\n"
        f"Recommendation: For this knowledge base, {winner} is preferred "
        f"due to {'higher retrieval accuracy' if fixed_rate != para_rate else 'comparable performance with fewer chunks'}."
    )

    return analysis
