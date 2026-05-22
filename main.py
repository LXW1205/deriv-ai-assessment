"""
Deriv AI Assessment — Mini-RAG Pipeline
=========================================

Replayable mini-RAG pipeline that:
1. Ingests a product knowledge base from local files
2. Indexes it with hybrid retrieval (BM25 + Gemini embeddings)
3. Answers user questions with citation-strict responses
4. Evaluates retrieval quality deterministically

Usage:
    python main.py                  # Run full pipeline
    python main.py --help           # Show options
    python validate.py              # Validate artifacts
    python api.py                   # Start FastAPI server
"""

import argparse
import sys

from pipeline import run_pipeline


def main():
    """Entry point for the mini-RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="Mini-RAG Pipeline — Ingest, retrieve, answer, evaluate"
    )
    parser.add_argument(
        "--kb-dir",
        default="kb",
        help="Knowledge base directory (default: kb)",
    )
    parser.add_argument(
        "--queries",
        default="queries.json",
        help="Queries JSON file (default: queries.json)",
    )
    parser.add_argument(
        "--no-chunking-comparison",
        action="store_true",
        help="Skip chunking comparison stage",
    )
    parser.add_argument(
        "--no-grounding-check",
        action="store_true",
        help="Skip grounding check stage",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            kb_dir=args.kb_dir,
            queries_path=args.queries,
            run_chunking_comparison=not args.no_chunking_comparison,
            run_grounding_check=not args.no_grounding_check,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Pipeline failed — {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
