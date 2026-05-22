# Mini-RAG Pipeline

A replayable mini-RAG pipeline that ingests a product knowledge base, indexes it with hybrid retrieval, answers user questions with citation-strict responses, and evaluates retrieval quality deterministically.

## Architecture

```
kb/*.txt ──┐
            ├──► Ingestion ──► Chunking ──► Hybrid Index (BM25 + Gemini Embeddings)
queries.json┘                                                              │
                                                                           ▼
                                                                    Retrieval (top-k)
                                                                           │
                                    ┌──────────────────────────────────────┤
                                    ▼                                      ▼
                            Answer Generation                      Evaluation
                            (Gemini API)                      (Deterministic)
                                    │                                      │
                                    ▼                                      ▼
                            artifacts/answers.json                artifacts/eval.json
```

## Pipeline Stages

The pipeline enforces strict stage ordering:

```
INIT → DOCUMENTS_LOADED → DOCUMENTS_CHUNKED → INDEX_BUILT
→ RETRIEVAL_COMPLETE → ANSWERS_GENERATED → EVALUATION_COMPLETE
→ VALIDATION_COMPLETE → RESULTS_FINALISED
```

Final answers are never generated before retrieval completes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the full pipeline
python main.py

# Validate artifacts
python validate.py

# Start the API server (optional)
python api.py
```

Or using Make:

```bash
make install
make run
make validate
make api
```

## Artifacts

After running the pipeline, the following artifacts are generated:

| File | Description |
|------|-------------|
| `artifacts/chunks.json` | All document chunks with metadata |
| `artifacts/retrieval.json` | Top-k retrieval results per query |
| `artifacts/answers.json` | Citation-strict answers per query |
| `artifacts/eval.json` | Retrieval evaluation with aggregate summary |
| `artifacts/grounding_check.json` | Citation grounding validation |
| `artifacts/chunking_comparison.json` | Comparison of chunking strategies |
| `llm_calls.jsonl` | LLM call log (one JSON object per call) |

## Pipeline Components

### 1. Document Ingestion & Chunking (`ingestion.py`)

- Loads all `.txt` files from `kb/` directory
- Parses `Title:` and `Section:` headers
- Two chunking strategies:
  - **fixed_token**: 512 tokens with 15% overlap
  - **paragraph**: Paragraph-based with sentence splitting for long paragraphs

### 2. Hybrid Retrieval (`retrieval.py`)

- **BM25**: Pure Python implementation for keyword-based scoring
- **Gemini Embeddings**: `text-embedding-004` model for semantic similarity
- **Hybrid scoring**: Weighted combination (50/50) of normalised BM25 and cosine similarity scores

### 3. Answer Generation (`generation.py`)

- Uses Gemini API (`gemini-2.5-flash`) for grounded answer generation
- Prompts enforce citation-strict responses
- Controlled vocabulary: `grounded_answer`, `insufficient_context`, `conflicting_context`
- All LLM calls logged to `llm_calls.jsonl`

### 4. Deterministic Evaluation (`evaluation.py`)

- Compares retrieved documents against `expected_doc_titles` from `queries.json`
- Status labels: `hit`, `partial_hit`, `miss`
- Aggregate summary with `top3_hit_rate`

### 5. Grounding Check (`grounding.py`)

- Validates each citation exists in retrieval results
- Checks keyword and phrase overlap between answer and cited chunk
- Heuristic scoring for text support

### 6. Chunking Comparison (`chunking_comparison.py`)

- Runs full pipeline with both chunking strategies
- Compares hit rates and provides tradeoff analysis

### 7. FastAPI Server (`api.py`)

- `POST /answer` endpoint for real-time Q&A
- Returns citation-strict answers
- `GET /health` for health checks

## Controlled Vocabularies

### Answer Labels
- `grounded_answer`
- `insufficient_context`
- `conflicting_context`

### Retrieval Statuses
- `hit`
- `partial_hit`
- `miss`

### Citation Format
- `[doc_title §chunk_id]`

## Validation

```bash
python validate.py
```

Checks:
- All required artifacts exist
- JSON files are valid
- All queries were processed
- Each query has ≥3 retrieved chunks
- Retrieval scores are numeric
- Answer labels use controlled vocabulary
- Grounded answers include citations
- Citations refer only to retrieved chunks
- Evaluation statuses use controlled vocabulary
- Aggregate evaluation summary is present

## Configuration

Environment variables (in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Gemini API key | Required |
| `KB_DIR` | Knowledge base directory | `kb` |
| `QUERIES_PATH` | Queries JSON file | `queries.json` |

## Notes

- No LLM? The pipeline can run without an LLM for retrieval and evaluation. Answer generation will fall back to `insufficient_context` if the API is unavailable.
- The API endpoint uses deterministic extraction (no LLM) for fast responses.
- The pipeline works with any `.txt` files in `kb/` — filenames and order don't matter.
