# DocChat — Agentic RAG over FDA Drug Labels

> **Verified document Q&A with LangGraph orchestration, hybrid retrieval, and dual safety guards against hallucination and out-of-scope answers.**

[![CI](https://github.com/ehabelshridy/docchat-agentic/actions/workflows/ci.yml/badge.svg)](https://github.com/ehabelshridy/docchat-agentic/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

DocChat is a production-style **Corrective Agentic RAG** system built on [LangGraph](https://github.com/langchain-ai/langgraph). It answers clinical questions exclusively from indexed FDA drug-label documents — refusing to hallucinate or answer outside its knowledge base — and shows users a live verification trail (relevance status, groundedness status, retry count, source citations) for every response.

![Demo](/demo.gif)

The project was built as a portfolio piece demonstrating:
- **Agentic pipeline design** with conditional edges and self-correcting loops
- **Hybrid retrieval** (BM25 keyword + ChromaDB semantic + RRF fusion)
- **Dual LLM-as-judge safety layers** (relevance grading *per chunk* + groundedness verification)
- **FastAPI + vanilla JS frontend** served from a single process

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐
│   retrieve  │  Hybrid retrieval: BM25 + ChromaDB dense search,
│    node     │  results fused with Reciprocal Rank Fusion (RRF)
└──────┬──────┘
       │
       ▼
┌─────────────────┐     irrelevant chunk      ┌───────────────┐
│ grade_relevance │ ─────────────────────────►│ rewrite_query │
│     node        │  (per-chunk, temp=0)       │     node      │
│                 │◄──────────────────────────│  (retry loop) │
└──────┬──────────┘     rewritten query       └───────────────┘
       │ all chunks irrelevant
       │ after MAX_RETRIES
       │                        ┌──────────────────┐
       │ at least 1 relevant    │                  │
       ▼                        │   fallback node  │
┌─────────────────┐             │ "out of scope"   │
│ generate_answer │             │                  │
│     node        │             └──────────────────┘
└──────┬──────────┘                     ▲
       │                                │
       ▼                                │
┌──────────────────┐   not grounded     │
│ check_groundedness├───────────────────┘ (after MAX_RETRIES
│     node         │   (temp=0)            → "unverified")
└──────┬───────────┘
       │ grounded
       ▼
  Final response
  + source citations
  + verification status strip (shown in UI)
```

**Two independent retry budgets** (`MAX_RELEVANCE_RETRIES = 2`, `MAX_GENERATION_RETRIES = 2`) guarantee both loops always terminate — either with a verified answer or a transparent fallback message, never an infinite loop or a hallucinated guess.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph `StateGraph` with conditional edges |
| **LLM** | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via HF Inference API |
| **Document parsing** | [Docling](https://github.com/DS4SD/docling) (PDF, DOCX, HTML, TXT) |
| **Chunking** | Docling `HybridChunker` (structure-aware, respects section headings) |
| **Vector store** | ChromaDB (persistent, local) |
| **Keyword search** | BM25Okapi (`rank-bm25`) with stopword filtering |
| **Retrieval fusion** | Reciprocal Rank Fusion (RRF) |
| **Backend** | FastAPI + Uvicorn (also serves the frontend as static files) |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **Data source** | [openFDA Drug Label API](https://open.fda.gov/apis/drug/label/) |

---

## Project Structure

```
docchat-agentic/
│
├── backend/                   # FastAPI app + LangGraph pipeline
│   ├── main.py                # FastAPI entry point; serves frontend at /
│   ├── graph.py               # LangGraph StateGraph (all nodes + edges)
│   ├── retrieval.py           # HybridRetriever: BM25 + ChromaDB + RRF
│   └── llm.py                 # HF Inference API calls (embeddings + chat)
│
├── frontend/                  # Single-page chat UI (no build step needed)
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── scripts/
│   ├── fetch_fda_labels.py    # Download drug labels from openFDA API → data/raw_labels/
│   ├── ingest_pipeline.py     # Docling → chunk → embed → ChromaDB + BM25
│   └── diagnostics/           # Standalone debug scripts (not part of app)
│       ├── diagnose_retrieval.py
│       ├── diagnose_relevance_judge.py
│       └── diagnose_groundedness.py
│
├── .env.example               # Required environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ehabelshridy/DocChat-agentic.git
cd DocChat-agentic

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```
# Open .env and set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
Enable the **"Make calls to Inference Providers"** permission.

### 3. Fetch FDA drug labels

```bash
# Download 100 drug labels from the openFDA API
python scripts/fetch_fda_labels.py --limit 100 --out data/raw_labels

# Or search for specific drugs by name
python scripts/fetch_fda_labels.py --search "betadine" --limit 10 --out data/raw_labels
python scripts/fetch_fda_labels.py --search "metformin" --limit 20 --out data/raw_labels
```

> This creates one `.txt` file per drug in `data/raw_labels/`, formatted with Markdown-style section headers (e.g. `## Dosage and Administration`) so Docling can parse them section-by-section.

### 4. Index the documents

```bash
# Parse, chunk, embed, and index into ChromaDB + BM25
python scripts/ingest_pipeline.py --input-dir data/raw_labels --chroma-dir chroma_db
```

> ⚠️ Re-run this command whenever you add new documents or change the chunking/tokenizer logic.

### 4. Run the app

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — the backend serves the frontend directly. One process, one port, no separate static file server.

---

## Key Design Decisions

### Why per-chunk relevance grading?

The naive approach (grade all retrieved chunks together as one blob) fails when hybrid retrieval returns one relevant chunk mixed with several irrelevant ones — the LLM judge sees more noise than signal and rejects everything. Grading each chunk independently keeps the relevant chunk regardless of how much noise surrounds it.

### Why temperature=0 on the judge nodes?

`grade_relevance_node` and `check_groundedness_node` make binary `true`/`false` decisions. Any temperature above 0 lets a small model (7B) flip its verdict on *identical input* across different calls — observed in practice. Setting `temperature=0.0` makes these classification decisions as deterministic as the underlying model allows. `generate_answer_node` and `rewrite_query_node` use `temperature=0.2`/`0.3` since those are genuinely creative text-generation tasks.

### Why hybrid retrieval (BM25 + dense)?

Dense (semantic) search is excellent at matching *meaning* but weak on exact terms: drug names, dosage codes (`500mg`), NDC codes. BM25 is excellent on exact terms but blind to paraphrase. Combining both via Reciprocal Rank Fusion (RRF) captures the strengths of each — validated empirically: "renal impairment dosage 500mg" retrieving the correct chunk via BM25 even when the semantic query returned unrelated drugs.

### Why Docling?

Docling parses document *structure* (section headings, tables, lists) rather than treating a file as a flat string. The `HybridChunker` respects section boundaries, so every chunk carries a `headings` metadata field (e.g. `Dosage and Administration`). This lets the retriever and the judge reason about *which section* an answer comes from, not just keyword overlap.

---

## Diagnostics

Three standalone scripts help isolate pipeline problems without running the full app:

```bash
# Check what the retriever actually returns for a query
python scripts/diagnostics/diagnose_retrieval.py "Betadine"

# Check what the relevance judge sees and how it decides
python scripts/diagnostics/diagnose_relevance_judge.py "what is the use of betadine"

# Check the generated answer and the groundedness verdict
python scripts/diagnostics/diagnose_groundedness.py "what is the use of betadine"
```

---

## API Reference

### `POST /chat`

```json
// Request
{ "question": "What is the maximum dose of Metformin for renal impairment?" }

// Response
{
  "answer": "The maximum recommended dose is 1000 mg per day for patients with eGFR 30-45 mL/min/1.73m².",
  "sources": [
    { "source_file": "Metformin_a1b2c3d4.txt", "section": "Dosage and Administration" }
  ],
  "relevance_retries": 0,
  "generation_retries": 0
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

**Ehab El-Shridy** — AI Developer specializing in Agentic AI and RAG systems.


