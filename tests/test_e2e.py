"""
tests/test_e2e.py

End-to-end tests — run the full pipeline against the real ChromaDB
index and real HuggingFace Inference API. These tests:
  - Require HF_TOKEN in .env
  - Require chroma_db/ to be populated (run ingest_pipeline.py first)
  - Cost a small number of HF API credits per run
  - Are skipped automatically if the index or token is missing

Run with:
    cd D:\\project\\DocChat_agentic
    pytest tests/test_e2e.py -v

Skip in CI (to avoid API costs):
    pytest tests/ -v --ignore=tests/test_e2e.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# ── Skip conditions ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"
BM25_INDEX   = CHROMA_DIR / "bm25_index.pkl"

def _has_index():
    return CHROMA_DIR.exists() and BM25_INDEX.exists()

def _has_token():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(os.getenv("HF_TOKEN"))

needs_index = pytest.mark.skipif(
    not _has_index(),
    reason="chroma_db/ not found — run scripts/ingest_pipeline.py first",
)
needs_token = pytest.mark.skipif(
    not _has_token(),
    reason="HF_TOKEN not set in .env",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def query(question):
    from graph import run_query
    return run_query(question)


# ── Tests ─────────────────────────────────────────────────────────────────────

@needs_index
@needs_token
class TestKnownDrugs:
    """
    These drugs are known to be in the index (from the openFDA fetch).
    Each test asserts that the answer contains at least one expected
    clinical term from the actual label content — not just that it
    returned something.
    """

    def test_betadine_use(self):
        result = query("what is the use of betadine")
        assert result["answer"], "Expected a non-empty answer"
        answer_lower = result["answer"].lower()
        expected_terms = ["first aid", "infection", "antiseptic", "povidone", "cuts", "scrapes", "burns"]
        matched = [t for t in expected_terms if t in answer_lower]
        assert matched, (
            f"Answer did not contain any expected term from {expected_terms}.\n"
            f"Got: {result['answer']}"
        )

    def test_metformin_dosage(self):
        result = query("what is the maximum dose of metformin for renal impairment")
        assert result["answer"], "Expected a non-empty answer"
        answer_lower = result["answer"].lower()
        expected_terms = ["1000", "renal", "egfr", "impairment", "mg"]
        matched = [t for t in expected_terms if t in answer_lower]
        assert matched, (
            f"Answer did not contain any expected term from {expected_terms}.\n"
            f"Got: {result['answer']}"
        )

    def test_out_of_scope_question_returns_fallback(self):
        """A question about a non-drug topic should trigger the fallback."""
        result = query("what is the capital of France")
        answer_lower = result["answer"].lower()
        fallback_signals = ["couldn't find", "outside the scope", "couldn't generate"]
        matched = [s for s in fallback_signals if s in answer_lower]
        assert matched, (
            f"Expected a fallback message for an out-of-scope question.\n"
            f"Got: {result['answer']}"
        )


@needs_index
@needs_token
class TestResponseStructure:
    """Validates the response schema regardless of content."""

    def test_response_has_required_keys(self):
        result = query("what is betadine used for")
        assert "answer"  in result
        assert "sources" in result
        assert "retries" in result
        assert "relevance"  in result["retries"]
        assert "generation" in result["retries"]

    def test_retries_are_non_negative_integers(self):
        result = query("what is betadine used for")
        assert isinstance(result["retries"]["relevance"],  int)
        assert isinstance(result["retries"]["generation"], int)
        assert result["retries"]["relevance"]  >= 0
        assert result["retries"]["generation"] >= 0

    def test_grounded_answer_has_sources(self):
        result = query("what is betadine used for")
        if "couldn't" not in result["answer"].lower():
            assert len(result["sources"]) > 0, (
                "A grounded answer must have at least one source citation"
            )
