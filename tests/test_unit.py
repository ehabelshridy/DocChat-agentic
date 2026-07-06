"""
tests/test_unit.py

Unit tests for pure-logic functions — no LLM calls, no ChromaDB,
no network access. These run in milliseconds and validate the core
algorithmic building blocks independently of any external service.

Run with:
    cd D:\\project\\DocChat_agentic
    pytest tests/test_unit.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from retrieval import tokenize_for_bm25, HybridRetriever
from llm import parse_json_response


# ── tokenize_for_bm25 ────────────────────────────────────────────────────────

class TestTokenizer:
    def test_removes_stopwords(self):
        tokens = tokenize_for_bm25("what is the use of betadine")
        assert "what" not in tokens
        assert "is"   not in tokens
        assert "the"  not in tokens
        assert "of"   not in tokens

    def test_keeps_drug_name(self):
        tokens = tokenize_for_bm25("what is the use of betadine")
        assert "betadine" in tokens

    def test_keeps_use_and_uses(self):
        # 'use' and 'uses' are kept because they map to an FDA label
        # section heading — excluding them would hurt retrieval
        assert "use"  in tokenize_for_bm25("use of the drug")
        assert "uses" in tokenize_for_bm25("uses and indications")

    def test_keeps_dosage_numbers(self):
        tokens = tokenize_for_bm25("max dose is 500mg daily")
        assert "500mg" in tokens
        assert "dose"  in tokens

    def test_lowercases(self):
        tokens = tokenize_for_bm25("Betadine POVIDONE-IODINE")
        assert "betadine" in tokens
        assert "povidone" in tokens

    def test_empty_string(self):
        assert tokenize_for_bm25("") == []

    def test_only_stopwords(self):
        assert tokenize_for_bm25("what is the a an") == []


# ── RRF fusion ───────────────────────────────────────────────────────────────

class TestRRF:
    def _rrf(self, dense, sparse, k=60):
        return HybridRetriever._rrf(dense, sparse, k=k)

    def test_appears_in_both_ranks_highest(self):
        dense  = ["a", "b", "c"]
        sparse = ["c", "a", "d"]
        fused  = self._rrf(dense, sparse)
        # 'a' is #1 dense + #2 sparse, 'c' is #3 dense + #1 sparse
        # Both should outrank 'b' (dense only) and 'd' (sparse only)
        assert fused.index("a") < fused.index("b")
        assert fused.index("c") < fused.index("b")
        assert fused.index("a") < fused.index("d")

    def test_result_contains_all_unique_ids(self):
        fused = self._rrf(["a", "b"], ["c", "d"])
        assert set(fused) == {"a", "b", "c", "d"}

    def test_empty_lists(self):
        assert self._rrf([], []) == []

    def test_one_empty_list(self):
        fused = self._rrf(["a", "b", "c"], [])
        assert fused == ["a", "b", "c"]

    def test_identical_lists_preserves_order(self):
        ids   = ["x", "y", "z"]
        fused = self._rrf(ids, ids)
        assert fused == ids


# ── parse_json_response ───────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_clean_json(self):
        raw    = '{"is_relevant": true, "reasoning": "matches"}'
        parsed = parse_json_response(raw)
        assert parsed["is_relevant"] is True

    def test_json_with_markdown_fence(self):
        raw    = '```json\n{"is_grounded": false}\n```'
        parsed = parse_json_response(raw)
        assert parsed["is_grounded"] is False

    def test_json_embedded_in_text(self):
        raw    = 'Here is my answer: {"is_relevant": true} done.'
        parsed = parse_json_response(raw)
        assert parsed.get("is_relevant") is True

    def test_fallback_keyword_true(self):
        raw    = "I think this is true and relevant."
        parsed = parse_json_response(raw)
        assert parsed.get("decision") is True

    def test_fallback_keyword_false(self):
        raw    = "No, this is unrelated."
        parsed = parse_json_response(raw)
        assert parsed.get("decision") is False

    def test_grounded_false_parsed(self):
        raw    = '{"is_grounded": false, "reasoning": "invented detail"}'
        parsed = parse_json_response(raw)
        assert parsed["is_grounded"] is False
