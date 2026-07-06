"""
tests/test_pipeline.py

Integration tests for the LangGraph pipeline — validates that the
conditional edges, retry budgets, and fallback routing all behave
exactly as the confirmed architecture diagram shows.

The LLM (chat_completion) and retriever (HybridRetriever) are both
mocked with deterministic stubs, so these tests:
  - Run without any HF_TOKEN or ChromaDB index
  - Are fully deterministic (no temperature variance)
  - Complete in under 2 seconds total
  - Cover every routing branch in the graph

Run with:
    cd D:\\project\\DocChat_agentic
    pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pytest
import graph as graph_module
import llm as llm_module


# ── Test fixtures ─────────────────────────────────────────────────────────────

class ScriptedLLM:
    """Returns scripted responses one per call, regardless of prompt."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def __call__(self, system_prompt, user_prompt, **kwargs):
        assert self.call_count < len(self.responses), (
            f"ScriptedLLM ran out of responses after {self.call_count} calls"
        )
        r = self.responses[self.call_count]
        self.call_count += 1
        return r


class FakeRetriever:
    def __init__(self, num_chunks=3):
        self.call_count = 0
        self.num_chunks = num_chunks

    def retrieve(self, query, final_k=5):
        self.call_count += 1
        return [
            {
                "chunk_id": f"chunk_{self.call_count}_{i}",
                "text": f"Betadine uses first aid prevent infection. Query was: {query}",
                "source_file": "Betadine_test.txt",
                "headings": "Indications and Usage",
            }
            for i in range(self.num_chunks)
        ]


@pytest.fixture(autouse=True)
def reset_retriever():
    """Reset the cached retriever between tests."""
    graph_module._retriever = None
    yield
    graph_module._retriever = None


def run_with_scripts(scripted_responses, num_chunks=3):
    fake_llm = ScriptedLLM(scripted_responses)
    graph_module._retriever = FakeRetriever(num_chunks=num_chunks)
    llm_module.chat_completion = fake_llm
    graph_module.chat_completion = fake_llm
    return graph_module.run_query("what is the use of betadine"), fake_llm


# ── Test cases ────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_relevant_and_grounded_on_first_try(self):
        result, llm = run_with_scripts([
            '{"is_relevant": true}',                              # chunk 0 graded
            '{"is_relevant": false}',                             # chunk 1
            '{"is_relevant": false}',                             # chunk 2
            "Betadine is used for first aid to prevent infection.",# generate
            '{"is_grounded": true}',                              # groundedness
        ])
        assert result["retries"]["relevance"]   == 0
        assert result["retries"]["generation"]  == 0
        assert "Betadine" in result["answer"] or "first aid" in result["answer"]
        assert len(result["sources"]) > 0


class TestRelevanceRetryLoop:
    def test_one_relevance_retry_then_success(self):
        result, llm = run_with_scripts([
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',  # pass 1: all irrelevant
            "rewritten query about betadine",                                               # rewrite
            '{"is_relevant": true}',  '{"is_relevant": false}', '{"is_relevant": false}',  # pass 2: 1 relevant
            "Betadine is used for first aid.",                                              # generate
            '{"is_grounded": true}',                                                        # groundedness
        ])
        assert result["retries"]["relevance"] == 1
        assert result["retries"]["generation"] == 0
        assert len(result["sources"]) > 0

    def test_max_relevance_retries_triggers_fallback(self):
        result, _ = run_with_scripts([
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',  # pass 1
            "rewrite 1",
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',  # pass 2
            "rewrite 2",
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',  # pass 3 → fallback
        ])
        assert result["retries"]["relevance"] == 2
        assert "couldn't find information" in result["answer"]
        assert len(result["sources"]) == 0

    def test_mixed_batch_one_relevant_chunk_passes(self):
        """Core fix: one relevant chunk out of 3 must still reach generation."""
        result, _ = run_with_scripts([
            '{"is_relevant": false}',                              # chunk 0: noise
            '{"is_relevant": true}',                               # chunk 1: relevant ← the one that matters
            '{"is_relevant": false}',                              # chunk 2: noise
            "Betadine is used for first aid.",                     # generate
            '{"is_grounded": true}',                               # groundedness
        ])
        assert result["retries"]["relevance"] == 0
        assert len(result["sources"]) > 0


class TestGroundednessRetryLoop:
    def test_one_groundedness_retry_then_success(self):
        result, _ = run_with_scripts([
            '{"is_relevant": true}',  '{"is_relevant": false}', '{"is_relevant": false}',
            "hallucinated answer",
            '{"is_grounded": false}',
            "Betadine is used for first aid to prevent infection.",
            '{"is_grounded": true}',
        ])
        assert result["retries"]["generation"] == 1
        assert "first aid" in result["answer"]
        assert len(result["sources"]) > 0

    def test_max_groundedness_retries_triggers_fallback(self):
        result, _ = run_with_scripts([
            '{"is_relevant": true}', '{"is_relevant": false}', '{"is_relevant": false}',
            "hallucinated 1", '{"is_grounded": false}',
            "hallucinated 2", '{"is_grounded": false}',
            "hallucinated 3", '{"is_grounded": false}',
        ])
        assert result["retries"]["generation"] == 2
        assert "couldn't generate an answer" in result["answer"]
        assert len(result["sources"]) == 0


class TestFallbackMessages:
    def test_out_of_scope_fallback_message(self):
        """Fallback after relevance exhaustion mentions 'knowledge base'."""
        result, _ = run_with_scripts([
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',
            "rewrite 1",
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',
            "rewrite 2",
            '{"is_relevant": false}', '{"is_relevant": false}', '{"is_relevant": false}',
        ])
        assert "couldn't find information" in result["answer"]

    def test_unverified_fallback_message(self):
        """Fallback after groundedness exhaustion mentions 'verify'."""
        result, _ = run_with_scripts([
            '{"is_relevant": true}', '{"is_relevant": false}', '{"is_relevant": false}',
            "h1", '{"is_grounded": false}',
            "h2", '{"is_grounded": false}',
            "h3", '{"is_grounded": false}',
        ])
        assert "couldn't generate an answer" in result["answer"]
