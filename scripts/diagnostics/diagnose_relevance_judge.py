"""
diagnose_relevance_judge.py

Runs the EXACT retrieval + relevance-grading step the graph uses, but
prints the raw LLM response before any JSON parsing happens. This
tells us definitively whether:
  (a) the judge is looking at good context and still saying "no"
      (a judge/prompt problem), or
  (b) the judge's raw response isn't valid JSON and parse_json_response's
      fallback is guessing wrong (a parsing problem)

Usage (run from the project root, so ./chroma_db resolves correctly):
    python scripts/diagnostics/diagnose_relevance_judge.py "what is the use of betadine"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from retrieval import HybridRetriever
from llm import chat_completion, parse_json_response
from graph import RELEVANCE_SYSTEM_PROMPT, TOP_K


def main():
    if len(sys.argv) < 2:
        print('Usage: python diagnose_relevance_judge.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question!r}\n")

    retriever = HybridRetriever(top_k=TOP_K)
    chunks = retriever.retrieve(question, final_k=TOP_K)

    print("=" * 60)
    print(f"STEP 1: Retrieved {len(chunks)} chunks (this is exactly what grade_relevance sees)")
    print("=" * 60)
    for i, c in enumerate(chunks):
        print(f"  {i+1}. [{c['chunk_id']}] source={c['source_file']} | section={c['headings']}")
        print(f"      {c['text'][:200]}...")
    print()

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    user_prompt = f"Question: {question}\n\nRetrieved excerpts:\n{context}"

    print("=" * 60)
    print("STEP 2: Raw LLM response (before JSON parsing)")
    print("=" * 60)
    raw = chat_completion(RELEVANCE_SYSTEM_PROMPT, user_prompt, max_tokens=150, temperature=0.0)
    print(repr(raw))
    print()
    print("(human-readable):")
    print(raw)
    print()

    print("=" * 60)
    print("STEP 3: Parsed result")
    print("=" * 60)
    parsed = parse_json_response(raw)
    print(f"  Parsed dict: {parsed}")
    is_relevant = bool(parsed.get("is_relevant", parsed.get("decision", False)))
    print(f"  Final is_relevant value the graph would use: {is_relevant}")

    if "is_relevant" not in parsed:
        print()
        print("  ⚠️  'is_relevant' key was NOT found in the parsed JSON.")
        print("      This means parse_json_response fell back to its keyword")
        print("      heuristic ('true'/'yes' in the raw text), which is much")
        print("      less reliable than a real is_relevant field.")


if __name__ == "__main__":
    main()
