"""
diagnose_groundedness.py

Runs the full pipeline up through check_groundedness, printing the
raw output at every stage: the relevant chunks that survived
filtering, the generated answer, and the raw groundedness judge
response (before parsing). This isolates whether the problem is:
  (a) the generated answer actually drifting from the source text, or
  (b) the groundedness judge being overly strict / misreading a
      genuinely faithful answer

Usage (run from the project root, so ./chroma_db resolves correctly):
    python scripts/diagnostics/diagnose_groundedness.py "what is the use of betadine"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from retrieval import HybridRetriever
from llm import chat_completion, parse_json_response
from graph import (
    RELEVANCE_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GROUNDEDNESS_SYSTEM_PROMPT,
    TOP_K,
)


def main():
    if len(sys.argv) < 2:
        print('Usage: python diagnose_groundedness.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question!r}\n")

    retriever = HybridRetriever(top_k=TOP_K)
    chunks = retriever.retrieve(question, final_k=TOP_K)

    print("=" * 60)
    print("STEP 1: Per-chunk relevance grading")
    print("=" * 60)
    relevant_chunks = []
    for c in chunks:
        user_prompt = f"Question: {question}\n\nExcerpt:\n{c['text']}"
        raw = chat_completion(RELEVANCE_SYSTEM_PROMPT, user_prompt, max_tokens=100, temperature=0.0)
        parsed = parse_json_response(raw)
        is_relevant = bool(parsed.get("is_relevant", parsed.get("decision", False)))
        status = "RELEVANT" if is_relevant else "skip"
        print(f"  [{status}] {c['chunk_id']} | {c['source_file']} | {c['headings']}")
        if is_relevant:
            relevant_chunks.append(c)
            print(f"      text: {c['text'][:300]}")

    if not relevant_chunks:
        print("\nNo relevant chunks survived filtering -- groundedness check never runs in the real graph.")
        sys.exit(0)

    print()
    print("=" * 60)
    print("STEP 2: Generate answer (using ONLY the relevant chunks above)")
    print("=" * 60)
    context = "\n\n---\n\n".join(
        f"[Source: {c['source_file']} | Section: {c['headings']}]\n{c['text']}"
        for c in relevant_chunks
    )
    gen_prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}"
    answer = chat_completion(GENERATE_SYSTEM_PROMPT, gen_prompt, max_tokens=512, temperature=0.2)
    print(f"Generated answer:\n{answer}\n")

    print("=" * 60)
    print("STEP 3: Groundedness check -- raw LLM response (before parsing)")
    print("=" * 60)
    ground_context = "\n\n---\n\n".join(c["text"] for c in relevant_chunks)
    ground_prompt = f"Document excerpts:\n{ground_context}\n\nGenerated answer:\n{answer}"
    raw_ground = chat_completion(GROUNDEDNESS_SYSTEM_PROMPT, ground_prompt, max_tokens=150, temperature=0.0)
    print(repr(raw_ground))
    print()
    print("(human-readable):")
    print(raw_ground)

    parsed_ground = parse_json_response(raw_ground)
    print()
    print(f"Parsed: {parsed_ground}")
    is_grounded = bool(parsed_ground.get("is_grounded", parsed_ground.get("decision", False)))
    print(f"Final is_grounded the graph would use: {is_grounded}")


if __name__ == "__main__":
    main()
