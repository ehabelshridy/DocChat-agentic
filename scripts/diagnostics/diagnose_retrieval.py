"""
diagnose_retrieval.py

Bypasses the LangGraph pipeline entirely and calls HybridRetriever
directly, so you can see EXACTLY what chunks come back for a query --
no LLM judge in the loop to obscure whether the problem is retrieval
or grading.

Usage (run from the project root):
    python scripts/diagnostics/diagnose_retrieval.py "Betadine"
    python scripts/diagnostics/diagnose_retrieval.py "Betadine antiseptic uses"
"""

import sys
from pathlib import Path

# Allow `from retrieval import ...` to resolve to backend/retrieval.py
# regardless of the current working directory this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from retrieval import HybridRetriever, tokenize_for_bm25

def main():
    if len(sys.argv) < 2:
        print('Usage: python diagnose_retrieval.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query!r}\n")

    retriever = HybridRetriever(top_k=5)

    print("=" * 60)
    print("STEP 1: Raw BM25 (keyword) search")
    print("=" * 60)
    sparse_ids, sparse_docs, sparse_meta = retriever._sparse_search(query)
    tokens = tokenize_for_bm25(query)
    print(f"Tokenized query: {tokens}")
    if not sparse_ids:
        print("  No BM25 results at all.")
    for i, (cid, doc, meta) in enumerate(zip(sparse_ids, sparse_docs, sparse_meta)):
        print(f"  {i+1}. [{cid}] source={meta.get('source_file')} | section={meta.get('headings')}")
        print(f"      {doc[:120]}...")

    print()
    print("=" * 60)
    print("STEP 2: Raw dense (semantic) search")
    print("=" * 60)
    try:
        dense_ids, dense_docs, dense_meta = retriever._dense_search(query)
        for i, (cid, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_meta)):
            print(f"  {i+1}. [{cid}] source={meta.get('source_file')} | section={meta.get('headings')}")
            print(f"      {doc[:120]}...")
    except Exception as e:
        print(f"  Dense search FAILED: {e}")

    print()
    print("=" * 60)
    print("STEP 3: Check if 'betadine' exists ANYWHERE in the BM25 corpus")
    print("=" * 60)
    payload = retriever.bm25_payload
    found_anywhere = False
    for i, text in enumerate(payload["texts"]):
        if "betadine" in text.lower():
            found_anywhere = True
            meta = payload["metadatas"][i]
            print(f"  FOUND in chunk: source={meta.get('source_file')} | section={meta.get('headings')}")
            print(f"    {text[:150]}...")
    if not found_anywhere:
        print("  'betadine' was NOT found in any indexed chunk's text.")
        print("  -> This means the word never made it into chroma_db/bm25_index.pkl.")
        print("  -> Check: was the Betadine document actually in --input-dir when you ran ingest_pipeline.py?")
        print("  -> Check: did Docling/the chunker silently skip or fail on that file? Re-run")
        print("     ingest_pipeline.py and watch for '[error]' lines in its output.")

    print()
    print("=" * 60)
    print("STEP 4: Total corpus size (sanity check)")
    print("=" * 60)
    print(f"  Total chunks indexed: {len(payload['texts'])}")
    print(f"  Unique source files: {len(set(m.get('source_file') for m in payload['metadatas']))}")


if __name__ == "__main__":
    main()
