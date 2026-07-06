"""
retrieval.py

Hybrid retrieval (dense + sparse, fused with RRF) as a reusable module
for the LangGraph nodes. This is the same logic validated in
test_hybrid_retrieval.py, refactored into a class so retrieve_node()
in graph.py can call it with a single line.
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, TypedDict

import chromadb
from chromadb.config import Settings

from llm import embed_query

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Common English words that carry almost no distinguishing signal for
# BM25 in this domain. Without filtering these, a question like "what
# is the use of betadine" tokenizes to ["what", "is", "the", "use",
# "of", "betadine"] -- and since "what", "is", "the", "use", "of" each
# appear in hundreds of unrelated chunks across the corpus, BM25 can
# rank an irrelevant chunk that happens to share 4 stopwords above the
# one chunk that actually contains "betadine". Filtering stopwords
# means BM25 scores are driven almost entirely by the meaningful term
# ("betadine"), which is what keyword search is supposed to be good at.
BM25_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "may", "of", "on", "or", "should", "that", "the",
    "their", "there", "these", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
})


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    source_file: str
    headings: str


def tokenize_for_bm25(text: str) -> List[str]:
    import re
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in BM25_STOPWORDS]


class HybridRetriever:
    """Loads the ChromaDB collection + BM25 pickle built by
    ingest_pipeline.py and answers queries with RRF-fused results."""

    # Project root = parent of the backend/ folder where this file lives.
    # chroma_db/ sits at the project root, not inside backend/.
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_CHROMA = str(_PROJECT_ROOT / "chroma_db")
    _DEFAULT_BM25   = str(_PROJECT_ROOT / "chroma_db" / "bm25_index.pkl")

    def __init__(
        self,
        chroma_dir: str = _DEFAULT_CHROMA,
        bm25_path: str = _DEFAULT_BM25,
        collection_name: str = "drug_labels",
        top_k: int = 5,
    ):
        self.top_k = top_k

        client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = client.get_collection(collection_name)

        with open(bm25_path, "rb") as f:
            self.bm25_payload = pickle.load(f)

    def _dense_search(self, query: str):
        query_emb = embed_query(query, EMBEDDING_MODEL_NAME)
        results = self.collection.query(query_embeddings=[query_emb], n_results=self.top_k)
        return results["ids"][0], results["documents"][0], results["metadatas"][0]

    def _sparse_search(self, query: str):
        bm25 = self.bm25_payload["bm25"]
        chunk_ids = self.bm25_payload["chunk_ids"]
        texts = self.bm25_payload["texts"]
        metadatas = self.bm25_payload["metadatas"]

        scores = bm25.get_scores(tokenize_for_bm25(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        return (
            [chunk_ids[i] for i in ranked],
            [texts[i] for i in ranked],
            [metadatas[i] for i in ranked],
        )

    @staticmethod
    def _rrf(dense_ids: List[str], sparse_ids: List[str], k: int = 60) -> List[str]:
        scores = defaultdict(float)
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] += 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            scores[doc_id] += 1.0 / (k + rank + 1)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def retrieve(self, query: str, final_k: int = 5) -> List[RetrievedChunk]:
        dense_ids, dense_docs, dense_meta = self._dense_search(query)
        sparse_ids, sparse_docs, sparse_meta = self._sparse_search(query)

        id_to_doc = dict(zip(dense_ids, dense_docs))
        id_to_doc.update(dict(zip(sparse_ids, sparse_docs)))
        id_to_meta = dict(zip(dense_ids, dense_meta))
        id_to_meta.update(dict(zip(sparse_ids, sparse_meta)))

        fused_ids = self._rrf(dense_ids, sparse_ids)[:final_k]

        chunks: List[RetrievedChunk] = []
        for cid in fused_ids:
            meta = id_to_meta.get(cid, {})
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=id_to_doc.get(cid, ""),
                    source_file=meta.get("source_file", ""),
                    headings=meta.get("headings", ""),
                )
            )
        return chunks
