"""
ingest_pipeline.py

Data engineering pipeline for the DocChat project:

    raw documents  -->  Docling parsing  -->  chunking  -->  embeddings
                                                                  |
                                                                  v
                                          ChromaDB (dense/vector index)
                                                  +
                                          BM25 index (sparse/keyword index)

The two indexes share the same chunk IDs, so a downstream retriever can
run both searches and merge the results with Reciprocal Rank Fusion
(RRF) to get hybrid retrieval. This script only builds the indexes.

Why Docling here (and not a plain text loader):
- Docling parses structure (headers, tables, lists) instead of treating
  a document as one flat blob of text. That structure is preserved in
  the chunk metadata (e.g. which section a chunk came from), which is
  exactly what lets a RAG system answer "what's the max dose for renal
  impairment patients?" by retrieving the Dosage section specifically,
  not a random 500-character window.
- Docling also reads PDFs, scanned PDFs (via OCR), Word docs, HTML, and
  PowerPoint with the same API.

Embeddings:
- Generated via OpenAI's Embeddings API (text-embedding-3-small).
- Chunking uses a tiktoken-based tokenizer (cl100k_base) so chunk sizes
  are calibrated to the same encoding used by gpt-4o-mini and
  text-embedding-3-small -- no HuggingFace dependency anywhere.

Usage:
    pip install -r requirements.txt

    # .env file in the project root:
    #   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    python scripts/ingest_pipeline.py \
        --input-dir data/raw_labels \
        --chroma-dir chroma_db
"""

import argparse
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from openai import OpenAI

# Formats Docling can natively read. Anything else is skipped with a warning.
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".pptx"}

EMBEDDING_MODEL_NAME  = "text-embedding-3-small"   # OpenAI model used for embeddings
EMBEDDING_DIM         = 1536   # text-embedding-3-small output dimension
EMBEDDING_BATCH_SIZE  = 100    # OpenAI allows up to 2048 inputs per request

# Maximum tokens per chunk fed to the embedding model.
# text-embedding-3-small supports up to 8191 tokens; we use 512 to keep
# chunks focused and retrieval precise.
CHUNK_MAX_TOKENS = 512


@dataclass
class Chunk:
    """One chunk of a document, ready to be embedded and indexed."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    headings: List[str] = field(default_factory=list)


# Common English words that carry almost no distinguishing signal for
# BM25 in this domain. Filtering these out at INDEX time (not just
# query time) keeps the BM25 vocabulary consistent with retrieval.py's
# tokenizer -- both must filter identically, or BM25 scores would be
# computed against a different term distribution than queries assume.
# Note "use"/"uses"/"used"/"using" are deliberately KEPT (not treated
# as stopwords) because "Uses" is a real FDA label section heading
# (e.g. on OTC products like Betadine) and a meaningful query term.
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer as DoclingBaseTokenizer


class TiktokenTokenizer(DoclingBaseTokenizer):
    """Docling-compatible tokenizer backed by tiktoken (OpenAI's tokenizer).

    Docling's HybridChunker requires a tokenizer that implements
    count_tokens() and get_max_tokens() to decide where to split chunks.
    This class wraps tiktoken so we can use it without any HuggingFace
    dependency.

    We use the 'cl100k_base' encoding which is the same encoding used by
    gpt-4o-mini and text-embedding-3-small, so chunk sizes are calibrated
    to the actual models we use.
    """
    max_tokens: int = CHUNK_MAX_TOKENS

    def model_post_init(self, __context) -> None:
        import tiktoken
        object.__setattr__(self, "_enc", tiktoken.get_encoding("cl100k_base"))

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return self._enc


BM25_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "may", "of", "on", "or", "should", "that", "the",
    "their", "there", "these", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
})


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple lowercase word tokenizer for BM25. Good enough for keyword
    matching on drug names, dosages (e.g. '500mg'), and codes."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in BM25_STOPWORDS]


def discover_files(input_dir: Path) -> List[Path]:
    files = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    skipped = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() not in SUPPORTED_SUFFIXES
    ]
    for s in skipped:
        print(f"  [skip] unsupported file type: {s.name}")
    return files


def convert_and_chunk(filepath: Path, converter: DocumentConverter, chunker: HybridChunker) -> List[Chunk]:
    """Run Docling conversion, then Docling's HybridChunker.

    HybridChunker is structure-aware: it respects section boundaries
    (headings) and table boundaries first, then applies token-aware
    splitting/merging so chunks fit the embedding model's context
    window without cutting a sentence or a table row in half.

    Header-merge fix: the very first chunk of each document (brand
    name / generic name / manufacturer, with no section heading) has
    no clinical content on its own. If that chunk happens to rank #1
    in retrieval for a broad query like "what about Betadine?", the
    relevance judge correctly rejects it -- there's nothing to answer
    with -- and the whole query falls back to "out of scope" even
    though the document clearly covers the drug. To prevent this, any
    heading-less chunk gets merged into the chunk that follows it, so
    every indexed chunk carries both the drug identity AND real
    section content together.
    """
    result = converter.convert(str(filepath))
    doc = result.document

    raw_chunks = []
    for chunk in chunker.chunk(doc):
        raw_headings = list(getattr(chunk.meta, "headings", None) or [])
        headings = [h for h in raw_headings if h and h.strip()]
        chunk_text = chunk.text.strip()
        if not chunk_text:
            continue
        raw_chunks.append((chunk_text, headings))

    merged: List[tuple] = []
    pending_header = None
    for chunk_text, headings in raw_chunks:
        if not headings and pending_header is None:
            # This is a header-only chunk (e.g. brand/generic/manufacturer
            # block) -- hold it and merge into the next chunk instead of
            # indexing it alone.
            pending_header = chunk_text
            continue
        if pending_header is not None:
            chunk_text = f"{pending_header}\n\n{chunk_text}"
            pending_header = None
        merged.append((chunk_text, headings))

    if pending_header is not None:
        # Edge case: the document was ONLY a header with no sections at
        # all. Keep it rather than silently dropping the document.
        merged.append((pending_header, []))

    chunks = []
    for i, (chunk_text, headings) in enumerate(merged):
        chunks.append(
            Chunk(
                chunk_id=f"{filepath.stem}_{i}",
                text=chunk_text,
                source_file=filepath.name,
                chunk_index=i,
                headings=headings,
            )
        )
    return chunks


def get_openai_client() -> OpenAI:
    """Load OPENAI_API_KEY from .env and return a ready OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY not found. Create a .env file with:\n"
            "  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "Get a key at https://platform.openai.com/api-keys"
        )
        sys.exit(1)
    return OpenAI(api_key=api_key)


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model_name: str,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_retries: int = 3,
) -> List[List[float]]:
    """Embed a list of texts via OpenAI Embeddings API, in batches.

    OpenAI allows up to 2048 inputs per request, so batch_size=100
    keeps us well within limits while minimising round trips.
    """
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start : start + batch_size]
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks) ...")

        for attempt in range(1, max_retries + 1):
            try:
                response = client.embeddings.create(model=model_name, input=batch)
                # response.data is ordered to match the input list
                all_embeddings.extend([item.embedding for item in response.data])
                break
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Embedding batch {batch_num} failed after {max_retries} attempts: {e}"
                    ) from e
                wait = 2 ** attempt
                print(f"    [retry {attempt}/{max_retries}] {e} -- waiting {wait}s")
                time.sleep(wait)

    return all_embeddings


def build_indexes(
    input_dir: str,
    chroma_dir: str,
    bm25_path: str,
    collection_name: str,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"Discovering files in {input_dir} ...")
    files = discover_files(input_path)
    if not files:
        print("No supported files found. Nothing to ingest.")
        sys.exit(1)
    print(f"Found {len(files)} file(s) to process.\n")

    print("Loading Docling converter + HybridChunker ...")
    converter = DocumentConverter()
    # TiktokenTokenizer wraps OpenAI's tiktoken library so HybridChunker
    # can count tokens and split chunks without any HuggingFace dependency.
    chunker = HybridChunker(tokenizer=TiktokenTokenizer(), max_tokens=CHUNK_MAX_TOKENS)

    all_chunks: List[Chunk] = []
    for filepath in files:
        print(f"  Parsing + chunking: {filepath.name}")
        try:
            file_chunks = convert_and_chunk(filepath, converter, chunker)
        except Exception as e:
            print(f"    [error] failed to process {filepath.name}: {e}")
            continue
        print(f"    -> {len(file_chunks)} chunk(s)")
        all_chunks.extend(file_chunks)

    if not all_chunks:
        print("\nNo chunks were produced. Aborting before writing indexes.")
        sys.exit(1)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    # --- Embeddings (via HF Inference API) + ChromaDB (dense / semantic index) ---
    print(f"\nConnecting to OpenAI API for model: {embedding_model_name} ...")
    openai_client = get_openai_client()

    print("Computing embeddings via OpenAI Embeddings API ...")
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(openai_client, texts, embedding_model_name)

    print(f"\nWriting to ChromaDB at {chroma_dir} (collection: {collection_name}) ...")
    client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))

    # Recreate the collection fresh each run so re-ingesting doesn't duplicate chunks.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"embedding_model": embedding_model_name},
    )

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "headings": " > ".join(c.headings) if c.headings else "",
            }
            for c in all_chunks
        ],
    )
    print(f"  Stored {collection.count()} vectors in ChromaDB.")

    # --- BM25 (sparse / keyword index) ---
    print(f"\nBuilding BM25 index ...")
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_payload = {
        "bm25": bm25,
        # Keep chunk_id -> text/metadata alongside the BM25 model so a
        # retriever can map BM25 scores back to the same chunk records
        # used by ChromaDB (same IDs on both sides = easy RRF fusion).
        "chunk_ids": [c.chunk_id for c in all_chunks],
        "texts": texts,
        "metadatas": [
            {"source_file": c.source_file, "headings": " > ".join(c.headings)}
            for c in all_chunks
        ],
    }
    os.makedirs(os.path.dirname(bm25_path) or ".", exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_payload, f)
    print(f"  Stored BM25 index at {bm25_path}")

    print("\nDone. Both indexes are aligned on chunk_id and ready for hybrid retrieval.")
    print(f"  - Dense index:  ChromaDB collection '{collection_name}' at {chroma_dir}")
    print(f"  - Sparse index: BM25 pickle at {bm25_path}")


def main():
    parser = argparse.ArgumentParser(description="Docling -> Chunk -> Embed -> ChromaDB + BM25 ingestion pipeline")
    parser.add_argument("--input-dir", default="./data/raw_labels", help="Directory of source documents")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB persistent storage directory")
    parser.add_argument("--bm25-path", default="./chroma_db/bm25_index.pkl", help="Path to save the BM25 pickle")
    parser.add_argument("--collection", default="drug_labels", help="ChromaDB collection name")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME, help="OpenAI embedding model name")
    args = parser.parse_args()

    start = time.time()
    build_indexes(
        input_dir=args.input_dir,
        chroma_dir=args.chroma_dir,
        bm25_path=args.bm25_path,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
    )
    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
