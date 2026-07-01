"""
fetch_fda_labels.py

Downloads drug labeling records from the openFDA Drug Label API and
writes each one as a structured plain-text document (one file per
drug), formatted to resemble a real prescribing-information leaflet.

The output files are intentionally kept as structured plain text with
Markdown-style section headers (## Dosage and Administration, etc.)
so that Docling's HybridChunker can parse them section-by-section
rather than treating each file as one undifferentiated blob -- which
is exactly what makes the downstream RAG retrieval accurate.

Usage examples
--------------
Fetch the first 100 labels (default):
    python scripts/fetch_fda_labels.py

Fetch 500 labels in batches of 20:
    python scripts/fetch_fda_labels.py --limit 500 --batch-size 20

Search for a specific drug by name:
    python scripts/fetch_fda_labels.py --search "betadine" --limit 10

Custom output directory:
    python scripts/fetch_fda_labels.py --limit 100 --out data/raw_labels

Requirements
------------
    pip install requests tqdm
    (both are listed in requirements.txt)
"""

import argparse
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.fda.gov/drug/label.json"

# Maximum records the free openFDA tier returns per request without
# an API key. Raise to 100 if you register for a free key.
FREE_TIER_MAX_BATCH = 20

# Sections extracted from each openFDA record.
# Ordered from most to least clinically critical; sections not present
# in a given record are silently skipped.
SECTIONS = [
    ("Indications and Usage",        "indications_and_usage"),
    ("Dosage and Administration",    "dosage_and_administration"),
    ("Contraindications",            "contraindications"),
    ("Boxed Warning",                "boxed_warning"),
    ("Warnings",                     "warnings"),
    ("Warnings and Cautions",        "warnings_and_cautions"),
    ("Adverse Reactions",            "adverse_reactions"),
    ("Drug Interactions",            "drug_interactions"),
    ("Use in Specific Populations",  "use_in_specific_populations"),
    ("Overdosage",                   "overdosage"),
    ("Clinical Pharmacology",        "clinical_pharmacology"),
    ("How Supplied",                 "how_supplied"),
    ("Storage and Handling",         "storage_and_handling"),
    ("Patient Counseling Information","patient_counseling_information"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_filename(name: str, max_len: int = 80) -> str:
    """Convert a drug name into a filesystem-safe filename segment."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_len] if cleaned else "unknown_drug"


def record_to_document(record: dict) -> str:
    """
    Render a single openFDA label record as structured plain text.

    The header block (brand / generic / manufacturer) is written first
    without a Markdown heading so it is always present. Then each
    clinical section is written under a '## Section Name' heading so
    that Docling's HybridChunker can split the file along section
    boundaries rather than at arbitrary token counts.
    """
    lines: list[str] = []

    openfda = record.get("openfda", {})
    brand        = openfda.get("brand_name",        ["Unknown Brand"])
    generic      = openfda.get("generic_name",       ["Unknown Generic"])
    manufacturer = openfda.get("manufacturer_name",  ["Unknown Manufacturer"])

    lines += [
        "PRESCRIBING INFORMATION",
        f"Brand Name: {', '.join(brand)}",
        f"Generic Name: {', '.join(generic)}",
        f"Manufacturer: {', '.join(manufacturer)}",
        "=" * 70,
        "",
    ]

    for title, key in SECTIONS:
        value = record.get(key)
        if not value:
            continue
        text = " ".join(value) if isinstance(value, list) else str(value)
        lines += [f"## {title}", text.strip(), ""]

    return "\n".join(lines)


# ── Network ──────────────────────────────────────────────────────────────────

def fetch_batch(
    limit: int,
    skip: int,
    search: str | None = None,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> list[dict]:
    """
    Fetch one batch of records from the openFDA API.

    Retries on transient network/server errors with exponential
    backoff. Raises on persistent failures so the caller can decide
    whether to abort or continue.
    """
    params: dict = {"limit": limit, "skip": skip}
    if search:
        # openFDA full-text search across all label fields
        params["search"] = search

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 404:
                # openFDA returns 404 when `skip` goes past the total
                # result count -- treat as "no more records"
                return []
            resp.raise_for_status()
            return resp.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff ** attempt
                tqdm.write(f"  [retry {attempt}/{max_retries}] {exc} — waiting {wait:.0f}s")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch batch (skip={skip}) after {max_retries} attempts: {last_exc}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_OUT  = str(_PROJECT_ROOT / "data" / "raw_labels")

    parser = argparse.ArgumentParser(
        description="Download FDA drug labels and save them as structured text files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Total number of labels to download (default: 100)",
    )
    parser.add_argument(
        "--out", type=str, default=_DEFAULT_OUT,
        help="Output directory for the .txt files (default: <project_root>/data/raw_labels)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=FREE_TIER_MAX_BATCH,
        help=f"Records per API call (default: {FREE_TIER_MAX_BATCH}; max 100 with a free API key)",
    )
    parser.add_argument(
        "--search", type=str, default=None,
        help='Optional openFDA search query, e.g. --search "betadine" or --search "openfda.brand_name:metformin"',
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"openFDA Drug Label Downloader")
    print(f"  Target : {args.limit} labels")
    print(f"  Search : {args.search or '(no filter — general sample)'}")
    print(f"  Output : {out_dir.resolve()}")
    print()

    fetched = 0
    written = 0
    skip    = 0
    errors  = 0

    with tqdm(total=args.limit, unit="label", desc="Downloading") as pbar:
        while fetched < args.limit:
            batch_size = min(args.batch_size, args.limit - fetched)

            try:
                results = fetch_batch(
                    limit=batch_size,
                    skip=skip,
                    search=args.search,
                )
            except RuntimeError as exc:
                tqdm.write(f"  [error] {exc}")
                errors += 1
                break

            if not results:
                tqdm.write("  No more results from the API.")
                break

            for record in results:
                openfda  = record.get("openfda", {})
                brand    = openfda.get("brand_name", ["unknown"])[0]
                set_id   = record.get("set_id", str(written))
                fname    = f"{safe_filename(brand)}_{set_id[:8]}.txt"
                fpath    = out_dir / fname

                try:
                    doc_text = record_to_document(record)
                    fpath.write_text(doc_text, encoding="utf-8")
                    written += 1
                except Exception as exc:
                    tqdm.write(f"  [skip] failed to write {fname}: {exc}")
                    errors += 1

            fetched += len(results)
            skip    += len(results)
            pbar.update(len(results))

            # Be polite to the free API tier — 2 req/sec is well within limits
            time.sleep(0.5)

    print()
    print(f"Download complete.")
    print(f"  Written : {written} files  →  {out_dir.resolve()}")
    if errors:
        print(f"  Errors  : {errors} (see messages above)")
    print()
    print("Next step — index the documents:")
    print(f"  python scripts/ingest_pipeline.py --input-dir {args.out} --chroma-dir chroma_db")


if __name__ == "__main__":
    main()
