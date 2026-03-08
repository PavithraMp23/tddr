"""
rag_module/ingestion.py
-----------------------
Step 1 of the RAG pipeline: convert raw regulation sources into
``RegulationDocument`` objects with attached temporal metadata.

Supported sources
-----------------
1. **PDF file path** (str / Path) — text extracted with ``pdfplumber``.
2. **Pre-structured dict** — use as-is (database export, synthetic test data).

Temporal metadata resolution order
-----------------------------------
1. Caller-supplied ``metadata`` kwarg overrides everything.
2. For PDFs: filename heuristic ``<NAME>_<YEAR>.pdf``
   → effective_from = YYYY-01-01; effective_to = (YYYY+4)-12-31 as best guess.
3. Defaults: version="unknown", effective_from="1900-01-01", effective_to=None.

Public API
----------
    from rag_module.ingestion import ingest_document
    doc = ingest_document("OSHA_2017.pdf")
    doc = ingest_document({"regulation": "OSHA", "version": "2017",
                           "effective_from": "2017-01-01",
                           "effective_to": "2020-12-31",
                           "text": "Chemicals must be stored ..."})
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union, Optional

from rag_module.models import RegulationDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(\d{4})")


def _parse_filename_metadata(filepath: str) -> dict:
    """
    Attempt to extract regulation name and year from a filename like:
        OSHA_2017.pdf  →  regulation=OSHA, version=2017
        GDPR_2018_v2.pdf → regulation=GDPR, version=2018
    Returns a dict with whatever could be extracted.
    """
    stem = Path(filepath).stem  # e.g. "OSHA_2017"
    parts = stem.split("_")

    regulation = parts[0].upper() if parts else "UNKNOWN"

    # Find the first 4-digit year
    year: Optional[str] = None
    for part in parts[1:]:
        m = _YEAR_RE.match(part)
        if m:
            year = m.group(1)
            break

    meta: dict = {"regulation": regulation}
    if year:
        meta["version"] = year
        meta["effective_from"] = f"{year}-01-01"
        # Heuristic: assume a 4-year validity window unless caller overrides
        meta["effective_to"] = f"{int(year) + 4}-12-31"

    return meta


def _extract_pdf_text(filepath: str) -> str:
    """
    Extract all text from a PDF using pdfplumber.
    Raises ImportError with a helpful message if the package is missing.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF ingestion. "
            "Install it with: pip install pdfplumber"
        ) from exc

    text_parts: list[str] = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(
    source: Union[str, Path, dict],
    metadata: Optional[dict] = None,
) -> RegulationDocument:
    """
    Ingest a regulation document from a PDF file or a pre-structured dict.

    Parameters
    ----------
    source   : str / Path  → treated as a PDF file path.
               dict         → treated as a pre-structured record; must contain
                              at least ``text`` and ``regulation``.
    metadata : Optional override dict. Keys accepted:
               ``regulation``, ``version``, ``effective_from``,
               ``effective_to``, ``source_file``.
               Any key supplied here takes priority over auto-detected values.

    Returns
    -------
    RegulationDocument
    """
    metadata = metadata or {}

    # ---- Branch 1: dict input -----------------------------------------------
    if isinstance(source, dict):
        merged = {**source, **metadata}  # caller metadata wins
        return RegulationDocument(
            regulation=str(merged.get("regulation", "UNKNOWN")).upper(),
            version=str(merged.get("version", "unknown")),
            effective_from=str(merged.get("effective_from", "1900-01-01")),
            effective_to=merged.get("effective_to"),          # None allowed
            text=str(merged.get("text", "")),
            source_file=merged.get("source_file"),
        )

    # ---- Branch 2: file path input ------------------------------------------
    filepath = str(source)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Regulation file not found: {filepath!r}")

    ext = Path(filepath).suffix.lower()
    if ext != ".pdf":
        raise ValueError(
            f"Unsupported file format {ext!r}. "
            "Only '.pdf' files are supported for automatic ingestion. "
            "For other formats pass a pre-structured dict."
        )

    # Extract text
    text = _extract_pdf_text(filepath)

    # Auto-detect metadata from filename
    auto_meta = _parse_filename_metadata(filepath)
    auto_meta["source_file"] = filepath

    # Caller metadata overrides auto-detection
    merged = {**auto_meta, **metadata}

    return RegulationDocument(
        regulation=str(merged.get("regulation", "UNKNOWN")).upper(),
        version=str(merged.get("version", "unknown")),
        effective_from=str(merged.get("effective_from", "1900-01-01")),
        effective_to=merged.get("effective_to"),
        text=text,
        source_file=merged.get("source_file", filepath),
    )
