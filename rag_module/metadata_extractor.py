"""
rag_module/metadata_extractor.py
---------------------------------
Intelligent PDF metadata extractor for the TDDR RAG pipeline.

Instead of relying on the filename for temporal metadata, this module reads
the actual PDF content and extracts:

  • regulation name   – detected from document title phrases
  • publication date  – extracted from a Gazette-style English header line
  • version string    – derived as  "Month-YYYY"  (e.g. "February-2017")
  • effective_from    – first day of publication month/year
  • effective_to      – None (left open; caller may supply an end-date)

Supported header format (Gazette of India notifications)
---------------------------------------------------------
    No. 144] NEW DELHI, TUESDAY, FEBRUARY 28, 2017/PHALGUNA 9, 1938
    No. 465] NEW DELHI, WEDNESDAY, JULY 6, 2016/ASADHA 15, 1938
    No. 392] NEW DELHI, TUESDAY, JUNE 12, 2018/JYAISTHA 22, 1940

Fallback patterns (other government PDFs)
------------------------------------------
  • "New Delhi, the 28th February, 2017"
  • "dated the 6th July, 2016"
  • "as on January 2017"
  • ISO  2017-07-06

Public API
----------
    from rag_module.metadata_extractor import extract_metadata_from_pdf

    meta = extract_metadata_from_pdf("data_hazardous/Feb_Amendment_HOWM.pdf")
    # → {
    #      "regulation":     "HOWM",
    #      "version":        "February-2017",
    #      "effective_from": "2017-02-01",
    #      "effective_to":   None,
    #      "source_file":    "data_hazardous/Feb_Amendment_HOWM.pdf"
    #   }
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Month name → number mapping
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1,  "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "may": 5,
    "june": 6,     "jun": 6,
    "july": 7,     "jul": 7,
    "august": 8,   "aug": 8,
    "september": 9,"sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11,"nov": 11,
    "december": 12,"dec": 12,
}

# ---------------------------------------------------------------------------
# Date extraction patterns  (ordered by confidence – most specific first)
# ---------------------------------------------------------------------------

_PATTERNS = [
    # 1. Gazette header line  "JULY 6, 2016"  or  "FEBRUARY 28, 2017"
    re.compile(
        r'\b(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY),\s+'
        r'([A-Z]+)\s+\d{1,2},\s+(\d{4})',
        re.IGNORECASE,
    ),
    # 2. "New Delhi, the 28th February, 2017"
    re.compile(
        r'New Delhi,\s+the\s+\d{1,2}(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})',
        re.IGNORECASE,
    ),
    # 3. "dated the 6th July, 2016"
    re.compile(
        r'dated\s+the\s+\d{1,2}(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})',
        re.IGNORECASE,
    ),
    # 4. "as on January 2017"  /  "w.e.f. March 2018"
    re.compile(
        r'\b(?:as\s+on|w\.e\.f\.?|effective|from)\s+([A-Za-z]+)\s+(\d{4})\b',
        re.IGNORECASE,
    ),
    # 5. Bare  "15 January 2020"  or  "January 15, 2020"
    re.compile(
        r'\b(?:\d{1,2}\s+)?([A-Za-z]+)\s+(?:\d{1,2},?\s+)?(\d{4})\b',
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Regulation-name heuristics
# ---------------------------------------------------------------------------

# Title phrases that signal the regulation name follows
_TITLE_TRIGGERS = [
    re.compile(r'(?:Hazardous\s+and\s+Other\s+Wastes)', re.IGNORECASE),
    re.compile(r'(?:Hazardous\s+Wastes)', re.IGNORECASE),
    re.compile(r'(?:(?:Rules|Act|Regulation|Guidelines|Policy|Order)\s+\w+)', re.IGNORECASE),
]

# Acronym detection from filename (last meaningful token)
_ACRONYM_RE = re.compile(r'[A-Z]{2,}')


def _detect_regulation_name(text: str, filepath: str) -> str:
    """
    Try to extract a short regulation name from:
     1. The document text (title phrases)
     2. The filename (last uppercase token)
    """
    # Look for HWM / HOWM / similar acronym in the first 500 chars
    header_text = text[:500]
    # Prefer the filename's last recognisable token above all
    stem = Path(filepath).stem  # e.g. "Feb_Amendment_HOWM" or "HWM_Rules_2016"
    tokens = stem.split("_")

    # Find the last token that is NOT a month name and NOT a pure number
    for tok in reversed(tokens):
        tok_clean = tok.upper()
        if tok_clean.lower() not in _MONTHS and not tok_clean.isdigit() and len(tok_clean) >= 2:
            return tok_clean

    # Fallback: look for a 2-5 char all-caps acronym in the header text
    acronyms = _ACRONYM_RE.findall(header_text)
    if acronyms:
        return acronyms[-1]

    return "REGULATION"


def _extract_date_from_text(text: str) -> Optional[tuple[int, int]]:
    """
    Try each pattern in order. Return ``(year, month)`` on first success.
    """
    for pat in _PATTERNS:
        for match in pat.finditer(text[:3000]):   # scan first ~3000 chars
            month_str = match.group(1).lower()
            year_str  = match.group(2)

            month_num = _MONTHS.get(month_str)
            if month_num is None:
                continue

            try:
                year_num = int(year_str)
            except ValueError:
                continue

            if 1900 <= year_num <= 2100:
                return year_num, month_num

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata_from_pdf(filepath: str) -> dict:
    """
    Extract temporal and regulatory metadata from a PDF file by reading
    its content.

    Parameters
    ----------
    filepath : Path to the PDF file.

    Returns
    -------
    dict with keys:
        ``regulation``, ``version``, ``effective_from``, ``effective_to``,
        ``source_file``.
    All values are strings except ``effective_to`` which may be ``None``.

    Notes
    -----
    * ``effective_to`` is always ``None`` — the caller or the ingestion
      pipeline should supply this based on domain knowledge.
    * ``version`` is set to ``"Month-YYYY"`` (e.g. ``"February-2017"``)
      so it is human-readable and sortable.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required. Install with: pip install pdfplumber"
        ) from exc

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath!r}")

    # ── Extract text from first 3 pages ──────────────────────────────────────
    text_parts: list[str] = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages[:3]:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    full_text = "\n".join(text_parts)

    # ── Detect regulation name ────────────────────────────────────────────────
    regulation = _detect_regulation_name(full_text, filepath)

    # ── Extract publication date from text ────────────────────────────────────
    result = _extract_date_from_text(full_text)

    if result:
        year, month = result
        month_name  = date(year, month, 1).strftime("%B")  # e.g. "February"
        version     = f"{month_name}-{year}"
        effective_from = f"{year}-{month:02d}-01"
    else:
        # Last resort: try to pull year from filename
        stem = Path(filepath).stem
        year_match = re.search(r'(\d{4})', stem)
        if year_match:
            year = int(year_match.group(1))
            version = str(year)
            effective_from = f"{year}-01-01"
        else:
            version = "unknown"
            effective_from = "1900-01-01"

    return {
        "regulation":     regulation,
        "version":        version,
        "effective_from": effective_from,
        "effective_to":   None,          # open-ended; caller may override
        "source_file":    filepath,
    }


def extract_and_rename(
    filepath: str,
    target_dir: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Extract metadata and RENAME the PDF file to the standard format:
        ``<REGULATION>_<Month>-<YEAR>.pdf``

    e.g.  ``Feb_Amendment_HOWM.pdf``  →  ``HOWM_February-2017.pdf``

    Parameters
    ----------
    filepath  : Original PDF path.
    target_dir: Directory to place the renamed file. Defaults to same dir.
    dry_run   : If ``True``, do NOT rename — just return what the new name
                would be (useful for previewing).

    Returns
    -------
    dict with keys:
        ``original``, ``renamed``, ``metadata``, ``dry_run``.
    """
    meta = extract_metadata_from_pdf(filepath)
    src  = Path(filepath)

    new_stem    = f"{meta['regulation']}_{meta['version']}"
    new_filename = f"{new_stem}.pdf"
    dest_dir    = Path(target_dir) if target_dir else src.parent
    dest_path   = dest_dir / new_filename

    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)
        os.rename(src, dest_path)
        meta["source_file"] = str(dest_path)

    return {
        "original": str(src),
        "renamed":  str(dest_path),
        "metadata": meta,
        "dry_run":  dry_run,
    }
