"""
rag_module/chunking.py
----------------------
Step 2 of the RAG pipeline: split a ``RegulationDocument`` into smaller
``RegulationChunk`` objects, each inheriting the parent's temporal metadata.

Two strategies (auto-selected):
--------------------------------
1. **Section-aware split** (preferred)
   Detects headers like ``Section 1.1``, ``§ 2.3``, ``SECTION 3`` and splits
   the document at each header boundary.  Each resulting block becomes one chunk.
   If a section block is still too long (> 2 × chunk_size words) it is further
   split with the sliding-window method.

2. **Sliding-window fallback**
   Used when no section headers are detected.  Splits by word count with an
   optional word-level overlap to preserve cross-boundary context.

Public API
----------
    from rag_module.chunking import chunk_document
    chunks = chunk_document(doc, chunk_size=400, overlap=50)
"""

from __future__ import annotations

import re
from typing import List

from rag_module.models import RegulationDocument, RegulationChunk

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches lines that begin with a section header:
# "Section 1", "Section 1.2", "Section 12A", "SECTION 3", "section 4",
# "§ 1.2", "§3", "§ 12A"
# re.MULTILINE makes ^ match at the start of every line.
# re.IGNORECASE handles Section / SECTION / section.
_SECTION_HEADER_RE = re.compile(
    r"^(?:section|§)\s*\d+(?:\.\d+)*[A-Za-z]?",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split *text* into word-level sliding windows."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap  # advance by (chunk_size - overlap)

    return chunks


def _section_split(text: str) -> List[str]:
    """
    Split *text* at section header boundaries.
    Returns a list of text blocks (each starting with its header, if any).
    """
    # Find all header positions
    matches = list(_SECTION_HEADER_RE.finditer(text))

    if not matches:
        return []  # caller falls back to sliding window

    blocks: List[str] = []

    # Text before the first header (preamble)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        blocks.append(preamble)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)

    return blocks


def _make_chunk(
    text: str,
    index: int,
    doc: RegulationDocument,
) -> RegulationChunk:
    """Construct a ``RegulationChunk`` with inherited temporal metadata."""
    chunk_id = f"{doc.regulation}_{doc.version}_chunk_{index}"
    return RegulationChunk(
        chunk_id=chunk_id,
        text=text,
        regulation=doc.regulation,
        version=doc.version,
        effective_from=doc.effective_from,
        effective_to=doc.effective_to,
        source_file=doc.source_file,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: RegulationDocument,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[RegulationChunk]:
    """
    Split a ``RegulationDocument`` into a list of ``RegulationChunk`` objects.

    Parameters
    ----------
    doc        : The ingested document to chunk.
    chunk_size : Target word count per chunk (sliding-window strategy).
                 Section-aware chunks may be larger; if so they are further
                 split until they are ≤ 2 × chunk_size words.
    overlap    : Word overlap between consecutive sliding-window chunks.
                 Ignored for section-aware splits unless a section is oversized.

    Returns
    -------
    List[RegulationChunk]
        Each chunk carries full temporal metadata from the parent document.
    """
    text = doc.text.strip()
    if not text:
        return []

    raw_blocks: List[str] = _section_split(text)

    if raw_blocks:
        # Section-aware path
        # Sub-split any block that is too long
        final_blocks: List[str] = []
        for block in raw_blocks:
            if len(block.split()) > 2 * chunk_size:
                final_blocks.extend(_sliding_window(block, chunk_size, overlap))
            else:
                final_blocks.append(block)
    else:
        # Sliding-window fallback
        final_blocks = _sliding_window(text, chunk_size, overlap)

    return [
        _make_chunk(block, idx, doc)
        for idx, block in enumerate(final_blocks)
        if block.strip()
    ]
