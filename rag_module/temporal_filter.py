"""
rag_module/temporal_filter.py
------------------------------
Step 5 of the RAG pipeline — and the core research contribution of the TDDR
system.

Traditional RAG
    Retrieve ALL chunks → post-filter by date
    ❌ wastes similarity budget on irrelevant-era documents

Temporal RAG (this module)
    Filter by date FIRST → run vector search only on valid chunks
    ✅ every retrieved chunk is guaranteed to be era-correct

Filtering logic
---------------
A chunk is valid for a given query date if:

    effective_from ≤ query_date ≤ effective_to

Special cases:
  * ``effective_to = None``  → chunk is "currently active" (no expiry).
    Only included when query_date IS None (current query) or query_date falls
    within the open-ended range.
  * ``query_date = None``    → no filter; return all chunks (current-time query).

Public API
----------
    from rag_module.temporal_filter import TemporalFilter, filter_by_time

    # Functional interface
    valid = filter_by_time(chunks, query_date="2018-06-01")

    # Class interface (re-usable across queries)
    tf = TemporalFilter()
    valid = tf.filter(chunks, query_date="2018-06-01")
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from rag_module.models import RegulationChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse an ISO date string ``"YYYY-MM-DD"`` → ``datetime.date``.
    Returns ``None`` if *date_str* is ``None`` or unparseable."""
    if date_str is None:
        return None
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def _chunk_valid(chunk: RegulationChunk, query_date: date) -> bool:
    """
    Return True if *chunk* was in force on *query_date*.

    Logic
    -----
    effective_from ≤ query_date  AND  (effective_to IS NULL  OR  query_date ≤ effective_to)
    """
    from_date = _parse_date(chunk.effective_from)
    to_date = _parse_date(chunk.effective_to)

    # effective_from must be parseable; if not, be permissive
    if from_date is not None and query_date < from_date:
        return False

    # effective_to = None means currently active → valid for any query_date
    if to_date is not None and query_date > to_date:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_by_time(
    chunks: List[RegulationChunk],
    query_date: Optional[str],
) -> List[RegulationChunk]:
    """
    Keep only chunks that were in force on *query_date*.

    Parameters
    ----------
    chunks     : Full list of ``RegulationChunk`` objects.
    query_date : ISO date string (``"YYYY-MM-DD"``) representing the point in
                 time of interest.  ``None`` means "no temporal constraint"
                 (all chunks are returned).

    Returns
    -------
    List[RegulationChunk]
        Subset of *chunks* valid at *query_date*, preserving original order.
    """
    if query_date is None:
        return list(chunks)   # no filter — current-time query

    qd = _parse_date(query_date)
    if qd is None:
        # Unparseable date → return all (fail-open)
        return list(chunks)

    return [c for c in chunks if _chunk_valid(c, qd)]


class TemporalFilter:
    """
    Stateless wrapper around ``filter_by_time`` for use as a pipeline component.

    Example
    -------
    >>> tf = TemporalFilter()
    >>> valid_chunks = tf.filter(all_chunks, query_date="2019-03-15")
    """

    def filter(
        self,
        chunks: List[RegulationChunk],
        query_date: Optional[str],
    ) -> List[RegulationChunk]:
        """Delegate to the module-level ``filter_by_time`` function."""
        return filter_by_time(chunks, query_date)

    def __repr__(self) -> str:
        return "TemporalFilter()"
