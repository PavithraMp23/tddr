"""
rag_module/retrieval.py
-----------------------
Step 6 (final) of the RAG pipeline: orchestrate the full Temporal RAG flow
and return the top-k regulation chunks most relevant to a user query.

Pipeline (Temporal RAG order)
------------------------------
    1. Temporal filter    ← filter_by_time(all_chunks, query_date)
    2. Embed query        ← embed_text(query)
    3. Vector search      ← VectorStore.search(query_vec, k)
    4. Return top-k       ← List[RegulationChunk]

This is the *research novelty*: temporal filtering happens **before** vector
search, so the similarity budget is spent only on era-correct documents.

Integration with user_input_module
------------------------------------
``retrieve_from_structured`` accepts a ``StructuredQuery`` produced by Module 1
and automatically extracts ``query_date`` from
``sq.filters.valid_time.reference_date``.

Public API
----------
    from rag_module.retrieval import TemporalRAGRetriever

    retriever = TemporalRAGRetriever(vector_store, all_chunks)
    results = retriever.retrieve("chemical storage rules", query_date="2018-06-01")
    results = retriever.retrieve_from_structured(structured_query)
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from rag_module.models import RegulationChunk
from rag_module.temporal_filter import filter_by_time
from rag_module.embedding import embed_text
from rag_module.vector_store import VectorStore

if TYPE_CHECKING:
    from user_input_module.models import StructuredQuery


class TemporalRAGRetriever:
    """
    Orchestrates the complete Temporal RAG retrieval pipeline.

    Parameters
    ----------
    vector_store : A ``VectorStore`` that has been populated with chunk embeddings.
    all_chunks   : The full list of ``RegulationChunk`` objects (same order as
                   they were added to the ``VectorStore``).  Needed so the
                   temporal filter can operate on all chunks before FAISS search.

    Notes
    -----
    *all_chunks* and the FAISS index MUST be in sync (same chunks, same order).
    Use ``VectorStore.add_chunks(chunks)`` followed by passing the same *chunks*
    list to this constructor to guarantee alignment.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        all_chunks: List[RegulationChunk],
    ) -> None:
        self._store = vector_store
        self._all_chunks = all_chunks

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_date: Optional[str] = None,
        k: int = 5,
    ) -> List[RegulationChunk]:
        """
        Retrieve the top-*k* most relevant regulation chunks for *query*,
        pre-filtered to chunks that were active on *query_date*.

        Parameters
        ----------
        query      : The user's natural-language question or search string.
        query_date : ISO date string ``"YYYY-MM-DD"`` representing the point in
                     time of the query.  ``None`` means "no temporal constraint"
                     (all chunks are candidates).
        k          : Maximum number of results to return.

        Returns
        -------
        List[RegulationChunk]
            Up to *k* chunks, ordered by decreasing relevance (ascending L2
            distance from the query embedding).  May be fewer than *k* if fewer
            valid chunks exist.
        """
        # Step 1: temporal pre-filter
        valid_chunks = filter_by_time(self._all_chunks, query_date)

        if not valid_chunks:
            return []

        # Step 2: build a temporary mini-store over valid chunks only
        # (avoids rebuilding the full index on every call while still
        #  restricting search to temporally-valid documents)
        mini_store = VectorStore()
        # Only add chunks that already have embeddings
        embeddable = [c for c in valid_chunks if c.embedding is not None]
        if embeddable:
            mini_store.add_chunks(embeddable)
        else:
            return valid_chunks[:k]  # no embeddings yet — return raw filter results

        # Step 3: embed the query
        query_vec = embed_text(query)

        # Step 4: vector search
        search_results = mini_store.search(query_vec, k=k)

        return [chunk for chunk, _dist in search_results]

    # ------------------------------------------------------------------
    # Integration with user_input_module
    # ------------------------------------------------------------------

    def retrieve_from_structured(
        self,
        structured_query: "StructuredQuery",
        k: int = 5,
    ) -> List[RegulationChunk]:
        """
        Retrieve using a ``StructuredQuery`` produced by Module 1
        (``user_input_module``).

        Extracts:
        * ``semantic_query``                    → search string
        * ``filters.valid_time``                → operator + dates → query_date

        Operator → query_date mapping
        ------------------------------
        ``before``   → one day before reference_date
                        (find chunks active in the era *preceding* the date)
        ``after``    → reference_date itself
                        (find chunks active from that date onward)
        ``as_of``    → reference_date
                        (find chunks active exactly on that date)
        ``between``  → reference_date
                        (find chunks active at the start of the range)
        ``in_year``  → reference_date (start of year)
        ``current``  → None  (no temporal filter; return all active chunks)

        Parameters
        ----------
        structured_query : Output of ``user_input_module.process_query()``.
        k                : Maximum results.

        Returns
        -------
        List[RegulationChunk]
        """
        from datetime import date, timedelta

        query_text = structured_query.semantic_query
        vt = structured_query.filters.valid_time
        operator = vt.operator
        ref_date_str = vt.reference_date

        if operator == "current" or ref_date_str is None:
            # No temporal constraint — search across all chunks
            query_date = None
        elif operator == "before":
            # We want chunks that were active just before the reference date.
            # Subtract one day so the filter includes chunks whose effective_to
            # equals the day before the boundary (e.g. 2017-12-31 for "before 2018").
            try:
                rd = date.fromisoformat(ref_date_str)
                query_date = (rd - timedelta(days=1)).isoformat()
            except (ValueError, TypeError):
                query_date = ref_date_str
        else:
            # after / as_of / between / in_year — use reference_date as-is
            query_date = ref_date_str

        return self.retrieve(query_text, query_date=query_date, k=k)

    def __repr__(self) -> str:
        return f"TemporalRAGRetriever(store={self._store}, chunks={len(self._all_chunks)})"
