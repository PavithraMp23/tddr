"""
rag_module/vector_store.py
--------------------------
Step 4 of the RAG pipeline: store chunk embeddings in a FAISS flat-L2 index
alongside a parallel metadata list.

Why FAISS?
----------
* Local and fully offline — no cloud dependency.
* Exact nearest-neighbour search (FlatL2) is appropriate at academic scale
  (thousands of chunks, not millions).
* Simple persistence: one .index file + one .pkl for metadata.

Persistence layout
------------------
    <save_dir>/
    ├── faiss.index       — FAISS binary index
    └── metadata.pkl      — list[RegulationChunk] (embeddings excluded)

Public API
----------
    from rag_module.vector_store import VectorStore
    vs = VectorStore()
    vs.add_chunks(chunks)
    results = vs.search(query_vec, k=5)   # [(chunk, distance), ...]
    vs.save("rag_module/data")
    vs2 = VectorStore.load("rag_module/data")
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from rag_module.models import RegulationChunk

# FAISS dimension constant — must match the chosen embedding model
_EMBEDDING_DIM = 384


def _require_faiss():
    """Import faiss or raise a helpful error."""
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for the vector store. "
            "Install it with: pip install faiss-cpu"
        ) from exc


class VectorStore:
    """
    A FAISS-backed vector store that holds regulation chunk embeddings and
    their associated metadata.

    Attributes
    ----------
    _index    : faiss.IndexFlatL2 — the vector index.
    _metadata : list[RegulationChunk] — parallel list; index ``i`` in
                ``_metadata`` corresponds to vector ``i`` in ``_index``.
    """

    def __init__(self) -> None:
        faiss = _require_faiss()
        self._index = faiss.IndexFlatL2(_EMBEDDING_DIM)
        self._metadata: List[RegulationChunk] = []

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[RegulationChunk]) -> None:
        """
        Add a list of embedded ``RegulationChunk`` objects to the index.

        Parameters
        ----------
        chunks : Must have ``.embedding`` set on every element
                 (call ``embed_chunks()`` first).

        Raises
        ------
        ValueError
            If any chunk is missing an embedding.
        """
        if not chunks:
            return

        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"The following chunks have no embedding — call embed_chunks() first: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        matrix = np.vstack([c.embedding for c in chunks]).astype(np.float32)
        self._index.add(matrix)  # type: ignore[arg-type]
        self._metadata.extend(chunks)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[RegulationChunk, float]]:
        """
        Find the *k* nearest chunks to ``query_vec``.

        Parameters
        ----------
        query_vec : 1-D float32 array of shape ``(384,)``.
        k         : Number of results to return.

        Returns
        -------
        List of ``(RegulationChunk, distance)`` tuples, ordered by ascending
        L2 distance (most similar first).
        """
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)

        query_matrix = query_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_matrix, k)  # type: ignore

        results: List[Tuple[RegulationChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:   # FAISS sentinel for "not found"
                continue
            results.append((self._metadata[int(idx)], float(dist)))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str = "rag_module/data") -> None:
        """
        Persist the FAISS index and metadata to *directory*.

        Creates the directory if it does not exist.
        """
        faiss = _require_faiss()
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "metadata.pkl")

        faiss.write_index(self._index, index_path)

        # Strip numpy arrays before pickling to keep files small
        slim_metadata = []
        for chunk in self._metadata:
            slim = RegulationChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                regulation=chunk.regulation,
                version=chunk.version,
                effective_from=chunk.effective_from,
                effective_to=chunk.effective_to,
                source_file=chunk.source_file,
                embedding=None,   # dropped — vectors live in FAISS
            )
            slim_metadata.append(slim)

        with open(meta_path, "wb") as f:
            pickle.dump(slim_metadata, f)

    @classmethod
    def load(cls, directory: str = "rag_module/data") -> "VectorStore":
        """
        Load a previously saved ``VectorStore`` from *directory*.

        Returns
        -------
        VectorStore
            Ready for search (but embeddings in chunk objects will be ``None``
            since they are stored in FAISS, not in the metadata list).

        Raises
        ------
        FileNotFoundError
            If the directory or required files are missing.
        """
        faiss = _require_faiss()

        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "metadata.pkl")

        for path in (index_path, meta_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"VectorStore file not found: {path!r}. "
                    "Call VectorStore.save() first."
                )

        vs = cls.__new__(cls)
        vs._index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            vs._metadata = pickle.load(f)

        return vs

    def __repr__(self) -> str:
        return f"VectorStore(size={self.size})"
