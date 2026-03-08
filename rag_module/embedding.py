"""
rag_module/embedding.py
-----------------------
Step 3 of the RAG pipeline: convert text into dense vector representations
using a sentence-transformer model.

Model
-----
``sentence-transformers/all-MiniLM-L6-v2``
  * Output dimension : 384
  * Size             : ~80 MB
  * Downloaded once to ~/.cache/huggingface on first use

The model is lazily loaded as a module-level singleton so it is only
initialised once per process.

Public API
----------
    from rag_module.embedding import embed_chunks, embed_text
    chunks = embed_chunks(chunks)          # fills chunk.embedding in-place
    vec    = embed_text("some query")      # np.ndarray shape (384,)
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING
import numpy as np

from rag_module.models import RegulationChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: "_ST | None" = None


def _get_model() -> "_ST":
    """Return the shared SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """
    Encode a single string into a 384-dimensional float32 embedding vector.

    Used at query time to embed the user's query before vector search.

    Parameters
    ----------
    text : The string to encode.

    Returns
    -------
    np.ndarray
        Shape ``(384,)``, dtype ``float32``.
    """
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)


def embed_chunks(chunks: List[RegulationChunk]) -> List[RegulationChunk]:
    """
    Encode all chunks in a batch and attach the embedding to each chunk
    in-place.  The original list is also returned for convenience.

    Parameters
    ----------
    chunks : List of ``RegulationChunk`` objects (embedding may be ``None``).

    Returns
    -------
    List[RegulationChunk]
        Same list with ``.embedding`` populated on every element.
    """
    if not chunks:
        return chunks

    model = _get_model()
    texts = [c.text for c in chunks]

    # Batch encode for efficiency
    embeddings: np.ndarray = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = embeddings.astype(np.float32)

    for chunk, vec in zip(chunks, embeddings):
        chunk.embedding = vec

    return chunks
