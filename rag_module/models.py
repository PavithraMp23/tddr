"""
rag_module/models.py
--------------------
Typed dataclasses shared across all RAG submodules.

    RegulationDocument  – output of ingestion
    RegulationChunk     – output of chunking (embedding field optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RegulationDocument:
    """
    A fully-ingested regulation document with temporal metadata.

    Parameters
    ----------
    regulation      : Short name of the regulation (e.g. "OSHA", "GDPR").
    version         : Version / edition string (e.g. "2017").
    effective_from  : ISO-8601 date string – when this version became active.
    effective_to    : ISO-8601 date string – when this version was superseded.
                      ``None`` means the version is currently active.
    text            : Full extracted text of the document.
    source_file     : Original file path (if ingested from PDF), else ``None``.
    """

    regulation: str
    version: str
    effective_from: str          # "YYYY-MM-DD"
    effective_to: Optional[str]  # "YYYY-MM-DD" or None  (currently active)
    text: str
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "regulation": self.regulation,
            "version": self.version,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "text": self.text[:120] + "..." if len(self.text) > 120 else self.text,
            "source_file": self.source_file,
        }

    def __repr__(self) -> str:
        return (
            f"RegulationDocument(regulation={self.regulation!r}, "
            f"version={self.version!r}, "
            f"effective_from={self.effective_from!r}, "
            f"effective_to={self.effective_to!r}, "
            f"text_len={len(self.text)})"
        )


@dataclass
class RegulationChunk:
    """
    A text chunk derived from a ``RegulationDocument``, ready for embedding.

    Temporal metadata is copied from the parent document so that every chunk
    can be filtered independently during retrieval.

    Parameters
    ----------
    chunk_id        : Unique identifier within the document
                      (``"{regulation}_{version}_chunk_{n}"``).
    text            : The chunk text.
    regulation      : Inherited from parent document.
    version         : Inherited from parent document.
    effective_from  : Inherited from parent document.
    effective_to    : Inherited from parent document (``None`` = currently active).
    source_file     : Inherited from parent document.
    embedding       : 384-dim numpy float32 array populated by embedding.py.
                      ``None`` until ``embed_chunks()`` is called.
    """

    chunk_id: str
    text: str
    regulation: str
    version: str
    effective_from: str
    effective_to: Optional[str]
    source_file: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:120] + "..." if len(self.text) > 120 else self.text,
            "regulation": self.regulation,
            "version": self.version,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "source_file": self.source_file,
            "has_embedding": self.embedding is not None,
        }

    def __repr__(self) -> str:
        return (
            f"RegulationChunk(chunk_id={self.chunk_id!r}, "
            f"regulation={self.regulation!r}, "
            f"version={self.version!r}, "
            f"text_len={len(self.text)}, "
            f"has_embedding={self.embedding is not None})"
        )
