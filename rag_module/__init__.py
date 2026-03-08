"""
rag_module
----------
Temporal RAG Pipeline for the TDDR (Temporal Regulation Drift Detector) system.

This module converts raw regulation documents (PDFs or dicts) into temporally-
aware vector embeddings and retrieves era-correct clauses for a given query date.

Pipeline
--------
    ingest_document()   →   chunk_document()   →   embed_chunks()
          ↓                                              ↓
    RegulationDocument         RegulationChunk[]    (384-dim vectors)
                                      ↓
                            VectorStore.add_chunks()
                                      ↓
                         TemporalRAGRetriever.retrieve()
                         (temporal filter BEFORE vector search)

Public API
----------
    from rag_module import (
        ingest_document,
        chunk_document,
        embed_chunks, embed_text,
        VectorStore,
        filter_by_time, TemporalFilter,
        TemporalRAGRetriever,
    )
"""

from rag_module.ingestion import ingest_document
from rag_module.chunking import chunk_document
from rag_module.embedding import embed_chunks, embed_text
from rag_module.vector_store import VectorStore
from rag_module.temporal_filter import filter_by_time, TemporalFilter
from rag_module.retrieval import TemporalRAGRetriever
from rag_module.models import RegulationDocument, RegulationChunk

__all__ = [
    # Submodule functions
    "ingest_document",
    "chunk_document",
    "embed_chunks",
    "embed_text",
    # Classes
    "VectorStore",
    "TemporalFilter",
    "TemporalRAGRetriever",
    # Filter function
    "filter_by_time",
    # Models
    "RegulationDocument",
    "RegulationChunk",
]
