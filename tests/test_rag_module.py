"""
tests/test_rag_module.py
-------------------------
Comprehensive pytest suite for the RAG Module (Module 2).

All tests use synthetic in-memory data — no real PDFs or internet access needed.
(The embedding tests DO require the sentence-transformers model to be downloaded
on first run — roughly 40s on a fast connection.)

Test classes
------------
    TestIngestion       – ingest_document() from dict and PDF-path parsing
    TestChunking        – chunk_document(), section-split and sliding-window
    TestEmbedding       – embed_text() and embed_chunks() vector shapes
    TestVectorStore     – add, search, save/load round-trip
    TestTemporalFilter  – filter_by_time() boundary conditions
    TestRetrieval       – end-to-end retrieve() and retrieve_from_structured()
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np
import pytest

from rag_module.models import RegulationDocument, RegulationChunk
from rag_module.ingestion import ingest_document, _parse_filename_metadata
from rag_module.chunking import chunk_document
from rag_module.temporal_filter import filter_by_time, TemporalFilter


# ===========================================================================
# Fixtures / Factories
# ===========================================================================

SAMPLE_TEXT = (
    "Section 1 General Provisions\n"
    "Chemicals must be stored in ventilated containers away from heat sources. "
    "All storage areas must be labelled with appropriate hazard signs.\n\n"
    "Section 2 Handling Procedures\n"
    "Employees must wear protective equipment when handling corrosive materials. "
    "Gloves, goggles, and aprons are mandatory.\n\n"
    "Section 3 Disposal\n"
    "Chemical waste must be collected in designated containers. "
    "Disposal must comply with local environmental regulations."
)


def _make_doc(
    regulation: str = "OSHA",
    version: str = "2017",
    effective_from: str = "2017-01-01",
    effective_to: str | None = "2020-12-31",
    text: str = SAMPLE_TEXT,
) -> RegulationDocument:
    return RegulationDocument(
        regulation=regulation,
        version=version,
        effective_from=effective_from,
        effective_to=effective_to,
        text=text,
    )


def _make_chunk(
    chunk_id: str = "OSHA_2017_chunk_0",
    text: str = "Chemicals must be stored in ventilated containers.",
    effective_from: str = "2017-01-01",
    effective_to: str | None = "2020-12-31",
) -> RegulationChunk:
    return RegulationChunk(
        chunk_id=chunk_id,
        text=text,
        regulation="OSHA",
        version="2017",
        effective_from=effective_from,
        effective_to=effective_to,
    )


def _embed_chunk(chunk: RegulationChunk, dim: int = 384) -> RegulationChunk:
    """Attach a synthetic (random) embedding for tests that don't need real models."""
    rng = np.random.default_rng(seed=42)
    chunk.embedding = rng.random(dim).astype(np.float32)
    return chunk


def _make_embedded_chunks(n: int = 3) -> List[RegulationChunk]:
    chunks = []
    for i in range(n):
        c = _make_chunk(chunk_id=f"OSHA_2017_chunk_{i}", text=f"Chunk {i} text about chemicals.")
        _embed_chunk(c)
        chunks.append(c)
    return chunks


# ===========================================================================
# 1. Ingestion Tests
# ===========================================================================

class TestIngestion:

    def test_dict_ingestion_basic(self):
        doc = ingest_document({
            "regulation": "osha",
            "version": "2017",
            "effective_from": "2017-01-01",
            "effective_to": "2020-12-31",
            "text": "All chemicals must be stored safely.",
        })
        assert doc.regulation == "OSHA"  # uppercased
        assert doc.version == "2017"
        assert doc.effective_from == "2017-01-01"
        assert doc.effective_to == "2020-12-31"
        assert "chemicals" in doc.text.lower()

    def test_dict_effective_to_none(self):
        doc = ingest_document({
            "regulation": "GDPR",
            "version": "2018",
            "effective_from": "2018-05-25",
            "text": "Personal data must be processed lawfully.",
        })
        assert doc.effective_to is None  # currently active

    def test_dict_defaults_for_missing_fields(self):
        doc = ingest_document({"text": "Some regulation text."})
        assert doc.regulation == "UNKNOWN"
        assert doc.version == "unknown"
        assert doc.effective_from == "1900-01-01"

    def test_metadata_override_for_dict(self):
        doc = ingest_document(
            {"regulation": "OSHA", "version": "2017",
             "effective_from": "2017-01-01", "text": "..."},
            metadata={"version": "2021"},  # override
        )
        assert doc.version == "2021"

    def test_filename_metadata_parser_standard(self):
        meta = _parse_filename_metadata("OSHA_2017.pdf")
        assert meta["regulation"] == "OSHA"
        assert meta["version"] == "2017"
        assert meta["effective_from"] == "2017-01-01"
        assert meta["effective_to"] == "2021-12-31"  # heuristic +4 years

    def test_filename_metadata_parser_gdpr(self):
        meta = _parse_filename_metadata("GDPR_2018_regulation.pdf")
        assert meta["regulation"] == "GDPR"
        assert meta["version"] == "2018"

    def test_filename_metadata_no_year(self):
        meta = _parse_filename_metadata("SomeRegulation.pdf")
        assert meta["regulation"] == "SOMEREGULATION"
        assert "version" not in meta

    def test_nonexistent_pdf_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest_document("/nonexistent/path/OSHA_2017.pdf")

    def test_non_pdf_extension_raises_value_error(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(b"fake docx")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                ingest_document(path)
        finally:
            os.unlink(path)


# ===========================================================================
# 2. Chunking Tests
# ===========================================================================

class TestChunking:

    def test_returns_list_of_chunks(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, RegulationChunk) for c in chunks)

    def test_section_aware_split_detects_sections(self):
        doc = _make_doc(text=SAMPLE_TEXT)
        chunks = chunk_document(doc)
        # SAMPLE_TEXT has 3 sections; should produce ≥ 3 chunks
        assert len(chunks) >= 3

    def test_metadata_propagation(self):
        doc = _make_doc(regulation="GDPR", version="2018",
                        effective_from="2018-05-25", effective_to=None)
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.regulation == "GDPR"
            assert c.version == "2018"
            assert c.effective_from == "2018-05-25"
            assert c.effective_to is None

    def test_chunk_ids_unique(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_sliding_window_fallback_on_plain_text(self):
        plain = "word " * 1000  # no section headers
        doc = _make_doc(text=plain)
        chunks = chunk_document(doc, chunk_size=100, overlap=20)
        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_overlap_creates_shared_words(self):
        # Build text that is exactly 3× chunk_size words
        text = " ".join([f"word{i}" for i in range(300)])
        doc = _make_doc(text=text)
        chunks = chunk_document(doc, chunk_size=100, overlap=20)
        # With overlap every chunk except the last should share 20 words with the next
        assert len(chunks) >= 2

    def test_empty_text_returns_empty_list(self):
        doc = _make_doc(text="")
        assert chunk_document(doc) == []

    def test_chunk_text_nonempty(self):
        doc = _make_doc()
        chunks = chunk_document(doc)
        assert all(c.text.strip() for c in chunks)


# ===========================================================================
# 3. Embedding Tests
# ===========================================================================

class TestEmbedding:
    """
    These tests require sentence-transformers to be installed.
    They are slow (~2s per test class on first run due to model loading).
    """

    @pytest.fixture(autouse=True)
    def _require_st(self):
        pytest.importorskip("sentence_transformers",
                             reason="sentence-transformers not installed")

    def test_embed_text_shape(self):
        from rag_module.embedding import embed_text
        vec = embed_text("Chemicals must be stored safely.")
        assert vec.shape == (384,)

    def test_embed_text_dtype(self):
        from rag_module.embedding import embed_text
        vec = embed_text("Safety regulations apply.")
        assert vec.dtype == np.float32

    def test_embed_text_not_all_zeros(self):
        from rag_module.embedding import embed_text
        vec = embed_text("OSHA chemical storage requirements.")
        assert not np.all(vec == 0)

    def test_embed_chunks_fills_embedding(self):
        from rag_module.embedding import embed_chunks
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(3)]
        result = embed_chunks(chunks)
        assert all(c.embedding is not None for c in result)

    def test_embed_chunks_returns_same_list(self):
        from rag_module.embedding import embed_chunks
        chunks = [_make_chunk()]
        result = embed_chunks(chunks)
        assert result is chunks  # in-place modification, same list

    def test_embed_chunks_correct_dim(self):
        from rag_module.embedding import embed_chunks
        chunks = [_make_chunk()]
        embed_chunks(chunks)
        assert chunks[0].embedding.shape == (384,)

    def test_embed_empty_list(self):
        from rag_module.embedding import embed_chunks
        result = embed_chunks([])
        assert result == []


# ===========================================================================
# 4. VectorStore Tests
# ===========================================================================

class TestVectorStore:

    @pytest.fixture(autouse=True)
    def _require_faiss(self):
        pytest.importorskip("faiss", reason="faiss-cpu not installed")

    def test_initial_size_zero(self):
        from rag_module.vector_store import VectorStore
        vs = VectorStore()
        assert vs.size == 0

    def test_add_chunks_increases_size(self):
        from rag_module.vector_store import VectorStore
        vs = VectorStore()
        chunks = _make_embedded_chunks(3)
        vs.add_chunks(chunks)
        assert vs.size == 3

    def test_search_returns_correct_count(self):
        from rag_module.vector_store import VectorStore
        vs = VectorStore()
        chunks = _make_embedded_chunks(5)
        vs.add_chunks(chunks)
        qvec = np.random.rand(384).astype(np.float32)
        results = vs.search(qvec, k=3)
        assert len(results) == 3

    def test_search_returns_chunks_and_distances(self):
        from rag_module.vector_store import VectorStore
        vs = VectorStore()
        chunks = _make_embedded_chunks(2)
        vs.add_chunks(chunks)
        qvec = np.random.rand(384).astype(np.float32)
        results = vs.search(qvec, k=2)
        assert all(isinstance(c, RegulationChunk) for c, _ in results)
        assert all(isinstance(d, float) for _, d in results)

    def test_search_empty_store_returns_empty(self):
        from rag_module.vector_store import VectorStore
        vs = VectorStore()
        qvec = np.random.rand(384).astype(np.float32)
        assert vs.search(qvec, k=5) == []

    def test_add_chunks_missing_embedding_raises(self):
        from rag_module.vector_store import VectorStore
        vs = VectorStore()
        chunk = _make_chunk()  # no embedding
        with pytest.raises(ValueError, match="no embedding"):
            vs.add_chunks([chunk])

    def test_save_and_load_round_trip(self):
        from rag_module.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore()
            chunks = _make_embedded_chunks(4)
            vs.add_chunks(chunks)
            vs.save(tmpdir)

            assert os.path.isfile(os.path.join(tmpdir, "faiss.index"))
            assert os.path.isfile(os.path.join(tmpdir, "metadata.pkl"))

            vs2 = VectorStore.load(tmpdir)
            assert vs2.size == 4

    def test_load_missing_dir_raises(self):
        from rag_module.vector_store import VectorStore
        with pytest.raises(FileNotFoundError):
            VectorStore.load("/nonexistent/path/store")


# ===========================================================================
# 5. Temporal Filter Tests
# ===========================================================================

class TestTemporalFilter:

    def _chunk(self, eff_from, eff_to):
        return _make_chunk(effective_from=eff_from, effective_to=eff_to)

    def test_in_range_returns_chunk(self):
        c = self._chunk("2017-01-01", "2020-12-31")
        result = filter_by_time([c], "2018-06-01")
        assert c in result

    def test_before_range_excluded(self):
        c = self._chunk("2017-01-01", "2020-12-31")
        result = filter_by_time([c], "2016-12-31")
        assert c not in result

    def test_after_range_excluded(self):
        c = self._chunk("2017-01-01", "2020-12-31")
        result = filter_by_time([c], "2021-01-01")
        assert c not in result

    def test_on_effective_from_boundary(self):
        c = self._chunk("2017-01-01", "2020-12-31")
        result = filter_by_time([c], "2017-01-01")
        assert c in result

    def test_on_effective_to_boundary(self):
        c = self._chunk("2017-01-01", "2020-12-31")
        result = filter_by_time([c], "2020-12-31")
        assert c in result

    def test_open_ended_effective_to_included(self):
        """effective_to=None means currently active → always included."""
        c = self._chunk("2017-01-01", None)
        result = filter_by_time([c], "2025-01-01")
        assert c in result

    def test_no_query_date_returns_all(self):
        chunks = [
            self._chunk("2017-01-01", "2020-12-31"),
            self._chunk("2021-01-01", None),
        ]
        result = filter_by_time(chunks, None)
        assert len(result) == 2

    def test_mixed_bag(self):
        chunks = [
            self._chunk("2015-01-01", "2016-12-31"),  # expired before query
            self._chunk("2017-01-01", "2020-12-31"),  # valid
            self._chunk("2021-01-01", None),           # not yet active
        ]
        result = filter_by_time(chunks, "2018-06-01")
        assert len(result) == 1
        assert result[0].effective_from == "2017-01-01"

    def test_class_interface(self):
        tf = TemporalFilter()
        c = self._chunk("2017-01-01", "2020-12-31")
        result = tf.filter([c], "2019-03-15")
        assert c in result

    def test_invalid_date_string_returns_all(self):
        """If query_date is unparseable, fail-open and return all chunks."""
        c = self._chunk("2017-01-01", "2020-12-31")
        result = filter_by_time([c], "not-a-date")
        assert c in result


# ===========================================================================
# 6. Retrieval Tests
# ===========================================================================

class TestRetrieval:

    @pytest.fixture(autouse=True)
    def _require_deps(self):
        pytest.importorskip("faiss", reason="faiss-cpu not installed")
        pytest.importorskip("sentence_transformers",
                             reason="sentence-transformers not installed")

    def _build_retriever(self):
        """Build a small in-memory retriever with 4 synthetic chunks."""
        from rag_module.embedding import embed_chunks
        from rag_module.vector_store import VectorStore
        from rag_module.retrieval import TemporalRAGRetriever

        chunks = [
            RegulationChunk(
                chunk_id=f"OSHA_2017_chunk_{i}",
                text=f"OSHA 2017 rule {i}: chemical storage requirement.",
                regulation="OSHA", version="2017",
                effective_from="2017-01-01", effective_to="2020-12-31",
            )
            for i in range(3)
        ] + [
            RegulationChunk(
                chunk_id="OSHA_2021_chunk_0",
                text="OSHA 2021 updated rule: ventilation mandatory.",
                regulation="OSHA", version="2021",
                effective_from="2021-01-01", effective_to=None,
            )
        ]
        embed_chunks(chunks)

        vs = VectorStore()
        vs.add_chunks(chunks)
        return TemporalRAGRetriever(vs, chunks), chunks

    def test_retrieve_returns_list(self):
        retriever, _ = self._build_retriever()
        results = retriever.retrieve("chemical storage", query_date="2018-06-01")
        assert isinstance(results, list)

    def test_retrieve_with_date_filters_era(self):
        retriever, _ = self._build_retriever()
        results = retriever.retrieve("chemical storage", query_date="2018-06-01", k=5)
        # Only 2017 version should appear (2021 not in range)
        for c in results:
            assert c.version == "2017"

    def test_retrieve_without_date_returns_all_eras(self):
        retriever, _ = self._build_retriever()
        results = retriever.retrieve("chemical storage", query_date=None, k=10)
        versions = {c.version for c in results}
        assert len(versions) >= 2  # both 2017 and 2021

    def test_retrieve_respects_k(self):
        retriever, _ = self._build_retriever()
        results = retriever.retrieve("chemical storage", query_date=None, k=2)
        assert len(results) <= 2

    def test_retrieve_from_structured_query(self):
        """Integration test with the user_input_module StructuredQuery."""
        pytest.importorskip("langdetect",
                             reason="langdetect not installed — needed by user_input_module")
        from user_input_module import process_query
        retriever, _ = self._build_retriever()
        sq = process_query("What were the chemical storage rules in 2018?")
        results = retriever.retrieve_from_structured(sq, k=5)
        assert isinstance(results, list)

    def test_retrieve_empty_store_returns_empty(self):
        from rag_module.vector_store import VectorStore
        from rag_module.retrieval import TemporalRAGRetriever
        vs = VectorStore()
        retriever = TemporalRAGRetriever(vs, [])
        results = retriever.retrieve("anything", query_date="2018-01-01")
        assert results == []
