"""
tests/test_output_module.py
----------------------------
Unit tests for the Output Module (Module 4) of the TDDR pipeline.

Covers:
    - CitationResolver  — exact match, keyword match, no match, deduplication
    - ResponseBuilder   — full build(), metadata presence
    - FinalSystemResponse / LegalEvidence — serialisation
"""

from __future__ import annotations

import sys
import os
import pytest

# Ensure project root is importable when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from output_module.response_models import LegalEvidence, FinalSystemResponse, EXCERPT_MAX_CHARS
from output_module.citation_resolver import CitationResolver
from output_module.response_builder import ResponseBuilder


# ---------------------------------------------------------------------------
# Fake data helpers
# ---------------------------------------------------------------------------

class _FakeChunk:
    """Minimal stand-in for ``RegulationChunk``."""
    def __init__(self, regulation, version, effective_from, effective_to, text,
                 chunk_id="chunk_1", source_file=None):
        self.chunk_id       = chunk_id
        self.regulation     = regulation
        self.version        = version
        self.effective_from = effective_from
        self.effective_to   = effective_to
        self.text           = text
        self.source_file    = source_file
        self.embedding      = None


class _FakeLLMResponse:
    """Minimal stand-in for ``LLMResponse``."""
    def __init__(self, answer="Test answer.", cited_sections=None,
                 explanation="Test explanation.", confidence="high",
                 model_used="mock/test", prompt_tokens=100):
        self.answer          = answer
        self.cited_sections  = cited_sections or []
        self.explanation     = explanation
        self.confidence      = confidence
        self.model_used      = model_used
        self.prompt_tokens   = prompt_tokens


_CHUNK_RULE5 = _FakeChunk(
    regulation="SWM_RULES",
    version="2016",
    effective_from="2016-04-01",
    effective_to=None,
    text=(
        "Rule 5 — Duties of Local Authorities\n"
        "No person shall dispose construction and demolition waste or debris "
        "on public roads, footpaths, drains, or any public place in the city. "
        "Violation of this rule shall attract a fine as specified by the authority."
    ),
    chunk_id="SWM_RULES_2016_chunk_0",
)

_CHUNK_RULE10 = _FakeChunk(
    regulation="SWM_RULES",
    version="2021",
    effective_from="2021-01-01",
    effective_to=None,
    text=(
        "Rule 10 — Extended Producer Responsibility\n"
        "Every producer, importer and brand owner shall be responsible for "
        "collecting back plastic waste generated due to their products."
    ),
    chunk_id="SWM_RULES_2021_chunk_1",
)


# ---------------------------------------------------------------------------
# CitationResolver Tests
# ---------------------------------------------------------------------------

class TestCitationResolver:

    def setup_method(self):
        self.resolver = CitationResolver()

    def test_exact_match_returns_evidence(self):
        """A cited section present verbatim in a chunk → LegalEvidence returned."""
        ev = self.resolver.resolve(["Rule 5"], [_CHUNK_RULE5])
        assert len(ev) == 1
        assert ev[0].section      == "Rule 5"
        assert ev[0].regulation   == "SWM_RULES"
        assert ev[0].version      == "2016"
        assert ev[0].effective_from == "2016-04-01"
        assert "Rule 5" in ev[0].excerpt or len(ev[0].excerpt) > 0

    def test_case_insensitive_match(self):
        """Match should be case-insensitive."""
        ev = self.resolver.resolve(["rule 5"], [_CHUNK_RULE5])
        assert len(ev) == 1
        assert ev[0].regulation == "SWM_RULES"

    def test_no_match_returns_unknown_evidence(self):
        """Unmatched citation → UNKNOWN hollow record, NOT dropped."""
        # Use a section label with no keyword overlap with the SWM_RULES chunks
        ev = self.resolver.resolve(["XYZ_NONEXISTENT_SECTION_999"], [_CHUNK_RULE5, _CHUNK_RULE10])
        assert len(ev) == 1
        assert ev[0].regulation == "UNKNOWN"
        assert ev[0].excerpt == ""
        assert "XYZ_NONEXISTENT_SECTION_999" in ev[0].section

    def test_deduplication(self):
        """Same (regulation, version, section) should only appear once."""
        # Two identical citations
        ev = self.resolver.resolve(["Rule 5", "Rule 5"], [_CHUNK_RULE5])
        assert len(ev) == 1

    def test_multiple_citations_resolved_to_different_chunks(self):
        """Each citation resolves to the correct chunk independently."""
        ev = self.resolver.resolve(["Rule 5", "Rule 10"], [_CHUNK_RULE5, _CHUNK_RULE10])
        assert len(ev) == 2
        regs  = {e.regulation for e in ev}
        vers  = {e.version    for e in ev}
        sects = {e.section    for e in ev}
        assert regs  == {"SWM_RULES"}
        assert vers  == {"2016", "2021"}
        assert sects == {"Rule 5", "Rule 10"}

    def test_empty_cited_sections(self):
        """Empty input → empty output."""
        ev = self.resolver.resolve([], [_CHUNK_RULE5])
        assert ev == []

    def test_empty_chunks(self):
        """No chunks → all citations return UNKNOWN."""
        ev = self.resolver.resolve(["Rule 5"],  [])
        assert len(ev) == 1
        assert ev[0].regulation == "UNKNOWN"

    def test_keyword_fallback_match(self):
        """If exact substring misses, keyword fallback should still hit."""
        # "disposal" is a keyword in "Rule 5" chunk text but not cited exactly
        ev = self.resolver.resolve(["construction disposal"], [_CHUNK_RULE5])
        # Should find a match via keyword fallback
        assert len(ev) == 1
        assert ev[0].regulation != "UNKNOWN"


# ---------------------------------------------------------------------------
# Excerpt Tests
# ---------------------------------------------------------------------------

class TestExcerptExtraction:

    def setup_method(self):
        self.resolver = CitationResolver()

    def test_excerpt_max_length(self):
        """Excerpt should never exceed EXCERPT_MAX_CHARS + ellipsis overhead."""
        long_chunk = _FakeChunk(
            regulation="TEST", version="2023",
            effective_from="2023-01-01", effective_to=None,
            text="Rule 1 " + "x" * 600,
        )
        ev = self.resolver.resolve(["Rule 1"], [long_chunk])
        assert len(ev[0].excerpt) <= EXCERPT_MAX_CHARS + 10  # +10 for ellipsis chars

    def test_excerpt_contains_section_context(self):
        """Excerpt should contain text near the matched section."""
        ev = self.resolver.resolve(["Rule 5"], [_CHUNK_RULE5])
        # Should start near "Rule 5"
        assert "Rule 5" in ev[0].excerpt or "Duties" in ev[0].excerpt


# ---------------------------------------------------------------------------
# ResponseBuilder Tests
# ---------------------------------------------------------------------------

class TestResponseBuilder:

    def setup_method(self):
        self.builder = ResponseBuilder()
        self.chunks  = [_CHUNK_RULE5, _CHUNK_RULE10]
        self.llm     = _FakeLLMResponse(
            cited_sections=["Rule 5"],
            model_used="mock/mistral",
            prompt_tokens=250,
        )

    def test_build_returns_final_system_response(self):
        final = self.builder.build(self.llm, self.chunks)
        assert isinstance(final, FinalSystemResponse)

    def test_answer_propagated(self):
        final = self.builder.build(self.llm, self.chunks)
        assert final.answer == self.llm.answer

    def test_confidence_propagated(self):
        final = self.builder.build(self.llm, self.chunks)
        assert final.confidence == "high"

    def test_legal_basis_populated(self):
        final = self.builder.build(self.llm, self.chunks)
        assert len(final.legal_basis) >= 1
        assert final.legal_basis[0].section == "Rule 5"

    def test_metadata_fields_present(self):
        final = self.builder.build(self.llm, self.chunks)
        assert "model_used"       in final.metadata
        assert "retrieved_chunks" in final.metadata
        assert "generated_at"     in final.metadata
        assert final.metadata["retrieved_chunks"] == len(self.chunks)
        assert final.metadata["model_used"] == "mock/mistral"

    def test_zero_chunks_no_crash(self):
        final = self.builder.build(self.llm, [])
        assert isinstance(final, FinalSystemResponse)
        assert final.metadata["retrieved_chunks"] == 0


# ---------------------------------------------------------------------------
# FinalSystemResponse / LegalEvidence serialisation Tests
# ---------------------------------------------------------------------------

class TestSerialisation:

    def _make_evidence(self):
        return LegalEvidence(
            regulation="SWM_RULES", version="2016",
            section="Rule 5", effective_from="2016-04-01",
            effective_to=None, excerpt="No person shall dispose..."
        )

    def _make_response(self):
        return FinalSystemResponse(
            answer="Test answer.", confidence="high",
            explanation="Explanation text.",
            legal_basis=[self._make_evidence()],
            metadata={"model_used": "mock", "retrieved_chunks": 1},
        )

    def test_legal_evidence_to_dict_keys(self):
        d = self._make_evidence().to_dict()
        for key in ("regulation", "version", "section", "effective_from",
                    "effective_to", "excerpt", "source_file"):
            assert key in d, f"Missing key: {key}"

    def test_final_response_to_dict_keys(self):
        d = self._make_response().to_dict()
        for key in ("answer", "confidence", "explanation", "legal_basis", "metadata"):
            assert key in d, f"Missing key: {key}"

    def test_legal_basis_is_list_of_dicts(self):
        d = self._make_response().to_dict()
        assert isinstance(d["legal_basis"], list)
        assert isinstance(d["legal_basis"][0], dict)

    def test_to_json_valid_json(self):
        import json
        j = self._make_response().to_json()
        parsed = json.loads(j)
        assert parsed["confidence"] == "high"

    def test_effective_to_none_serialised(self):
        d = self._make_evidence().to_dict()
        assert d["effective_to"] is None
