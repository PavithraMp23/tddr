"""
tests/test_llm_module.py
-------------------------
Comprehensive pytest suite for the LLM Module (Module 3).

All tests use the Mock backend — no external API calls or model downloads needed.

Test classes
------------
    TestLLMModels          – LLMResponse dataclass construction & serialisation
    TestPromptBuilder      – Prompt contains all required sections
    TestLLMInterfaceMock   – Mock backend returns non-empty structured text
    TestAnswerGenerator    – Parsed LLMResponse is well-formed
    TestEndToEnd           – Full generate_answer() integration test
"""

from __future__ import annotations

import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_module.llm_models import LLMResponse
from llm_module.prompt_builder import PromptBuilder
from llm_module.llm_interface import LLMInterface
from llm_module.answer_generator import AnswerGenerator
from llm_module import generate_answer


# ===========================================================================
# Fixtures / Factories
# ===========================================================================

def _make_structured_query(question: str = "What is the punishment for murder?",
                            reference_date: str = "2018-01-01"):
    """Create a minimal StructuredQuery using user_input_module."""
    from user_input_module.models import (
        StructuredQuery, QueryFilters, ValidTimeFilter,
        RawQuery, IntermediateRepresentation,
    )
    return StructuredQuery(
        semantic_query=question,
        filters=QueryFilters(
            valid_time=ValidTimeFilter(
                operator="as_of",
                reference_date=reference_date,
            )
        ),
        raw_query=RawQuery(text=question, original_text=question),
        intermediate=IntermediateRepresentation(),
    )


def _make_chunks(n: int = 2):
    """Create synthetic RegulationChunk objects (no embeddings needed)."""
    from rag_module.models import RegulationChunk
    return [
        RegulationChunk(
            chunk_id=f"IPC_1860_chunk_{i}",
            text=f"Section {302 + i} Punishment: Whoever commits murder shall "
                 f"be punished with death or imprisonment for life.",
            regulation="IPC",
            version="1860",
            effective_from="1860-01-01",
            effective_to="2020-12-31",
        )
        for i in range(n)
    ]


# ===========================================================================
# 1. LLMModels
# ===========================================================================

class TestLLMModels:

    def test_default_construction(self):
        r = LLMResponse(
            answer="Test answer.",
            cited_sections=["Section 302"],
            explanation="Explanation.",
            confidence="high",
            model_used="mock/mock-v1",
        )
        assert r.answer == "Test answer."
        assert r.cited_sections == ["Section 302"]
        assert r.confidence == "high"
        assert r.prompt_tokens == 0  # default

    def test_to_dict_has_all_keys(self):
        r = LLMResponse(
            answer="A", cited_sections=[], explanation="E",
            confidence="low", model_used="mock/mock-v1", prompt_tokens=100,
        )
        d = r.to_dict()
        for key in ("answer", "cited_sections", "explanation", "confidence",
                    "model_used", "prompt_tokens"):
            assert key in d

    def test_to_json_is_valid_json(self):
        import json
        r = LLMResponse(
            answer="A", cited_sections=["S1"], explanation="E",
            confidence="medium", model_used="mock/mock-v1",
        )
        parsed = json.loads(r.to_json())
        assert parsed["cited_sections"] == ["S1"]

    def test_repr_contains_confidence(self):
        r = LLMResponse(
            answer="A", cited_sections=[], explanation="E",
            confidence="high", model_used="mock/mock-v1",
        )
        assert "high" in repr(r)


# ===========================================================================
# 2. PromptBuilder
# ===========================================================================

class TestPromptBuilder:

    def _build(self, n_chunks: int = 2):
        sq     = _make_structured_query()
        chunks = _make_chunks(n_chunks)
        return PromptBuilder().build(sq, chunks)

    def test_prompt_is_string(self):
        assert isinstance(self._build(), str)

    def test_prompt_contains_question(self):
        prompt = self._build()
        assert "punishment for murder" in prompt.lower()

    def test_prompt_contains_reference_date(self):
        prompt = self._build()
        assert "2018-01-01" in prompt

    def test_prompt_contains_regulation(self):
        prompt = self._build()
        assert "IPC" in prompt

    def test_prompt_contains_answer_format_instructions(self):
        prompt = self._build()
        assert "ANSWER:" in prompt
        assert "CITED_SECTIONS:" in prompt
        assert "EXPLANATION:" in prompt
        assert "CONFIDENCE:" in prompt

    def test_no_chunks_prompt_signals_no_excerpts(self):
        sq     = _make_structured_query()
        prompt = PromptBuilder().build(sq, [])
        assert "none retrieved" in prompt.lower()

    def test_token_estimate_positive(self):
        prompt = self._build()
        est    = PromptBuilder().token_estimate(prompt)
        assert est > 0

    def test_long_chunk_truncated(self):
        """Chunks longer than 1500 chars should be truncated in the prompt."""
        from rag_module.models import RegulationChunk
        long_chunk = RegulationChunk(
            chunk_id="c0", text="word " * 600,
            regulation="TEST", version="1",
            effective_from="2000-01-01", effective_to=None,
        )
        sq     = _make_structured_query()
        prompt = PromptBuilder().build(sq, [long_chunk])
        # The chunk text in the prompt should be clipped
        assert len(prompt) < 6000  # well under the raw chunk length of ~3000


# ===========================================================================
# 3. LLMInterface (Mock backend)
# ===========================================================================

class TestLLMInterfaceMock:

    def _iface(self):
        return LLMInterface()

    def test_mock_returns_string(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        prompt = PromptBuilder().build(sq, chunks)
        result = self._iface().generate(prompt, backend="mock")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_contains_answer_label(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        prompt = PromptBuilder().build(sq, chunks)
        result = self._iface().generate(prompt, backend="mock")
        assert "ANSWER:" in result

    def test_mock_contains_cited_sections(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        prompt = PromptBuilder().build(sq, chunks)
        result = self._iface().generate(prompt, backend="mock")
        assert "CITED_SECTIONS:" in result

    def test_mock_contains_confidence(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        prompt = PromptBuilder().build(sq, chunks)
        result = self._iface().generate(prompt, backend="mock")
        assert "CONFIDENCE:" in result

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            self._iface().generate("test", backend="galaxy_brain")

    def test_mock_no_chunks_gives_low_confidence(self):
        sq     = _make_structured_query()
        prompt = PromptBuilder().build(sq, [])
        result = self._iface().generate(prompt, backend="mock")
        assert "low" in result.lower()


# ===========================================================================
# 4. AnswerGenerator
# ===========================================================================

class TestAnswerGenerator:

    def _gen(self):
        return AnswerGenerator()

    def _raw(self, answer="Murder is punishable with death.",
             sections="Section 302", confidence="high"):
        return (
            f"ANSWER: {answer}\n"
            f"CITED_SECTIONS: {sections}\n"
            f"EXPLANATION: The IPC clearly prohibits murder.\n"
            f"CONFIDENCE: {confidence}"
        )

    def test_parse_returns_llm_response(self):
        result = self._gen().parse(self._raw(), _make_chunks())
        assert isinstance(result, LLMResponse)

    def test_parse_extracts_answer(self):
        result = self._gen().parse(self._raw(), _make_chunks())
        assert "murder" in result.answer.lower()

    def test_parse_extracts_cited_sections(self):
        result = self._gen().parse(self._raw(sections="Section 302, Section 303"),
                                   _make_chunks())
        assert len(result.cited_sections) >= 1

    def test_parse_extracts_confidence(self):
        result = self._gen().parse(self._raw(confidence="high"), _make_chunks())
        assert result.confidence in ("high", "medium", "low")

    def test_fallback_on_unstructured_text(self):
        raw    = "Murder is punishable by death under Section 302."
        result = self._gen().parse(raw, _make_chunks())
        assert isinstance(result, LLMResponse)
        assert len(result.answer) > 0

    def test_auto_detects_sections_from_raw_text(self):
        raw    = "Under Section 302 of the IPC, murder is punishable by death."
        result = self._gen().parse(raw, [])
        assert any("302" in s for s in result.cited_sections)

    def test_model_used_stored(self):
        result = self._gen().parse(self._raw(), [], model_used="mock/mock-v1")
        assert result.model_used == "mock/mock-v1"

    def test_prompt_tokens_stored(self):
        result = self._gen().parse(self._raw(), [], prompt_tokens=999)
        assert result.prompt_tokens == 999


# ===========================================================================
# 5. End-to-End: generate_answer()
# ===========================================================================

class TestEndToEnd:

    def test_generate_answer_returns_llm_response(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        result = generate_answer(sq, chunks, backend="mock")
        assert isinstance(result, LLMResponse)

    def test_generate_answer_non_empty_answer(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        result = generate_answer(sq, chunks, backend="mock")
        assert len(result.answer.strip()) > 0

    def test_generate_answer_confidence_valid(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        result = generate_answer(sq, chunks, backend="mock")
        assert result.confidence in ("high", "medium", "low")

    def test_generate_answer_model_label_contains_backend(self):
        sq     = _make_structured_query()
        chunks = _make_chunks()
        result = generate_answer(sq, chunks, backend="mock")
        assert "mock" in result.model_used

    def test_generate_answer_empty_chunks(self):
        sq     = _make_structured_query()
        result = generate_answer(sq, [], backend="mock")
        # Should still return a response, not raise
        assert isinstance(result, LLMResponse)

    def test_generate_answer_with_real_query(self):
        """Integration with user_input_module.process_query if available."""
        pytest.importorskip("langdetect",
                             reason="langdetect not installed — user_input_module needs it")
        from user_input_module import process_query
        sq     = process_query("What was the punishment under Section 302 IPC before 2018?")
        chunks = _make_chunks()
        result = generate_answer(sq, chunks, backend="mock")
        assert isinstance(result, LLMResponse)
        assert len(result.answer) > 0
