"""
llm_module/answer_generator.py
-------------------------------
Parses raw LLM output (a string) into a structured ``LLMResponse``.

The parser expects the LLM to return text in the format produced by
``prompt_builder.py``::

    ANSWER: <main answer>
    CITED_SECTIONS: <comma-separated sections>
    EXPLANATION: <reasoning>
    CONFIDENCE: <high|medium|low>

Falls back gracefully if the LLM doesn't follow the exact format.

Public API
----------
    from llm_module.answer_generator import AnswerGenerator
    gen      = AnswerGenerator()
    response = gen.parse(raw_text, chunks, model_used="mock-v1", prompt_tokens=512)
"""

from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

from llm_module.llm_models import LLMResponse

if TYPE_CHECKING:
    from rag_module.models import RegulationChunk


# Regex patterns to extract structured fields from LLM output
_FIELD_PATTERNS = {
    "answer":           re.compile(r"ANSWER\s*:\s*(.+?)(?=\nCITED_SECTIONS|\nEXPLANATION|\nCONFIDENCE|$)", re.DOTALL | re.IGNORECASE),
    "cited_sections":   re.compile(r"CITED_SECTIONS\s*:\s*(.+?)(?=\nANSWER|\nEXPLANATION|\nCONFIDENCE|$)", re.DOTALL | re.IGNORECASE),
    "explanation":      re.compile(r"EXPLANATION\s*:\s*(.+?)(?=\nANSWER|\nCITED_SECTIONS|\nCONFIDENCE|$)", re.DOTALL | re.IGNORECASE),
    "confidence":       re.compile(r"CONFIDENCE\s*:\s*(.+?)(?=\nANSWER|\nCITED_SECTIONS|\nEXPLANATION|$)", re.DOTALL | re.IGNORECASE),
}

# Section identifiers to detect inside raw text for auto-citation
_SECTION_RE = re.compile(
    r"\b(Section\s+\d+\w*|Article\s+\d+\w*|Rule\s+\d+\w*|Clause\s+\d+\w*)",
    re.IGNORECASE,
)

_VALID_CONFIDENCE = {"high", "medium", "low"}


class AnswerGenerator:
    """
    Converts raw LLM text into a typed ``LLMResponse``.

    Parsing strategy
    ----------------
    1. Try to extract structured fields using regex.
    2. Fall back to treating the entire raw text as the answer if parsing fails.
    3. Auto-detect cited sections from the raw text if the CITED_SECTIONS field
       is missing or empty.
    4. Cross-check cited sections against known chunk sections to set confidence.
    """

    def parse(
        self,
        raw_text: str,
        chunks: List["RegulationChunk"],
        model_used: str = "unknown",
        prompt_tokens: int = 0,
    ) -> LLMResponse:
        """
        Parse raw LLM output into a structured ``LLMResponse``.

        Parameters
        ----------
        raw_text      : Raw string returned by the LLM backend.
        chunks        : Retrieved chunks (used for citation verification).
        model_used    : LLM backend identifier (for metadata).
        prompt_tokens : Approximate character count of the prompt (for transparency).

        Returns
        -------
        LLMResponse
        """
        raw_text = raw_text.strip()

        # ── 1. Extract structured fields ─────────────────────────────────────
        answer      = self._extract("answer",         raw_text)
        cited_raw   = self._extract("cited_sections", raw_text)
        explanation = self._extract("explanation",    raw_text)
        confidence  = self._extract("confidence",     raw_text).lower().strip(".")

        # ── 2. Fallback: if no structured answer found, use all raw text ──────
        if not answer:
            answer = raw_text  # unstructured LLM output; show as-is

        if not explanation:
            explanation = "See cited sections for supporting statutory text."

        # ── 3. Parse cited sections list ──────────────────────────────────────
        cited_sections: List[str] = []
        if cited_raw and cited_raw.strip().lower() not in ("", "none", "n/a", "-"):
            # Split on commas or semicolons
            cited_sections = [
                s.strip()
                for s in re.split(r"[,;]", cited_raw)
                if s.strip()
            ]

        # Auto-detect from raw text if still empty
        if not cited_sections:
            cited_sections = list(dict.fromkeys(_SECTION_RE.findall(raw_text)))  # dedup, ordered

        # ── 4. Validate confidence ────────────────────────────────────────────
        if confidence not in _VALID_CONFIDENCE:
            confidence = self._infer_confidence(cited_sections, chunks)

        return LLMResponse(
            answer=answer.strip(),
            cited_sections=cited_sections,
            explanation=explanation.strip(),
            confidence=confidence,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(field: str, text: str) -> str:
        """Extract a single named field from structured LLM output."""
        m = _FIELD_PATTERNS[field].search(text)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _infer_confidence(
        cited_sections: List[str],
        chunks: List["RegulationChunk"],
    ) -> str:
        """
        Infer confidence from whether any cited section appears in retrieved chunks.

        high   → ≥1 cited section found verbatim inside a chunk's text
        medium → cited sections detected in answer but not verified in chunks
        low    → no citations at all
        """
        if not cited_sections:
            return "low"

        # Check if any cited section token appears in any chunk's text
        chunk_corpus = " ".join(c.text for c in chunks).lower()
        for sec in cited_sections:
            if sec.lower() in chunk_corpus:
                return "high"

        return "medium"
