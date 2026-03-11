"""
llm_module/prompt_builder.py
-----------------------------
Converts a StructuredQuery + retrieved RegulationChunks into a well-formed
LLM prompt that minimises hallucination and maximises citation fidelity.

Public API
----------
    from llm_module.prompt_builder import PromptBuilder
    builder = PromptBuilder()
    prompt  = builder.build(structured_query, chunks)
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from user_input_module.models import StructuredQuery
    from rag_module.models import RegulationChunk


_SYSTEM_PREAMBLE = """\
You are a legal compliance assistant specialised in Indian and international regulations.

INSTRUCTIONS:
1. Answer the question using ONLY the regulation excerpts provided below.
2. Do NOT invent or assume any law that is not shown in the excerpts.
3. Always cite the exact regulation name and section when making a claim.
4. If the regulations do not contain enough information, say so clearly.
5. Keep your answer concise, accurate, and legally precise.
"""

_ANSWER_FORMAT = """\
Respond in the following exact format (no extra keys, no markdown headers):

ANSWER: <one-paragraph direct answer>
CITED_SECTIONS: <comma-separated list of section identifiers, e.g. Section 302, Article 5>
EXPLANATION: <one or two sentences explaining the legal basis>
CONFIDENCE: <high | medium | low>
"""


class PromptBuilder:
    """
    Builds a grounded prompt for retrieval-augmented legal Q&A.

    The prompt structure is:
        [System preamble]
        Reference date: ...
        [Numbered regulation excerpts]
        Question: ...
        [Answer format instructions]
    """

    def build(
        self,
        structured_query: "StructuredQuery",
        chunks: List["RegulationChunk"],
    ) -> str:
        """
        Build the complete prompt string.

        Parameters
        ----------
        structured_query : Output of ``user_input_module.process_query()``.
        chunks           : Retrieved ``RegulationChunk`` objects from the RAG module.

        Returns
        -------
        str
            A fully formed prompt ready to be sent to an LLM.
        """
        parts: List[str] = []

        # ── System preamble ──────────────────────────────────────────────────
        parts.append(_SYSTEM_PREAMBLE.strip())
        parts.append("")

        # ── Reference date ───────────────────────────────────────────────────
        ref_date = (
            structured_query.filters.valid_time.reference_date
            if structured_query.filters.valid_time.reference_date
            else "not specified (use latest applicable version)"
        )
        parts.append(f"Reference date: {ref_date}")
        parts.append("")

        # ── Regulation excerpts ───────────────────────────────────────────────
        if chunks:
            parts.append("REGULATION EXCERPTS:")
            for i, chunk in enumerate(chunks, start=1):
                eff_to = chunk.effective_to or "present"
                header = (
                    f"[{i}] {chunk.regulation} v{chunk.version} "
                    f"(active: {chunk.effective_from} – {eff_to})"
                )
                parts.append(header)
                # Truncate very long chunks to keep prompt manageable
                text = chunk.text.strip()
                if len(text) > 1500:
                    text = text[:1497] + "..."
                parts.append(text)
                parts.append("")
        else:
            parts.append("REGULATION EXCERPTS: (none retrieved — answer based on general knowledge is not permitted)")
            parts.append("")

        # ── Question ─────────────────────────────────────────────────────────
        question = structured_query.semantic_query or structured_query.raw_query.text
        parts.append(f"QUESTION: {question}")
        parts.append("")

        # ── Answer format ─────────────────────────────────────────────────────
        parts.append(_ANSWER_FORMAT.strip())

        return "\n".join(parts)

    def token_estimate(self, prompt: str) -> int:
        """Rough token estimate (1 token ≈ 4 characters for English text)."""
        return max(1, len(prompt) // 4)
