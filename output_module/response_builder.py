"""
output_module/response_builder.py
-----------------------------------
Main orchestrator of the Output Module (Module 4).

Takes a ``LLMResponse`` from Module 3 and the ``RegulationChunk`` list from
Module 2, resolves citations to concrete legal evidence, and assembles the
final ``FinalSystemResponse``.

Pipeline inside this module
---------------------------
    LLMResponse
         │
         ▼
    CitationResolver.resolve()
         │
         ▼
    Deduplicated LegalEvidence[]
         │
         ▼
    FinalSystemResponse  ← metadata attached here
"""

from __future__ import annotations

import datetime
from typing import List, TYPE_CHECKING

from output_module.citation_resolver import CitationResolver
from output_module.response_models import FinalSystemResponse, LegalEvidence

if TYPE_CHECKING:
    from llm_module.llm_models import LLMResponse
    from rag_module.models import RegulationChunk

_resolver = CitationResolver()


class ResponseBuilder:
    """
    Assembles a ``FinalSystemResponse`` from Module-3 and Module-2 outputs.

    Usage
    -----
    ::

        from output_module.response_builder import ResponseBuilder

        builder  = ResponseBuilder()
        final    = builder.build(llm_response, chunks)
        api_json = final.to_json()
    """

    def build(
        self,
        llm_response: "LLMResponse",
        chunks: List["RegulationChunk"],
    ) -> FinalSystemResponse:
        """
        Build a ``FinalSystemResponse`` from LLM output and retrieved chunks.

        Parameters
        ----------
        llm_response : ``LLMResponse`` produced by Module 3.
        chunks       : ``RegulationChunk`` list produced by Module 2.

        Returns
        -------
        FinalSystemResponse
            Typed, JSON-serialisable final output of the pipeline.
        """
        # Step 1 — Resolve citations to legal evidence
        legal_basis: List[LegalEvidence] = _resolver.resolve(
            cited_sections=llm_response.cited_sections,
            chunks=chunks,
        )

        # Step 2 — Attach metadata
        metadata = {
            "model_used":       llm_response.model_used,
            "retrieved_chunks": len(chunks),
            "cited_sections":   llm_response.cited_sections,
            "prompt_tokens":    llm_response.prompt_tokens,
            "generated_at":     datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Step 3 — Assemble
        return FinalSystemResponse(
            answer      = llm_response.answer,
            confidence  = llm_response.confidence,
            explanation = llm_response.explanation,
            legal_basis = legal_basis,
            metadata    = metadata,
        )
