"""
output_module
-------------
Output Module (Module 4) of the TDDR pipeline.

Receives the ``LLMResponse`` from Module 3 and the ``RegulationChunk`` list
from Module 2, then packages them into a traceable ``FinalSystemResponse``
with resolved legal evidence, source excerpts, and operational metadata.

Pipeline position
-----------------
    StructuredQuery  (Module 1)
           │
    RegulationChunks (Module 2)
           │
    LLMResponse      (Module 3)
           │
    CitationResolver → ResponseBuilder
           │
    FinalSystemResponse

Public API
----------
::

    from output_module import build_response, FinalSystemResponse

    final = build_response(llm_response, chunks)
    print(final.to_json())

"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from output_module.response_models import LegalEvidence, FinalSystemResponse
from output_module.citation_resolver import CitationResolver
from output_module.response_builder import ResponseBuilder

if TYPE_CHECKING:
    from llm_module.llm_models import LLMResponse
    from rag_module.models import RegulationChunk

# Module-level singleton (reused across calls)
_builder = ResponseBuilder()


def build_response(
    llm_response: "LLMResponse",
    chunks: List["RegulationChunk"],
) -> FinalSystemResponse:
    """
    End-to-end Module 4 pipeline.

    Parameters
    ----------
    llm_response : ``LLMResponse`` produced by Module 3.
    chunks       : ``RegulationChunk`` list produced by Module 2.

    Returns
    -------
    FinalSystemResponse
        Typed, JSON-serialisable final output — ready for API / UI /
        report generator consumption.
    """
    return _builder.build(llm_response, chunks)


__all__ = [
    "build_response",
    "FinalSystemResponse",
    "LegalEvidence",
    "CitationResolver",
    "ResponseBuilder",
]
