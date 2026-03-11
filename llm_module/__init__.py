"""
llm_module
----------
LLM Module (Module 3) of the TDDR pipeline.

Converts retrieved regulation chunks + a structured query into a grounded,
cited, human-readable answer via a configurable LLM backend.

Pipeline position
-----------------
    StructuredQuery (Module 1)
           │
    RegulationChunks (Module 2)
           │
    PromptBuilder  →  LLMInterface  →  AnswerGenerator
           │
    LLMResponse

Public API
----------
    from llm_module import generate_answer, LLMResponse

    response = generate_answer(
        structured_query = sq,           # from user_input_module
        chunks           = chunks,        # from rag_module
        backend          = "mock",        # "mock" | "ollama" | "openai" | "huggingface"
        model            = None,          # uses backend default if None
        api_key          = None,          # required for openai / huggingface
    )
    print(response.answer)
    print(response.cited_sections)
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from llm_module.llm_models import LLMResponse
from llm_module.prompt_builder import PromptBuilder
from llm_module.llm_interface import LLMInterface
from llm_module.answer_generator import AnswerGenerator

if TYPE_CHECKING:
    from user_input_module.models import StructuredQuery
    from rag_module.models import RegulationChunk

# Module-level singletons (re-used across calls for efficiency)
_builder   = PromptBuilder()
_interface = LLMInterface()
_generator = AnswerGenerator()


def generate_answer(
    structured_query: "StructuredQuery",
    chunks: List["RegulationChunk"],
    backend: str = "mock",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> LLMResponse:
    """
    End-to-end Module 3 pipeline.

    Parameters
    ----------
    structured_query : ``StructuredQuery`` produced by Module 1.
    chunks           : ``RegulationChunk`` list produced by Module 2.
    backend          : LLM backend — ``"mock"``, ``"ollama"``, ``"openai"``,
                       or ``"huggingface"``.
    model            : Model name/ID (backend-specific; uses default if None).
    api_key          : API key for ``openai`` or ``huggingface`` backends.
    timeout          : HTTP timeout in seconds (ignored for ``mock``).

    Returns
    -------
    LLMResponse
        Structured answer with citation, explanation, and confidence score.
    """
    # Step 1 – Build prompt
    prompt = _builder.build(structured_query, chunks)
    prompt_tokens = _builder.token_estimate(prompt)

    # Step 2 – Call LLM
    from llm_module.llm_interface import _DEFAULT_MODELS
    resolved_model = model or _DEFAULT_MODELS.get(backend.lower(), "unknown")
    model_used = f"{backend}/{resolved_model}"

    raw_text = _interface.generate(
        prompt,
        backend=backend,
        model=resolved_model,
        api_key=api_key,
        timeout=timeout,
    )

    # Step 3 – Parse into structured response
    return _generator.parse(
        raw_text,
        chunks,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
    )


__all__ = [
    "generate_answer",
    "LLMResponse",
    "PromptBuilder",
    "LLMInterface",
    "AnswerGenerator",
]
