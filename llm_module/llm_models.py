"""
llm_module/llm_models.py
------------------------
Data models for the LLM Module (Module 3) of the TDDR pipeline.

    LLMResponse – structured output produced by the answer_generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class LLMResponse:
    """
    Structured output from the LLM Module.

    Parameters
    ----------
    answer          : Main grounded answer, directly addressing the user's query.
    cited_sections  : List of regulation section IDs cited in the answer
                      (e.g. ["Section 302", "Section 375"]).
    explanation     : Short reasoning or elaboration that grounds the answer
                      in the retrieved regulations.
    confidence      : Confidence level: "high", "medium", or "low".
                      Based on whether cited sections were found in retrieved chunks.
    model_used      : Human-readable identifier for the LLM backend used
                      (e.g. "mock", "ollama/mistral", "openai/gpt-3.5-turbo").
    prompt_tokens   : Approximate number of characters in the prompt
                      (used for transparency / debugging).
    """

    answer: str
    cited_sections: List[str]
    explanation: str
    confidence: str        # "high" | "medium" | "low"
    model_used: str
    prompt_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "cited_sections": self.cited_sections,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "prompt_tokens": self.prompt_tokens,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return (
            f"LLMResponse(confidence={self.confidence!r}, "
            f"cited={self.cited_sections}, "
            f"model={self.model_used!r})"
        )
