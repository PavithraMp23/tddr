"""
Layer A — Query Interface Layer

Responsibilities:
  - Input validation (non-empty, within length limits)
  - Basic preprocessing: strip, whitespace collapse, lowercasing
  - Optional language detection (via langdetect; falls back to "en")

Output: RawQuery dataclass
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from user_input_module.models import RawQuery

logger = logging.getLogger(__name__)


MAX_QUERY_LENGTH = 1_000   # characters
MIN_QUERY_LENGTH = 3       # characters

""" Helper function to clean the input data """

def _detect_language(text: str) -> str:
    try:
        from langdetect import detect 
        code = detect(text)
        return str(code)
    except ImportError:
        logger.debug("langdetect not installed - defaulting language to 'en'.")
        return "en"
    except Exception as exc:  
        logger.debug("Language detection failed (%s) - defaulting to 'en'.", exc)
        return "en"



def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _preprocess(text: str) -> str:

    text = _collapse_whitespace(text)
    return text.lower()


# Public interface

class QueryValidationError(ValueError):
    """Raised when the raw input fails validation checks."""


def validate_and_preprocess(raw_text: str) -> RawQuery:
    
    if not isinstance(raw_text, str):
        raise TypeError(
            f"Query must be a string, got {type(raw_text).__name__!r}."
        )

    original = raw_text  # preserve original before any mutation

    stripped = raw_text.strip()
    if not stripped:
        raise QueryValidationError(
            "Query must not be empty or whitespace-only."
        )

    if len(stripped) < MIN_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query is too short (minimum {MIN_QUERY_LENGTH} characters)."
        )

    if len(stripped) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"Query exceeds the maximum allowed length of {MAX_QUERY_LENGTH} characters "
            f"(got {len(stripped)})."
        )

    cleaned = _preprocess(stripped)
    language = _detect_language(cleaned)

    logger.debug(
        "QueryInterface: original=%r, cleaned=%r, language=%s",
        original,
        cleaned,
        language,
    )

    return RawQuery(
        text=cleaned,
        original_text=original,
        language=language,
    )
