"""
Layer B -- Semantic & Structural Parser

Responsibilities:
  - Extract target legal entity  (section numbers, act name)
  - Classify query intent        (punishment inquiry, definition, etc.)
  - Detect temporal expressions  (before/after/as-of/between/current)

Output: IntermediateRepresentation dataclass
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

from user_input_module.models import RawQuery, IntermediateRepresentation

logger = logging.getLogger(__name__)


# 1. Entity Extraction

_SECTION_PATTERN = re.compile(
    r"section\s+(\d+(?:\s*[a-z](?![a-z]))?)",
    re.IGNORECASE,
)

# Known acts -- extend this list as the corpus grows.
_KNOWN_ACTS = [
    "ipc",
    "crpc",
    "cpc",
    "evidence act",
    "constitution",
    "companies act",
    "prevention of corruption act",
    "ndps act",
    "pocso act",
    "it act",
    "information technology act",
    "motor vehicles act",
    "mva",
    "income tax act",
    "customs act",
    "gst act",
    "arms act",
    "negotiable instruments act",
    "hwm",
    "howm",
    "hazardous waste management",
    "hazardous and other waste",
]


def _normalise_section_token(raw: str) -> str:
    cleaned = raw.strip().replace(" ", "")
    return re.sub(r"([a-z])$", lambda m: m.group(1).upper(), cleaned)


def _extract_sections(text: str) -> List[str]:

    matches = _SECTION_PATTERN.findall(text)
    return [_normalise_section_token(m) for m in matches]


def _extract_section(text: str) -> Optional[str]:
    sections = _extract_sections(text)
    return sections[0] if sections else None


def _extract_act(text: str) -> Optional[str]:
    lower = text.lower()
    for act in _KNOWN_ACTS:
        if re.search(rf"\b{re.escape(act)}\b", lower):
            return act.upper()
    return None


def _build_entity_string(
    section_ids: List[str],
    act_name: Optional[str],
) -> Optional[str]:

    if section_ids and act_name:
        if len(section_ids) == 1:
            return f"Section {section_ids[0]} {act_name}"
        listed = ", ".join(section_ids)
        return f"Sections {listed} {act_name}"
    if section_ids:
        if len(section_ids) == 1:
            return f"Section {section_ids[0]}"
        listed = ", ".join(section_ids)
        return f"Sections {listed}"
    if act_name:
        return act_name
    return None


# 2. Intent Classification

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "punishment_inquiry": [
        r"\bpunishment\b", r"\bpenalty\b", r"\bpenalties\b",
        r"\bsentence\b", r"\bfine\b", r"\bimprisonment\b",
        r"\bjail\b", r"\brigorous\b", r"\bdeath\b", r"\blife sentence\b",
    ],
    "definition_request": [
        r"\bdefined\b", r"\bdefinition\b", r"\bmeaning\b",
        r"\bwhat is\b", r"\bwhat are\b", r"\bexplain\b",
        r"\bdescribe\b", r"\bdefine\b",
    ],
    "amendment_inquiry": [
        r"\bamended\b", r"\bamendment\b", r"\bchange\b",
        r"\bmodification\b", r"\bsubstituted\b", r"\binserted\b",
        r"\bomitted\b", r"\brepealed\b",
    ],
    "validity_request": [
        r"\bvalid\b", r"\bapply\b", r"\bapplicable\b",
        r"\benforce\b", r"\benforced\b", r"\bin force\b",
        r"\bin effect\b", r"\boperative\b",
    ],
}


def _classify_intent(text: str) -> str:

    lower = text.lower()
    for intent, patterns in _INTENT_KEYWORDS.items():
        if any(re.search(pat, lower) for pat in patterns):
            return intent
    return "general"


# 3. Temporal Expression Detection

_TEMPORAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "between 2015 and 2019" / "between 01-01-2015 and 31-12-2019"
    ("between", re.compile(
        r"between\s+"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\w+\s+\d{4})"
        r"\s+and\s+"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\w+\s+\d{4})",
        re.IGNORECASE,
    )),
    # "as of 2015" / "as at June 2020" / "as of 01-06-2015"
    ("as_of", re.compile(
        r"(?:as\s+of|as\s+at)\s+"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )),
    # "before 2018" / "prior to 2020" / "before January 2018"
    ("before", re.compile(
        r"(?:before|prior\s+to)\s+"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )),
    # "after 2020" / "since 2015" / "after June 2019"
    ("after", re.compile(
        r"(?:after|since)\s+"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{4}|\d{4})",
        re.IGNORECASE,
    )),
    # "in 2017" / "during 2018"
    ("in_year", re.compile(
        r"(?:in|during)\s+((?:19|20)\d{2})\b",
        re.IGNORECASE,
    )),
    # "current" / "now" / "today" / "latest" / "present"
    ("current", re.compile(
        r"\b(current(?:ly)?|now|today|latest|present(?:ly)?|at\s+present)\b",
        re.IGNORECASE,
    )),
]


def _extract_temporal_expression(text: str) -> Optional[str]:
    for _label, pattern in _TEMPORAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0).strip().lower()
    return None



# Public interface

from typing import List

def _extract_version_refs(text: str) -> List[str]:
    """
    Extract specific version references like 'feb amendment' or '2019 version'.
    Normalizes 'february' -> 'feb', 'july' -> 'jul', etc.
    """
    import re
    refs = []
    if re.search(r'\b(feb|february)\b', text):
        refs.append('feb')
    if re.search(r'\b(jul|july)\b', text):
        refs.append('jul')
    if re.search(r'\b(jun|june)\b', text):
        refs.append('jun')
    
    for m in re.finditer(r'\b(19\d{2}|20\d{2})\b', text):
        refs.append(m.group(1))

    return refs

def parse(raw_query: RawQuery) -> IntermediateRepresentation:
    """
    Parse *raw_query* and return an :class:`IntermediateRepresentation`.

    Parameters
    ----------
    raw_query : RawQuery
        Pre-processed query produced by Layer A.

    Returns
    -------
    IntermediateRepresentation
        Contains extracted entities (single + multi), intent, and temporal
        expression.

    Notes
    -----
    ``section_ids`` holds **all** detected sections (publication-grade
    multi-entity support).  ``section_id`` is always ``section_ids[0]``
    (or ``None`` when nothing was found) for backward compatibility.
    """
    text = raw_query.text  # already lowercased by Layer A

    section_ids = _extract_sections(text)
    section_id = section_ids[0] if section_ids else None
    act_name = _extract_act(text)
    entity = _build_entity_string(section_ids, act_name)
    intent = _classify_intent(text)
    temporal_expression = _extract_temporal_expression(text)
    version_refs = _extract_version_refs(text)

    result = IntermediateRepresentation(
        entity=entity,
        section_id=section_id,
        section_ids=section_ids,
        act_name=act_name,
        intent=intent,
        temporal_expression=temporal_expression,
        version_refs=version_refs,
    )

    logger.debug("SemanticParser output: %s", result.to_dict())
    return result
