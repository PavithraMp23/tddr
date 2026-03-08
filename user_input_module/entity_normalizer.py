"""

Layer B.5 -- Canonical Entity Resolver

Sits between Layer B (SemanticParser) and Layer C (TemporalConstraintGenerator).

Responsibility:
  Take the raw (section_ids, section_id, act_name) extracted by Layer B and
  produce stable, collision-free canonical identifiers for use in downstream
  retrieval, drift detection, and logging.

Canonical ID format:

    <ACT_TOKEN>::<ENTITY_TYPE>::<IDENTIFIER>

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from user_input_module.models import IntermediateRepresentation

logger = logging.getLogger(__name__)



# Act alias normalisation table

_ACT_ALIASES: dict[str, str] = {
    # Indian Penal Code
    "ipc": "IPC",
    "indian penal code": "IPC",

    # Code of Criminal Procedure
    "crpc": "CRPC",
    "code of criminal procedure": "CRPC",

    # Code of Civil Procedure
    "cpc": "CPC",
    "code of civil procedure": "CPC",

    # Indian Evidence Act
    "evidence act": "EVIDENCE_ACT",
    "indian evidence act": "EVIDENCE_ACT",

    # Constitution of India
    "constitution": "CONSTITUTION",
    "constitution of india": "CONSTITUTION",

    # Companies Act
    "companies act": "COMPANIES_ACT",

    # Prevention of Corruption Act
    "prevention of corruption act": "POCA",
    "poca": "POCA",

    # NDPS Act
    "ndps act": "NDPS_ACT",
    "ndps": "NDPS_ACT",

    # POCSO Act
    "pocso act": "POCSO_ACT",
    "pocso": "POCSO_ACT",

    # Information Technology Act
    "it act": "IT_ACT",
    "information technology act": "IT_ACT",

    # Motor Vehicles Act
    "motor vehicles act": "MVA",
    "mva": "MVA",

    # Income Tax Act
    "income tax act": "ITA",
    "ita": "ITA",

    # Customs Act
    "customs act": "CUSTOMS_ACT",

    # GST Act
    "gst act": "GST_ACT",
    "gst": "GST_ACT",

    # Arms Act
    "arms act": "ARMS_ACT",

    # Negotiable Instruments Act
    "negotiable instruments act": "NIA",
    "nia": "NIA",
}

# Sentinel used when no act token is available.
_UNKNOWN_ACT = "UNKNOWN_ACT"


# Public output dataclass

@dataclass
class NormalizedEntity:
    canonical_id: Optional[str]
    canonical_ids: List[str] = field(default_factory=list)
    entity_type: Optional[str] = None
    act_token: Optional[str] = None


# Internal helpers


def _normalize_act(act_name: Optional[str]) -> Optional[str]:
    """
    Map a raw act name (as returned by Layer B) to its canonical token.

    Returns ``None`` if *act_name* is ``None`` or unrecognised.
    """
    if not act_name:
        return None
    return _ACT_ALIASES.get(act_name.lower().strip())


def _build_canonical_id(
    act_token: Optional[str],
    entity_type: str,
    identifier: str,
) -> str:

    act_part = act_token if act_token else _UNKNOWN_ACT
    return f"{act_part}::{entity_type}::{identifier}"


# Public interface

def normalize(ir: IntermediateRepresentation) -> NormalizedEntity:
 
    act_token = _normalize_act(ir.act_name)

    # Use section_ids list; fall back to [section_id] for compat with
    # callers that pass an IR built without section_ids.
    section_ids: List[str] = ir.section_ids if ir.section_ids else (
        [ir.section_id] if ir.section_id else []
    )

    # --- Case 1: one or more sections detected ----------------------------
    if section_ids:
        canonical_ids = [
            _build_canonical_id(
                act_token=act_token,
                entity_type="SECTION",
                identifier=sid.upper(),
            )
            for sid in section_ids
        ]
        primary = canonical_ids[0]
        result = NormalizedEntity(
            canonical_id=primary,
            canonical_ids=canonical_ids,
            entity_type="section",
            act_token=act_token,
        )
        logger.debug(
            "EntityNormalizer: sections=%r, act=%r -> %r",
            section_ids, ir.act_name, canonical_ids,
        )
        return result

    # --- Case 2: act only (no section) ------------------------------------
    if ir.act_name:
        # Fallback to raw act name uppercased if alias not found
        effective_token = act_token if act_token else ir.act_name.upper().replace(" ", "_")
        canonical_id = _build_canonical_id(
            act_token=effective_token,
            entity_type="ACT",
            identifier="ROOT",
        )
        result = NormalizedEntity(
            canonical_id=canonical_id,
            canonical_ids=[canonical_id],
            entity_type="act",
            act_token=effective_token,
        )
        logger.debug(
            "EntityNormalizer: no section, act=%r -> %r",
            ir.act_name, canonical_id,
        )
        return result

    # --- Case 3: nothing extracted ----------------------------------------
    logger.debug("EntityNormalizer: no entity found -> None")
    return NormalizedEntity(
        canonical_id=None,
        canonical_ids=[],
        entity_type=None,
        act_token=None,
    )
