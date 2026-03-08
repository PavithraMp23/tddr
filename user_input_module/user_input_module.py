"""
user_input_module.py
--------------------
Orchestrator -- Public API of the User Input Module

Chains the internal layers in sequence:

    Layer A (QueryInterface)
        |  RawQuery
    Layer B (SemanticParser)
        |  IntermediateRepresentation
    Layer B.5 (EntityNormalizer)
        |  NormalizedEntity (canonical_id, canonical_ids, entity_type, act_token)
    Layer C (TemporalConstraintGenerator)
        |  TemporalConstraint
    ----------------------------------
    StructuredQuery  ->  downstream RAG

Public function:

    process_query(raw_text: str) -> StructuredQuery
"""

from __future__ import annotations

import logging

from user_input_module.query_interface import validate_and_preprocess
from user_input_module.semantic_parser import parse
from user_input_module.entity_normalizer import normalize
from user_input_module.temporal_constraint_gen import generate
from user_input_module.models import (
    QueryFilters,
    StructuredQuery,
    ValidTimeFilter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic query builder
# ---------------------------------------------------------------------------

def _build_semantic_query(entity: str | None, intent: str) -> str:
    """
    Produce a clean, retrieval-friendly reformulation of the query.

    Example:
        entity="Section 21 IPC", intent="punishment_inquiry"
        -> "punishment under Section 21 IPC"
    """
    _INTENT_PHRASE: dict[str, str] = {
        "punishment_inquiry": "punishment under",
        "definition_request": "definition of",
        "amendment_inquiry": "amendments to",
        "validity_request": "validity of",
        "general": "information on",
    }
    prefix = _INTENT_PHRASE.get(intent, "information on")
    if entity:
        return f"{prefix} {entity}"
    return prefix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_query(raw_text: str) -> StructuredQuery:
    """
    Transform a raw natural-language legal query into a structured,
    retrieval-ready :class:`StructuredQuery`.

    Pipeline
    --------
    1. Layer A -- validate & pre-process the text.
    2. Layer B -- extract entities (all sections), intent, temporal expr.
    3. Layer B.5 -- produce canonical entity IDs for every detected section.
    4. Layer C -- convert the temporal expression into a SQL constraint.
    5. Compose the final :class:`StructuredQuery`.

    """
    logger.info("process_query called with: %r", raw_text)

    # Layer A -- validate & pre-process
    raw_query = validate_and_preprocess(raw_text)

    # Layer B -- extract entity, intent, and temporal expression
    ir = parse(raw_query)

    # Layer B.5 -- canonical entity normalisation (all sections)
    normalized = normalize(ir)

    # Layer C -- convert the temporal expression into a SQL constraint
    constraint = generate(ir)

    # Assemble filters
    valid_time = ValidTimeFilter(
        operator=constraint.operator,
        reference_date=constraint.reference_date,
        end_date=constraint.end_date,
        sql_fragment=constraint.sql_fragment,
    )
    filters = QueryFilters(
        section_id=ir.section_id,
        section_ids=ir.section_ids,
        act_name=ir.act_name,
        canonical_entity_id=normalized.canonical_id,
        canonical_entity_ids=normalized.canonical_ids,
        valid_time=valid_time,
    )

    # Build semantic query string
    semantic_query = _build_semantic_query(ir.entity, ir.intent)

    result = StructuredQuery(
        semantic_query=semantic_query,
        filters=filters,
        raw_query=raw_query,
        intermediate=ir,
        canonical_entity_id=normalized.canonical_id,
        entity_type=normalized.entity_type,
    )

    logger.info("process_query result: %s", result.to_json())
    return result


# ---------------------------------------------------------------------------
# CLI smoke-test (python -m user_input_module.user_input_module)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.WARNING)

    _DEMO_QUERIES = [
        "What was the punishment under Section 21 before 2018?",
        "Define Section 375 IPC as of June 2015.",
        "Was Section 124A IPC valid between 2010 and 2020?",
        "What is the current punishment under Section 302 IPC?",
        "How was Section 66A IT Act amended after 2015?",
        "What does Section 21A say?",          # no temporal expression
        "Show me the latest version of Section 498A IPC.",
        "Compare Section 21 and Section 22 IPC before 2018.",  # multi-section
    ]

    separator = "-" * 70
    for query in _DEMO_QUERIES:
        print(separator)
        print(f"  INPUT : {query}")
        try:
            sq = process_query(query)
            output = {
                "semantic_query": sq.semantic_query,
                "canonical_entity_id": sq.canonical_entity_id,
                "filters": sq.filters.to_dict(),
            }
            print(f"  OUTPUT: {json.dumps(output, indent=4)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR : {exc}")
    print(separator)
