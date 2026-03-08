"""
user_input_module
-----------------
Query Interpretation & Temporal Constraint Extraction Layer for the
Temporal Regulation Drift Detector (TDDR) pipeline.

Public API
----------
    from user_input_module import process_query
    result = process_query("What was the punishment under Section 21 before 2018?")
    print(result.to_json())
    # result.canonical_entity_id  →  "IPC::SECTION::21"
"""

from user_input_module.user_input_module import process_query
from user_input_module.entity_normalizer import normalize, NormalizedEntity
from user_input_module.models import (
    RawQuery,
    IntermediateRepresentation,
    TemporalConstraint,
    StructuredQuery,
    QueryFilters,
    ValidTimeFilter,
)

__all__ = [
    "process_query",
    "normalize",
    "NormalizedEntity",
    "RawQuery",
    "IntermediateRepresentation",
    "TemporalConstraint",
    "StructuredQuery",
    "QueryFilters",
    "ValidTimeFilter",
]
