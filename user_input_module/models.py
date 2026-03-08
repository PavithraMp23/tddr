"""

Typed data models for every interchange object in the User Input Module.


    Layer A  (QueryInterface)
    Layer B  (SemanticParser)
    Layer B.5 (EntityNormalizer) 
    Layer C  (TemporalConstraintGenerator)

"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# Layer A output


@dataclass
class RawQuery:

    text: str
    original_text: str
    language: str = "en"

    def to_dict(self) -> dict:
        return asdict(self)


# Layer B output
@dataclass
class IntermediateRepresentation:

    entity: Optional[str] = None
    section_id: Optional[str] = None
    section_ids: List[str] = field(default_factory=list)
    act_name: Optional[str] = None
    intent: str = "general"
    temporal_expression: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Layer C output

@dataclass
class TemporalConstraint:
   
    operator: str = "current"
    reference_date: Optional[str] = None
    end_date: Optional[str] = None
    sql_fragment: str = "valid_to IS NULL"

    def to_dict(self) -> dict:
        return asdict(self)


# Final module output


@dataclass
class ValidTimeFilter:

    operator: str
    reference_date: Optional[str] = None
    end_date: Optional[str] = None
    sql_fragment: str = "valid_to IS NULL"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryFilters:
    
    section_id: Optional[str] = None
    section_ids: List[str] = field(default_factory=list)
    act_name: Optional[str] = None
    canonical_entity_id: Optional[str] = None
    canonical_entity_ids: List[str] = field(default_factory=list)
    valid_time: ValidTimeFilter = field(
        default_factory=lambda: ValidTimeFilter(operator="current")
    )

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "section_ids": self.section_ids,
            "act_name": self.act_name,
            "canonical_entity_id": self.canonical_entity_id,
            "canonical_entity_ids": self.canonical_entity_ids,
            "valid_time": self.valid_time.to_dict(),
        }


@dataclass
class StructuredQuery:
  
    semantic_query: str
    filters: QueryFilters
    raw_query: RawQuery
    intermediate: IntermediateRepresentation
    canonical_entity_id: Optional[str] = None
    entity_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "semantic_query": self.semantic_query,
            "canonical_entity_id": self.canonical_entity_id,
            "entity_type": self.entity_type,
            "filters": self.filters.to_dict(),
            "raw_query": self.raw_query.to_dict(),
            "intermediate": self.intermediate.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
