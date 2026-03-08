"""
tests/test_user_input_module.py
--------------------------------
Comprehensive pytest suite for the User Input Module.

Covers:
  - Layer A: validation, preprocessing, QueryValidationError
  - Layer B: section extraction, act detection, intent classification,
             temporal expression detection
  - Layer C: all temporal operators (before/after/as_of/between/in_year/current)
  - Orchestrator: end-to-end process_query() integration
  - Edge cases: no temporal expression, conflicting/ambiguous phrases,
                implicit current-time queries
"""

import pytest
from user_input_module.query_interface import validate_and_preprocess, QueryValidationError
from user_input_module.semantic_parser import parse
from user_input_module.temporal_constraint_gen import generate
from user_input_module.user_input_module import process_query
from user_input_module.models import RawQuery, IntermediateRepresentation


# ===========================================================================
# Helpers
# ===========================================================================

def _make_raw(text: str) -> RawQuery:
    """Shortcut: create a RawQuery without going through Layer A validation."""
    return RawQuery(text=text.lower(), original_text=text)


def _make_ir(temporal_expression=None) -> IntermediateRepresentation:
    """Shortcut: create a minimal IntermediateRepresentation for Layer C tests."""
    return IntermediateRepresentation(temporal_expression=temporal_expression)


# ===========================================================================
# Layer A — Query Interface Tests
# ===========================================================================

class TestQueryInterface:

    def test_valid_query_returns_raw_query(self):
        rq = validate_and_preprocess("What is Section 21?")
        assert rq.text == "what is section 21?"
        assert rq.original_text == "What is Section 21?"

    def test_lowercasing(self):
        rq = validate_and_preprocess("SECTION 21 IPC PUNISHMENT")
        assert rq.text == "section 21 ipc punishment"

    def test_whitespace_collapse(self):
        rq = validate_and_preprocess("Section   21   IPC")
        assert "  " not in rq.text

    def test_strips_leading_trailing_whitespace(self):
        rq = validate_and_preprocess("   Section 21 IPC   ")
        assert not rq.text.startswith(" ")
        assert not rq.text.endswith(" ")

    def test_empty_string_raises(self):
        with pytest.raises(QueryValidationError, match="empty"):
            validate_and_preprocess("")

    def test_whitespace_only_raises(self):
        with pytest.raises(QueryValidationError, match="empty"):
            validate_and_preprocess("   \t\n  ")

    def test_too_short_raises(self):
        with pytest.raises(QueryValidationError, match="too short"):
            validate_and_preprocess("AB")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_and_preprocess(12345)  # type: ignore[arg-type]

    def test_language_field_present(self):
        rq = validate_and_preprocess("What is Section 21?")
        assert isinstance(rq.language, str)
        assert len(rq.language) >= 2   # at least "en"

    def test_very_long_query_raises(self):
        long_text = "a " * 600  # >1000 chars
        with pytest.raises(QueryValidationError, match="maximum"):
            validate_and_preprocess(long_text)


# ===========================================================================
# Layer B — Semantic Parser Tests
# ===========================================================================

class TestSemanticParser:

    # --- Entity Extraction ---

    def test_section_number_extracted(self):
        ir = parse(_make_raw("punishment under section 21"))
        assert ir.section_id == "21"

    def test_section_with_letter_suffix(self):
        ir = parse(_make_raw("what does section 21a say?"))
        assert ir.section_id == "21A"

    def test_section_three_digits(self):
        ir = parse(_make_raw("define section 302 ipc"))
        assert ir.section_id == "302"

    def test_act_ipc_detected(self):
        ir = parse(_make_raw("section 302 ipc punishment"))
        assert ir.act_name == "IPC"

    def test_act_crpc_detected(self):
        ir = parse(_make_raw("section 161 crpc statement"))
        assert ir.act_name == "CRPC"

    def test_entity_combined(self):
        ir = parse(_make_raw("punishment under section 21 ipc before 2018"))
        assert ir.entity == "Section 21 IPC"

    def test_no_section_returns_none(self):
        ir = parse(_make_raw("what is the punishment for theft?"))
        assert ir.section_id is None

    # --- Intent Classification ---

    def test_punishment_intent(self):
        ir = parse(_make_raw("what is the punishment under section 302?"))
        assert ir.intent == "punishment_inquiry"

    def test_definition_intent(self):
        ir = parse(_make_raw("define theft according to law"))
        assert ir.intent == "definition_request"

    def test_amendment_intent(self):
        ir = parse(_make_raw("was section 66a amended in 2015?"))
        assert ir.intent == "amendment_inquiry"

    def test_validity_intent(self):
        ir = parse(_make_raw("is section 124a still valid?"))
        assert ir.intent == "validity_request"

    def test_general_intent_default(self):
        ir = parse(_make_raw("show me section 21"))
        assert ir.intent == "general"

    # --- Temporal Expression Detection ---

    def test_before_expression(self):
        ir = parse(_make_raw("punishment under section 21 before 2018"))
        assert ir.temporal_expression is not None
        assert "before" in ir.temporal_expression

    def test_after_expression(self):
        ir = parse(_make_raw("section 66a after 2015"))
        assert ir.temporal_expression is not None
        assert "after" in ir.temporal_expression

    def test_as_of_expression(self):
        ir = parse(_make_raw("section 375 as of june 2015"))
        assert ir.temporal_expression is not None
        assert "as of" in ir.temporal_expression

    def test_between_expression(self):
        ir = parse(_make_raw("section 124a between 2010 and 2020"))
        assert ir.temporal_expression is not None
        assert "between" in ir.temporal_expression

    def test_current_expression(self):
        ir = parse(_make_raw("what is the current punishment under section 302?"))
        assert ir.temporal_expression is not None
        assert "current" in ir.temporal_expression

    def test_no_temporal_expression(self):
        ir = parse(_make_raw("what does section 21 say?"))
        assert ir.temporal_expression is None


# ===========================================================================
# Layer C — Temporal Constraint Generator Tests
# ===========================================================================

class TestTemporalConstraintGen:

    def test_before_year(self):
        ir = _make_ir("before 2018")
        c = generate(ir)
        assert c.operator == "before"
        assert c.reference_date == "2018-01-01"
        assert "valid_from < '2018-01-01'" in c.sql_fragment
        assert "valid_to" in c.sql_fragment

    def test_after_year(self):
        ir = _make_ir("after 2020")
        c = generate(ir)
        assert c.operator == "after"
        assert c.reference_date == "2020-01-01"
        assert "valid_from >= '2020-01-01'" in c.sql_fragment

    def test_as_of_year(self):
        ir = _make_ir("as of 2015")
        c = generate(ir)
        assert c.operator == "as_of"
        assert c.reference_date == "2015-01-01"
        assert "valid_from <= '2015-01-01'" in c.sql_fragment

    def test_as_of_month_year(self):
        ir = _make_ir("as of june 2015")
        c = generate(ir)
        assert c.operator == "as_of"
        assert c.reference_date == "2015-06-01"

    def test_between_years(self):
        ir = _make_ir("between 2015 and 2019")
        c = generate(ir)
        assert c.operator == "between"
        assert c.reference_date == "2015-01-01"
        assert c.end_date == "2019-12-31"
        assert "valid_from >= '2015-01-01'" in c.sql_fragment
        assert "valid_to" in c.sql_fragment

    def test_in_year(self):
        ir = _make_ir("in 2017")
        c = generate(ir)
        assert c.operator == "in_year"
        assert c.reference_date == "2017-01-01"
        assert c.end_date == "2017-12-31"

    def test_current_keyword(self):
        ir = _make_ir("current")
        c = generate(ir)
        assert c.operator == "current"
        assert c.sql_fragment == "valid_to IS NULL"

    def test_latest_keyword(self):
        ir = _make_ir("latest")
        c = generate(ir)
        assert c.operator == "current"
        assert c.sql_fragment == "valid_to IS NULL"

    def test_no_temporal_expression_defaults_to_current(self):
        ir = _make_ir(None)
        c = generate(ir)
        assert c.operator == "current"
        assert c.sql_fragment == "valid_to IS NULL"

    def test_unrecognised_expression_defaults_to_current(self):
        ir = _make_ir("some ancient time")
        c = generate(ir)
        assert c.operator == "current"

    def test_before_month_year(self):
        ir = _make_ir("before january 2018")
        c = generate(ir)
        assert c.operator == "before"
        assert c.reference_date == "2018-01-01"

    def test_prior_to(self):
        ir = _make_ir("prior to 2020")
        c = generate(ir)
        assert c.operator == "before"
        assert c.reference_date == "2020-01-01"

    def test_since(self):
        ir = _make_ir("since 2015")
        c = generate(ir)
        assert c.operator == "after"
        assert c.reference_date == "2015-01-01"


# ===========================================================================
# Orchestrator — End-to-End Integration Tests
# ===========================================================================

class TestProcessQuery:

    def test_full_before_query(self):
        sq = process_query("What was the punishment under Section 21 before 2018?")
        assert sq.semantic_query
        assert sq.filters.section_id == "21"
        assert sq.filters.valid_time.operator == "before"
        assert sq.filters.valid_time.reference_date == "2018-01-01"
        assert "valid_from < '2018-01-01'" in sq.filters.valid_time.sql_fragment

    def test_full_as_of_query(self):
        sq = process_query("Define Section 375 IPC as of June 2015.")
        assert sq.filters.section_id == "375"
        assert sq.filters.act_name == "IPC"
        assert sq.filters.valid_time.operator == "as_of"
        assert sq.filters.valid_time.reference_date == "2015-06-01"

    def test_full_between_query(self):
        sq = process_query("Was Section 124A IPC valid between 2010 and 2020?")
        assert sq.filters.valid_time.operator == "between"
        assert sq.filters.valid_time.reference_date == "2010-01-01"
        assert sq.filters.valid_time.end_date == "2020-12-31"

    def test_full_current_query(self):
        sq = process_query("What is the current punishment under Section 302 IPC?")
        assert sq.filters.valid_time.operator == "current"
        assert sq.filters.valid_time.sql_fragment == "valid_to IS NULL"

    def test_full_after_query(self):
        sq = process_query("How was Section 66A IT Act amended after 2015?")
        assert sq.filters.valid_time.operator == "after"
        assert sq.filters.valid_time.reference_date == "2015-01-01"

    def test_no_temporal_defaults_to_current(self):
        sq = process_query("What does Section 21A say?")
        assert sq.filters.valid_time.operator == "current"
        assert sq.filters.valid_time.sql_fragment == "valid_to IS NULL"

    def test_latest_defaults_to_current(self):
        sq = process_query("Show me the latest version of Section 498A IPC.")
        assert sq.filters.valid_time.operator == "current"

    def test_semantic_query_punishment(self):
        sq = process_query("What was the punishment under Section 21 before 2018?")
        assert "punishment" in sq.semantic_query.lower()

    def test_semantic_query_definition(self):
        sq = process_query("Define Section 375 IPC as of June 2015.")
        assert "definition" in sq.semantic_query.lower()

    def test_to_json_serialisable(self):
        import json
        sq = process_query("What is the current punishment under Section 302 IPC?")
        dumped = json.loads(sq.to_json())
        assert "semantic_query" in dumped
        assert "filters" in dumped
        assert "valid_time" in dumped["filters"]

    def test_empty_input_raises(self):
        with pytest.raises(Exception):
            process_query("")

    def test_non_string_input_raises(self):
        with pytest.raises(TypeError):
            process_query(None)  # type: ignore[arg-type]
