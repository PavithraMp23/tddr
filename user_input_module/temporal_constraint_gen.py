"""
Layer C — Temporal Constraint Generator

Responsibility:
  Convert a linguistic temporal expression (from Layer B) into a formal
  database-ready TemporalConstraint carrying:
    - operator       : semantic label
    - reference_date : ISO-8601 string (primary anchor)
    - end_date       : ISO-8601 string (for interval queries)
    - sql_fragment   : ready-to-paste SQL WHERE clause fragment

"""

from __future__ import annotations

import re
import logging
from datetime import date, datetime
from typing import Optional

from user_input_module.models import IntermediateRepresentation, TemporalConstraint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Month name → number mapping
# ---------------------------------------------------------------------------

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "may": 5,
    "june": 6,     "jun": 6,
    "july": 7,     "jul": 7,
    "august": 8,   "aug": 8,
    "september": 9,"sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11,"nov": 11,
    "december": 12,"dec": 12,
}


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

def _year_start(year: int) -> str:
    """Return 'YYYY-01-01'."""
    return f"{year:04d}-01-01"


def _year_end(year: int) -> str:
    """Return 'YYYY-12-31'."""
    return f"{year:04d}-12-31"


def _parse_date_token(token: str, use_end: bool = False) -> Optional[str]:
    """
    Attempt to parse a date token into an ISO-8601 string.

    Handles:
      - Bare 4-digit year:            "2018"          → "2018-01-01" or "2018-12-31"
      - Month name + year:            "january 2018"   → "2018-01-01"
      - DD/MM/YYYY or DD-MM-YYYY:     "01/06/2015"     → "2015-06-01"
      - YYYY-MM-DD / YYYY/MM/DD:      "2015-06-01"     → "2015-06-01"
      - ISO-8601 string already:      "2015-06-01"     → "2015-06-01"

    Parameters
    ----------
    token : str
        Normalised (lowercase, stripped) date fragment.
    use_end : bool
        When True and only a year is given, resolve to Dec 31 instead of Jan 1.

    Returns
    -------
    str | None
        ISO-8601 date string, or None if parsing fails.
    """
    token = token.strip().lower()

    # Already ISO-8601: YYYY-MM-DD
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", token)
    if iso_match:
        return token  # already well-formed

    # YYYY/MM/DD
    ymds_match = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})", token)
    if ymds_match:
        y, m, d = ymds_match.groups()
        return f"{y}-{m}-{d}"

    # DD/MM/YYYY or DD-MM-YYYY
    dmy_match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", token)
    if dmy_match:
        d_s, m_s, y_s = dmy_match.groups()
        return f"{int(y_s):04d}-{int(m_s):02d}-{int(d_s):02d}"

    # Bare 4-digit year
    year_match = re.fullmatch(r"((?:19|20)\d{2})", token)
    if year_match:
        year = int(year_match.group(1))
        return _year_end(year) if use_end else _year_start(year)

    # "month year" e.g. "january 2018", "jan 2018"
    month_year_match = re.fullmatch(
        r"([a-z]+)\s+(\d{4})", token
    )
    if month_year_match:
        month_name, year_s = month_year_match.groups()
        month_num = _MONTH_NAMES.get(month_name)
        if month_num:
            return f"{int(year_s):04d}-{month_num:02d}-01"

    logger.warning("Could not parse date token %r — treating as None.", token)
    return None


# ---------------------------------------------------------------------------
# Pattern matchers — each returns a TemporalConstraint or None
# ---------------------------------------------------------------------------

_BETWEEN_RE = re.compile(
    r"between\s+"
    r"([\d][^\s]+(?:\s+\d{4})?)\s+and\s+"
    r"([\d][^\s]+(?:\s+\d{4})?)",
    re.IGNORECASE,
)
_NAMED_BETWEEN_RE = re.compile(
    r"between\s+"
    r"([a-z]+\s+\d{4})\s+and\s+"
    r"([a-z]+\s+\d{4})",
    re.IGNORECASE,
)
# Date token — matches (in priority order):
#   1. DD/MM/YYYY or DD-MM-YYYY
#   2. Month-name + year  (e.g. "june 2015", "january 2018")
#   3. ISO-8601 YYYY-MM-DD
#   4. Bare 4-digit year
_DATE_TOKEN = (
    r"(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{4}"
    r"|[a-z]+\s+(?:19|20)\d{2}"
    r"|(?:19|20)\d{2}-\d{2}-\d{2}"
    r"|(?:19|20)\d{2}"
    r")"
)

_AS_OF_RE = re.compile(
    rf"(?:as\s+of|as\s+at)\s+({_DATE_TOKEN})",
    re.IGNORECASE,
)
_BEFORE_RE = re.compile(
    rf"(?:before|prior\s+to)\s+({_DATE_TOKEN})",
    re.IGNORECASE,
)
_AFTER_RE = re.compile(
    rf"(?:after|since)\s+({_DATE_TOKEN})",
    re.IGNORECASE,
)
_IN_YEAR_RE = re.compile(
    r"(?:in|during)\s+((19|20)\d{2})\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"\b(current(?:ly)?|now|today|latest|present(?:ly)?|at\s+present)\b",
    re.IGNORECASE,
)


def _try_between(expr: str) -> Optional[TemporalConstraint]:
    # Try named-month form first ("january 2015 and december 2019")
    for pattern in (_NAMED_BETWEEN_RE, _BETWEEN_RE):
        m = pattern.search(expr)
        if m:
            start_token, end_token = m.group(1).strip(), m.group(2).strip()
            start_date = _parse_date_token(start_token, use_end=False)
            end_date = _parse_date_token(end_token, use_end=True)
            if start_date and end_date:
                sql = (
                    f"valid_from >= '{start_date}' "
                    f"AND (valid_to <= '{end_date}' OR valid_to IS NULL)"
                )
                return TemporalConstraint(
                    operator="between",
                    reference_date=start_date,
                    end_date=end_date,
                    sql_fragment=sql,
                )
    return None


def _try_as_of(expr: str) -> Optional[TemporalConstraint]:
    m = _AS_OF_RE.search(expr)
    if m:
        token = m.group(1).strip()
        ref = _parse_date_token(token)
        if ref:
            sql = (
                f"valid_from <= '{ref}' "
                f"AND (valid_to > '{ref}' OR valid_to IS NULL)"
            )
            return TemporalConstraint(
                operator="as_of",
                reference_date=ref,
                sql_fragment=sql,
            )
    return None


def _try_before(expr: str) -> Optional[TemporalConstraint]:
    m = _BEFORE_RE.search(expr)
    if m:
        token = m.group(1).strip()
        ref = _parse_date_token(token)
        if ref:
            sql = (
                f"valid_from < '{ref}' "
                f"AND (valid_to >= '{ref}' OR valid_to IS NULL)"
            )
            return TemporalConstraint(
                operator="before",
                reference_date=ref,
                sql_fragment=sql,
            )
    return None


def _try_after(expr: str) -> Optional[TemporalConstraint]:
    m = _AFTER_RE.search(expr)
    if m:
        token = m.group(1).strip()
        ref = _parse_date_token(token)
        if ref:
            sql = f"valid_from >= '{ref}'"
            return TemporalConstraint(
                operator="after",
                reference_date=ref,
                sql_fragment=sql,
            )
    return None


def _try_in_year(expr: str) -> Optional[TemporalConstraint]:
    m = _IN_YEAR_RE.search(expr)
    if m:
        year = int(m.group(1))
        start = _year_start(year)
        end = _year_end(year)
        sql = f"valid_from >= '{start}' AND (valid_to <= '{end}' OR valid_to IS NULL)"
        return TemporalConstraint(
            operator="in_year",
            reference_date=start,
            end_date=end,
            sql_fragment=sql,
        )
    return None


def _current_constraint() -> TemporalConstraint:
    return TemporalConstraint(
        operator="current",
        sql_fragment="valid_to IS NULL",
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# Priority-ordered list of handler functions.
_HANDLERS = [
    _try_between,
    _try_as_of,
    _try_before,
    _try_after,
    _try_in_year,
]


def generate(ir: IntermediateRepresentation) -> TemporalConstraint:
    """
    Convert the temporal expression inside *ir* into a formal
    :class:`TemporalConstraint`.

    Strategy:
      1. If no temporal expression is present → default to "current".
      2. If the expression signals present/current → "current".
      3. Try each handler in priority order; use the first match.
      4. If nothing matches → fall back to "current" (conservative default).

    Parameters
    ----------
    ir : IntermediateRepresentation
        Output from Layer B.

    Returns
    -------
    TemporalConstraint
        Contains operator, reference_date, end_date, and sql_fragment.
    """
    expr = ir.temporal_expression

    # No temporal expression → default
    if not expr:
        logger.debug("No temporal expression — defaulting to 'current'.")
        return _current_constraint()

    # Explicit current/present markers
    if _CURRENT_RE.search(expr):
        logger.debug("Detected present-tense marker in %r → 'current'.", expr)
        return _current_constraint()

    # Try each handler in priority order
    for handler in _HANDLERS:
        constraint = handler(expr)
        if constraint is not None:
            logger.debug(
                "TemporalConstraintGen: handler=%s, constraint=%s",
                handler.__name__,
                constraint.to_dict(),
            )
            return constraint

    # Conservative fallback
    logger.warning(
        "Could not resolve temporal expression %r — falling back to 'current'.",
        expr,
    )
    return _current_constraint()
