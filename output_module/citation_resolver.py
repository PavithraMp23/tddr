"""
output_module/citation_resolver.py
------------------------------------
Matches LLM-cited section identifiers against retrieved regulation chunks
and returns structured ``LegalEvidence`` records.

Algorithm
---------
For each cited section (e.g. ``"Rule 5"``, ``"Section 302"``):
  1. Search every retrieved chunk for the section string (case-insensitive).
  2. On a match, extract a ≤300-char excerpt from the chunk.
  3. Deduplicate on ``(regulation, version, section)`` — keep the first match.
  4. If no chunk matches a citation, include a minimal evidence record with an
     empty excerpt so the citation is still visible in the final output.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from output_module.response_models import LegalEvidence, EXCERPT_MAX_CHARS

if TYPE_CHECKING:
    from rag_module.models import RegulationChunk


class CitationResolver:
    """Resolves LLM-cited section labels to concrete regulation evidence."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        cited_sections: List[str],
        chunks: List["RegulationChunk"],
    ) -> List[LegalEvidence]:
        """
        Match *cited_sections* against *chunks* and return evidence records.

        Parameters
        ----------
        cited_sections : Section labels produced by the LLM (e.g. ``["Rule 5"]``).
        chunks         : Retrieved ``RegulationChunk`` objects from Module 2.

        Returns
        -------
        List[LegalEvidence]
            One record per unique ``(regulation, version, section)`` triple.
        """
        seen: set[tuple[str, str, str]] = set()
        evidence: List[LegalEvidence] = []

        for section in cited_sections:
            matched = self._find_matching_chunk(section, chunks)

            if matched is not None:
                key = (matched.regulation, matched.version, section)
                if key in seen:
                    continue
                seen.add(key)

                excerpt = self._extract_excerpt(section, matched.text)
                evidence.append(
                    LegalEvidence(
                        regulation    = matched.regulation,
                        version       = matched.version,
                        section       = section,
                        effective_from= matched.effective_from,
                        effective_to  = matched.effective_to,
                        excerpt       = excerpt,
                        source_file   = matched.source_file,
                    )
                )
            else:
                # Citation not found in any chunk — include hollow record
                key = ("UNKNOWN", "?", section)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    LegalEvidence(
                        regulation    = "UNKNOWN",
                        version       = "?",
                        section       = section,
                        effective_from= "—",
                        effective_to  = None,
                        excerpt       = "",
                        source_file   = None,
                    )
                )

        return evidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_matching_chunk(
        section: str,
        chunks: List["RegulationChunk"],
    ) -> "RegulationChunk | None":
        """
        Return the first chunk whose text contains *section* (case-insensitive).

        Ranking preference (in order):
        1. Chunk text contains the exact section string (case-insensitive).
        2. Any word from the section string appears in the chunk text.
        """
        section_lower = section.lower().strip()

        # Pass 1: exact substring match
        for chunk in chunks:
            if section_lower in chunk.text.lower():
                return chunk

        # Pass 2: any keyword from section label
        keywords = [w for w in section_lower.split() if len(w) > 2]
        for chunk in chunks:
            text_lower = chunk.text.lower()
            if any(kw in text_lower for kw in keywords):
                return chunk

        return None

    @staticmethod
    def _extract_excerpt(section: str, text: str) -> str:
        """
        Extract a short excerpt from *text* that contains *section*.

        Attempts to start the excerpt just before the section mention so the
        reader sees context, then trims to ``EXCERPT_MAX_CHARS``.
        """
        lower_text    = text.lower()
        lower_section = section.lower().strip()

        pos = lower_text.find(lower_section)
        if pos == -1:
            # Fall back to the beginning of the chunk
            return text[:EXCERPT_MAX_CHARS].strip() + ("…" if len(text) > EXCERPT_MAX_CHARS else "")

        # Start slightly before the match for context
        start = max(0, pos - 40)
        end   = start + EXCERPT_MAX_CHARS
        raw   = text[start:end].strip()
        if start > 0:
            raw = "…" + raw
        if end < len(text):
            raw = raw + "…"
        return raw
