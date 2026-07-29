"""Detect and resolve contradictory extractions before synthesis.

The failure this exists for, observed in a real run: the agent extracted NVDA's
FY2025 R&D expense as $4,239M in one hop, having already retrieved the correct
~$12,914M in another. It noticed the conflict, mentioned it in the answer, and
then picked the wrong one. A synthesizer asked to "compose an answer" treats
choosing between candidates as its job, and it has no grounds to choose well.

So the choice is taken away from it. Extractions are grouped by what they are
*about* -- (entity, metric, period) -- and any group whose values disagree is
re-verified against the source passages before synthesis ever sees it. If
verification cannot settle it, the group is marked unresolved and the
synthesizer is instructed to report the conflict rather than resolve it.

Why grouping needs structured fields rather than string similarity: two
sub-questions can be worded very differently and mean the same thing ("NVDA R&D
FY2025" vs "Nvidia research and development expense for fiscal year 2025"), while
two nearly identical strings can differ in the one token that matters (FY2024 vs
FY2025). The extractor is therefore asked to emit entity/metric/period
explicitly, and grouping is done on those.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..nvidia import chat_json_chain

# Scale words as they appear in filings, mapped to multipliers so that
# "$12.9 billion" and "12,914 million" compare equal.
_SCALES = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "m": 1e6, "mm": 1e6,
    "billion": 1e9, "billions": 1e9, "b": 1e9, "bn": 1e9,
    "trillion": 1e12, "trillions": 1e12,
}

_NUM_RE = re.compile(
    r"\(?\$?\s*(-?\d[\d,]*\.?\d*)\s*\)?\s*"
    r"(thousand[s]?|million[s]?|billion[s]?|trillion[s]?|bn|mm|[kmb])?",
    re.I,
)

# Values within this relative distance are treated as the same figure -- filings
# round inconsistently between statements (60,922 vs 60,925 for the same line).
REL_TOLERANCE = 0.01


def parse_magnitude(text: str | None) -> float | None:
    """Best-effort numeric magnitude from an extracted value string.

    Handles "$12,914 million", "12.9 billion", "(1,234)" for negatives. Returns
    None when there is no parseable number, which is itself informative -- a
    non-numeric value cannot be contradiction-checked this way.
    """
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    raw, scale = m.group(1), (m.group(2) or "").lower()
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    if "(" in text[: m.start() + 1] and ")" in text[m.end() - 1 :]:
        value = -abs(value)
    return value * _SCALES.get(scale, 1.0)


def values_agree(a: float | None, b: float | None,
                 *, tolerance: float = REL_TOLERANCE) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= tolerance


@dataclass
class Conflict:
    key: tuple[str, str, str]
    candidates: list[Any] = field(default_factory=list)   # Evidence objects
    resolved_value: str | None = None
    resolved_chunk_id: str | None = None
    reason: str = ""

    @property
    def unresolved(self) -> bool:
        return self.resolved_value is None

    def render(self) -> str:
        entity, metric, period = self.key
        head = f"{entity} {metric} {period}"
        if self.resolved_value:
            return f"- {head} -> {self.resolved_value} (verified; conflict resolved)"
        alts = "; ".join(
            f"{c.value} [{c.citation}]" for c in self.candidates if c.value
        )
        return f"- {head} -> UNRESOLVED CONFLICT between: {alts}"


_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _norm_entity(ev: Any) -> str:
    """Prefer the ticker parsed from the citation over the model's free text.

    The extractor writes whatever the passage said -- "NVDA", "Nvidia",
    "NVIDIA Corporation" -- which will not group. The citation is machine-built
    by the retriever ("NVDA 10-K 2025, Item 7"), so its leading token is a
    reliable ticker. Fall back to the model's string only when there is no
    citation to read.
    """
    citation = (getattr(ev, "citation", None) or "").strip()
    if citation:
        head = citation.split()[0].strip(",")
        if head.isupper() and 1 <= len(head) <= 6:
            return head.lower()
    return (getattr(ev, "entity", None) or "").strip().lower()


def _norm_period(raw: str) -> str:
    """Collapse "FY2025" / "fiscal year 2025" / "2025" to "2025".

    An earlier version stripped the words "fiscal"/"year"/"fy" with word-boundary
    regexes, which silently failed on "FY2025" -- there is no boundary between
    "fy" and "2025", so the group key stayed "fy2025" and never matched "2025".
    Extracting the year outright avoids the whole class of problem.
    """
    m = _YEAR_RE.search(raw)
    if m:
        return m.group(0)
    return re.sub(r"\s+", " ", re.sub(r"\b(fiscal|year|fy|ended)\b", "", raw)).strip()


def period_year(text: str | None) -> str | None:
    """First 4-digit year in a period string, or None."""
    if not text:
        return None
    m = _YEAR_RE.search(text)
    return m.group(0) if m else None


def period_years(text: str | None) -> set[str]:
    """Every 4-digit year in a period string.

    The planner is *asked* for atomic sub-questions but demonstrably emits
    compound ones ("R&D spending for fiscal years 2024 and 2025"), and NVDA's
    fiscal phrasing spans two calendar years ("February 2024 to January 2025").
    Judging such strings by their first year alone rejected correct FY2024/
    FY2025 extractions as mismatches.
    """
    if not text:
        return set()
    return {m.group(0) for m in _YEAR_RE.finditer(text)}


def period_mismatch(requested: str | None, stated: str | None) -> bool:
    """Did the extractor answer about a different period than it was asked?

    This exists because of a failure that contradiction detection structurally
    *cannot* catch. Asked for AAPL's FY2018 R&D-to-revenue ratio -- a year
    outside the indexed filings -- the agent retrieved AAPL's genuine FY2024
    figures ($31,370M and $391,035M), labelled them "fiscal 2018", and returned
    a confident ratio with real citations to real chunks.

    Every extraction agreed with every other, so there was no disagreement to
    detect. Contradiction detection finds *inconsistency*; this failure is
    perfectly consistent and entirely wrong. A uniform error is invisible to a
    cross-check, which is why this guard is deterministic and separate.

    Conservative by design: only fires when both sides name at least one year
    and the sets share none. A missing year is not evidence of a mismatch, and
    neither is a compound request ("2024 and 2025") answered for one of its
    years, nor a fiscal range ("February 2024 to January 2025") whose first
    calendar year differs from its fiscal label.
    """
    a, b = period_years(requested), period_years(stated)
    return bool(a and b and not (a & b))


def group_key(ev: Any) -> tuple[str, str, str] | None:
    """(entity, metric, period), normalized. None if identity is unrecoverable."""
    entity = _norm_entity(ev)
    metric = (getattr(ev, "metric", None) or "").strip().lower()
    period = _norm_period((getattr(ev, "period", None) or "").strip().lower())
    if not (entity and metric and period):
        return None
    # Collapse trivial wording differences that would otherwise split a group.
    metric = re.sub(r"\b(expense|expenses|total|net)\b", "", metric).strip()
    metric = re.sub(r"\s+", " ", metric)
    return (entity, metric, period)


def find_conflicts(evidence: Sequence[Any]) -> list[Conflict]:
    """Group found evidence and return groups whose values disagree."""
    groups: dict[tuple[str, str, str], list[Any]] = {}
    for ev in evidence:
        if not getattr(ev, "found", False):
            continue
        key = group_key(ev)
        if key is None:
            continue
        groups.setdefault(key, []).append(ev)

    conflicts: list[Conflict] = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        mags = [parse_magnitude(i.value) for i in items]
        # A conflict exists if any pair of *numeric* values disagrees beyond
        # tolerance. Unparseable values are grouped but not compared -- two
        # identical textual claims ("cited supply constraints") are not a
        # disagreement, and values_agree() on None can never return True.
        if any(
            not values_agree(mags[i], mags[j])
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if mags[i] is not None and mags[j] is not None
        ):
            conflicts.append(Conflict(key=key, candidates=list(items)))
    return conflicts


VERIFY_PROMPT = """Two or more extractions disagree about the same figure. Determine which is correct.

Figure: {entity} -- {metric} -- {period}

Candidate values:
{candidates}

Source passages:
{passages}

Return JSON only:
{{"correct_value": "the right figure with units, or null if undeterminable",
  "passage_number": <1-based index of the passage that settles it, or null>,
  "reason": "one sentence"}}

Rules:
- Read the passages directly. Do not favour a candidate because it appeared more
  often -- an error repeated twice is still an error.
- Watch for a QUARTERLY figure mistaken for an annual one, and for a prior-year
  comparative column read as the current year. These are the usual causes.
- Sanity-check magnitude: a company's annual R&D or revenue does not change by
  more than roughly half year over year.
- If the passages genuinely do not settle it, return null. An honest "cannot
  determine" is correct; a guess is not."""


def resolve(
    conflict: Conflict,
    search: Callable[[str, int], Sequence[Any]],
    models: Sequence[str],
    *,
    k: int = 6,
) -> Conflict:
    """Re-retrieve for the disputed figure and adjudicate against the passages."""
    entity, metric, period = conflict.key
    query = f"{entity} {metric} {period}"
    hits = list(search(query, k))
    if not hits:
        conflict.reason = "no passages retrieved for re-verification"
        return conflict

    passages = "\n\n".join(
        f"[{i}] ({h.citation()})\n{h.text[:1200]}" for i, h in enumerate(hits, 1)
    )
    candidates = "\n".join(
        f"- {c.value} [{c.citation}]" for c in conflict.candidates if c.value
    )
    try:
        data, _ = chat_json_chain(
            models,
            [{"role": "user", "content": VERIFY_PROMPT.format(
                entity=entity, metric=metric, period=period,
                candidates=candidates, passages=passages)}],
            validate=lambda d: "correct_value" in d,
            max_tokens=700,
        )
    except Exception as exc:  # noqa: BLE001
        conflict.reason = f"verification call failed: {exc}"
        return conflict

    value = data.get("correct_value")
    if value in (None, "", "null"):
        conflict.reason = str(data.get("reason") or "passages did not settle it")
        return conflict

    n = data.get("passage_number")
    hit = hits[n - 1] if isinstance(n, int) and 1 <= n <= len(hits) else hits[0]
    conflict.resolved_value = str(value)
    conflict.resolved_chunk_id = hit.chunk_id
    conflict.reason = str(data.get("reason") or "")
    return conflict
