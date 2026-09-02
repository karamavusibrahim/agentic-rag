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

# How much of a passage the verifier is shown, and therefore how much of it may
# be used to ground the answer. The two must be the same number.
VERIFY_CHARS = 1200

# Implicit scaling -- reading a bare "12,914" as $12,914 million -- is only
# justified when the passage says it reports in thousands or millions. Applying
# it unconditionally let any bare number stand in for a scaled figure, so
# "the company had 100 employees" grounded a claim of "$100 million".
_SCALE_CAPTION = re.compile(
    r"\(?\bin\s+(thousand|million|billion)s?\b", re.I)


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


_YEARISH = re.compile(r"^(19|20)\d{2}$")

# Words that mark a following four-digit token as a date rather than a figure.
# "fiscal 2024", "FY 2024", but also "for the fiscal year ended 2024" and
# "year ending December 31, 2024" -- the keyword may sit a few tokens back.
_FISCAL_CTX = re.compile(
    r"(?:fiscal|fy|calendar|year)\b[\w\s,]{0,24}$", re.I)


def value_in_passage(value: str, passage: str,
                     *, tolerance: float = REL_TOLERANCE) -> bool:
    """Is the resolved figure actually present in the passage that was cited?

    Filings and extractions disagree about presentation, not magnitude: a
    passage prints "12,914" inside a table captioned "in millions" while the
    extractor reports "$12,914 million". So a bare number in the passage may
    legitimately stand for a scaled figure, and that has to be allowed.

    Allowing it carelessly is how the first version of this check passed three
    hallucinations:

    - ``value_in_passage("$0", ...)`` searched for the character "0" and
      matched any passage containing one.
    - A passage saying "$12.914 billion" matched a claimed "$12.914 trillion",
      because the passage's already-scaled value was multiplied again.
    - "FY2024" matched a claimed "$2.024 million": 2024 x 1e3.

    So implicit scaling is applied only where it is actually plausible:
    to a passage token carrying **no** scale word of its own, and never to a
    bare four-digit year. Those are heuristics, and they are the reason this
    function reports *grounding*, not correctness -- it can still be fooled by
    a passage that happens to contain the claimed digits in an unrelated row.
    Its job is to stop a figure that appears nowhere at all from being labelled
    "verified".

    A non-numeric value (e.g. "the Americas segment") cannot be checked this
    way, so fall back to a case-insensitive substring test.
    """
    target = parse_magnitude(value)
    if target is None:
        needle = value.strip().lower()
        return bool(needle) and needle in passage.lower()

    # A caption's scope is its sentence, not the whole passage. One chunk can
    # caption one table "(in thousands)" and the next "(in millions)", and
    # applying every caption to every number let Table A's employee count be
    # scaled by Table B's caption into a revenue figure. Sentence boundaries
    # (". " -- decimals never have a space after the point) keep a trailing
    # caption like "... 12,914 compared to 8,701 (in millions)." attached to
    # its own numbers without leaking across tables.
    caption_positions = [(m.start(), _SCALES[m.group(1).lower()])
                         for m in _SCALE_CAPTION.finditer(passage)]

    def applicable_scales(pos: int) -> set[float]:
        """The caption(s) that plausibly govern the number at `pos`.

        Two patterns coexist in filing text and each broke a previous version
        of this function. "Figures are in millions. Revenue was 2,024." puts
        the caption in an earlier sentence, so sentence-only scope missed it;
        "Table A (in thousands): ... Table B (in millions): ..." puts a second
        caption between an earlier caption and later numbers, so passage-wide
        scope let Table B's caption reach Table A's numbers. The union that
        respects both: the nearest caption *before* the number (a caption
        governs what follows it, until superseded), plus any caption in the
        same sentence (a trailing "(in millions)." governs the numbers before
        it in its own sentence).
        """
        scales: set[float] = set()
        preceding = [sc for at, sc in caption_positions if at < pos]
        if preceding:
            scales.add(preceding[-1])
        lo = passage.rfind(". ", 0, pos)
        lo = 0 if lo == -1 else lo + 2
        hi = passage.find(". ", pos)
        hi = len(passage) if hi == -1 else hi + 1
        scales.update(sc for at, sc in caption_positions if lo <= at < hi)
        return scales

    for m in _NUM_RE.finditer(passage):
        raw, scale_word = m.group(1), (m.group(2) or "").lower()
        try:
            found = float(raw.replace(",", ""))
        except ValueError:
            continue
        if "(" in m.group(0) and ")" in m.group(0):
            found = -abs(found)

        if scale_word:
            # The passage said what it means. Take it at its word and do not
            # invent a further multiplier.
            if values_agree(target, found * _SCALES.get(scale_word, 1.0),
                            tolerance=tolerance):
                return True
            continue

        # A four-digit token in explicit fiscal context is a date, full stop:
        # "fiscal 2024" and "FY 2024" never mean $2,024 whatever caption the
        # sentence carries, and scaling them manufactured plausible-looking
        # figures out of dates. Outside that context "year" is only a guess --
        # "Revenue (in millions): 2024" is a real figure -- so a year-shaped
        # token without fiscal context stays eligible for caption scaling.
        # (_NUM_RE captures sentence-final years as "2024.", hence the strip.)
        bare = raw.rstrip(".")
        if _YEARISH.match(bare) and \
                _FISCAL_CTX.search(passage[max(0, m.start(1) - 34):m.start(1)]):
            continue

        if values_agree(target, found, tolerance=tolerance):
            return True
        for scale in applicable_scales(m.start()):
            if values_agree(target, found * scale, tolerance=tolerance):
                return True
    return False


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
        f"[{i}] ({h.citation()})\n{h.text[:VERIFY_CHARS]}"
        for i, h in enumerate(hits, 1)
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

    # The passage index has to be a real index into what we actually sent. The
    # previous `else hits[0]` silently attached the first passage's citation to
    # a value the model may have read somewhere else -- or invented -- which is
    # exactly the failure the guard exists to prevent.
    # `isinstance(True, int)` is True in Python, so a model answering
    # `"passage_number": true` sailed through the range check as passage 1 --
    # reintroducing exactly the silent hits[0] fallback this guard replaced.
    n = data.get("passage_number")
    if not (isinstance(n, int) and not isinstance(n, bool)
            and 1 <= n <= len(hits)):
        conflict.reason = (
            f"verifier did not cite a passage it was shown (passage_number={n!r})"
        )
        return conflict
    hit = hits[n - 1]

    # `resolved_value` is rendered to the synthesizer as "(verified)". That word
    # has to mean something, so require the figure to actually occur in the
    # passage the verifier cited. Without this the resolver is free to return a
    # value that appears in no passage at all and have it enter synthesis
    # carrying more authority than the conflicting extractions it replaced.
    # Validate against exactly what the verifier was shown. Checking the full
    # passage let a figure sitting past the truncation point count as grounded,
    # so the model could return a value it never saw and have the check confirm
    # it -- a coincidence dressed up as verification.
    if not value_in_passage(str(value), hit.text[:VERIFY_CHARS]):
        conflict.reason = (
            f"verifier returned {value!r}, which does not appear in the passage "
            f"it cited ({hit.citation()})"
        )
        return conflict

    conflict.resolved_value = str(value)
    conflict.resolved_chunk_id = hit.chunk_id
    conflict.reason = str(data.get("reason") or "")
    return conflict
