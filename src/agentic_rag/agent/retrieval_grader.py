"""Optional corrective retrieval: grade the hop's passages before extracting.

Corrective RAG (Yan et al., arXiv 2401.15884) puts a lightweight evaluator
between retrieval and generation: it scores the retrieved documents for the
query, and the score chooses an action -- use them, discard them and search
again, or keep the ambiguous ones and supplement them. The failure it
targets is one this agent has met: a hop whose passages are all about the
wrong year or the wrong line item still reaches the extractor, which then
does its best with them, and "its best" is the FY2024 figure relabelled as
FY2018 that `period_mismatch` exists to catch downstream. Grading the
passages first is the upstream version of that guard.

The adaptation here is deliberately small and honest about what it is not:

- No web search. CRAG falls back to the open web when the corpus fails; this
  corpus is six filings and the whole point of the eval is that the answer
  is either in them or the agent must say so. The corrective action is a
  single re-query with the agent's existing filing-vocabulary reformulation.
- No decompose-then-recompose strip refinement. The extractor already reads
  passages selectively; stripping them further would just move text out of
  its view.
- Not measured. The grader costs one small LLM call per hop, and this
  repository quotes no number for it because the ON/OFF comparison has not
  been run against the hosted API. `run_agent_eval.py --retrieval-grader`
  is where that number would come from.

Everything network-bound is injected so the action logic is testable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

GRADE_PROMPT = """You judge whether retrieved passages can answer ONE sub-question.

Sub-question: {sub_question}

Passages:
{passages}

For each passage decide:
- "relevant"   it states the figure or fact asked for, for the company AND
               the fiscal period asked for
- "ambiguous"  it is about the right company and metric but a different
               period, or states the figure without a clear period
- "irrelevant" none of the above

Return JSON only:
{{"grades": ["relevant" | "ambiguous" | "irrelevant", ...]}}
One entry per passage, in order. Do not extract the answer."""

LABELS = ("relevant", "ambiguous", "irrelevant")
GRADE_CHARS = 1200


@dataclass
class Correction:
    """What the grader decided for one hop."""
    action: str                      # "correct" | "ambiguous" | "incorrect" | "skipped"
    hits: list[Any] = field(default_factory=list)
    grades: list[str] = field(default_factory=list)
    requery: str | None = None
    note: str = ""


def grade_passages(sub_question: str, hits: Sequence[Any], models: Sequence[str],
                   *, chat_json_chain: Callable[..., tuple[Any, str]],
                   ) -> list[str] | None:
    """One label per hit, or None if the grader call failed or was malformed.

    A malformed reply -- wrong length, unknown label -- is treated as a
    failed grade rather than coerced: a grader that is guessing about the
    passages is worse than no grader.
    """
    passages = "\n\n".join(
        f"[{i}] ({h.citation()})\n{h.text[:GRADE_CHARS]}"
        for i, h in enumerate(hits, 1))
    try:
        data, _ = chat_json_chain(
            list(models),
            [{"role": "user", "content": GRADE_PROMPT.format(
                sub_question=sub_question, passages=passages)}],
            validate=lambda d: isinstance(d.get("grades"), list),
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    retrieval grader failed: {exc}", file=sys.stderr)
        return None
    grades = [str(g).strip().lower() for g in data["grades"]]
    if len(grades) != len(hits) or any(g not in LABELS for g in grades):
        print(f"    retrieval grader returned an unusable reply: {grades!r}",
              file=sys.stderr)
        return None
    return grades


def correct_retrieval(sub_question: str, hits: Sequence[Any], *,
                      models: Sequence[str],
                      search: Callable[[str, int], Sequence[Any]],
                      reformulate: Callable[[Sequence[str]], Sequence[str]],
                      k: int,
                      chat_json_chain: Callable[..., tuple[Any, str]],
                      ) -> Correction:
    """CRAG's three actions over this corpus, without the web fallback.

    correct    at least one passage is relevant: keep the relevant ones
               (ambiguous ones too, so the period cross-check still sees
               them) and drop the irrelevant.
    ambiguous  nothing relevant, something ambiguous: keep the ambiguous
               passages and add the top hits of one reformulated query.
    incorrect  nothing usable: discard, re-query once with the reformulation
               and grade *that*; if it is no better, hand back the original
               hits so the extractor -- and its own guards -- still get a
               chance. The grader narrows; it never leaves a hop with less
               than it started with unless it found something better.
    skipped    the grader failed; the hop proceeds ungraded.
    """
    hits = list(hits)
    if not hits:
        return Correction("skipped", hits, note="nothing retrieved")
    grades = grade_passages(sub_question, hits, models,
                            chat_json_chain=chat_json_chain)
    if grades is None:
        return Correction("skipped", hits, note="grader unavailable")

    keep = [h for h, g in zip(hits, grades) if g != "irrelevant"]
    if "relevant" in grades:
        return Correction("correct", keep, grades)

    rewritten = list(reformulate([sub_question]) or [])
    requery = rewritten[0] if rewritten else None
    extra: list[Any] = []
    if requery and requery.strip() and requery.strip() != sub_question.strip():
        seen = {h.chunk_id for h in keep}
        extra = [h for h in search(requery, k) if h.chunk_id not in seen]

    if "ambiguous" in grades:
        return Correction("ambiguous", keep + extra, grades, requery)

    # Nothing usable. Grade the re-query's passages before trusting them.
    if extra:
        regrades = grade_passages(sub_question, extra, models,
                                  chat_json_chain=chat_json_chain)
        if regrades and any(g != "irrelevant" for g in regrades):
            better = [h for h, g in zip(extra, regrades) if g != "irrelevant"]
            return Correction("incorrect", better, grades, requery,
                              note="re-query produced usable passages")
    return Correction("incorrect", hits, grades, requery,
                      note="re-query no better; original passages kept")
