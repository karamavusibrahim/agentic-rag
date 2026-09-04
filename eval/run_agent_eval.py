#!/usr/bin/env python
"""End-to-end answer accuracy, and the contradiction-check ablation.

Two things are measured here that the retrieval eval in `sec-rag` cannot see.

**Answers, not passages.** Retrieval metrics reward putting the right chunk in
the top 10. They say nothing about whether the agent then read the right number
out of it -- which is where this agent actually failed.

**Whether the contradiction check earns its cost.** It adds a model call per
conflict, so it has to be worth something. The claim being tested is deliberately
narrow: it should convert *confidently wrong* answers into *honest abstentions*.
It is not expected to raise accuracy much, because an agent that mis-extracts a
figure in every hop has nothing to cross-check against. Scoring it on accuracy
alone would understate it; scoring it on abstention alone would let a broken
agent that abstains on everything look perfect. So outcomes are three-way:

    correct   the computed gold value appears in the answer
    wrong     an answer was given, and it is not the gold value
    abstain   the agent stated it could not determine the figure

The number that matters is the wrong -> abstain shift.

Two controls, because "the gold number appears in the answer" is a weak test on
its own:

  - **null-model hit rate.** For each question, every *other* question's gold
    value is also checked against the answer. An answer that lists many figures
    will match by luck, and the null rate exposes that. A real hit rate only
    means something to the extent it exceeds the null.
  - **unanswerable controls.** Questions about fiscal years outside the indexed
    corpus. Anything other than abstention there is fabrication, and it is
    reported separately rather than folded into accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from agentic_rag.agent.loop import MultiHopAgent  # noqa: E402
from agentic_rag.search_modes import make_search  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.retrieve.hybrid import Retriever  # noqa: E402

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
# Citations and SEC form names carry digits that are not figures: "[AAPL 10-K
# 2024, Item 8]" contributed "10" and "8", both of which sat inside the percent
# tolerance of real golds -- "See the 10-K filing." graded correct with no
# figure in it at all. Strip them before any number extraction.
_CITATION = re.compile(r"\[[^\]]*\]")
_FORM_TOKENS = re.compile(r"\b(?:10-[KQ]|8-K|20-F|Item\s+\d+[A-Z]?)\b", re.I)
# Phrases the synthesis prompt and the conflict renderer actually produce, plus
# the ordinary ways a model declines. Matched case-insensitively.
_ABSTAIN = re.compile(
    r"unresolved conflict|cannot (?:be )?(?:determine|calculat|comput)|"
    r"could not (?:be )?(?:determine|calculat|comput)|"
    r"not (?:be )?determined|unable to (?:determine|calculate|find|verify)|"
    r"do(?:es)? not (?:state|disclose|provide|contain)|not (?:found|available|"
    r"disclosed|stated|reported) in|insufficient (?:evidence|information)|"
    r"no (?:evidence|data|figure) (?:was )?(?:found|available)",
    re.I,
)
_COMPARATIVE = re.compile(
    r"\b(higher|larger|greater|more|bigger|exceed\w*|outspen\w*|spent more|"
    r"larger share)\b", re.I)
# Cues stating the *opposite* direction: "AAPL's net income was lower than
# MSFT's" names the smaller company first, so the verdict must be inverted.
_COMPARATIVE_INV = re.compile(
    r"\b(lower|smaller|less|fewer|trailed|lagged|behind|spent less)\b", re.I)
# A ticker inside a comparison phrase is the *object* of the comparison, not
# the subject of the claim: "Compared with MSFT, AAPL reported higher net
# income" names MSFT first and claims AAPL is larger. First-mention grading
# read that as MSFT's claim and marked a correct answer wrong. The phrase is
# removed before the subject is located; the ticker part is case-sensitive
# because tickers are upper-case and the connective is not.
_COMPARISON_OBJECT = re.compile(
    r"(?i:compared\s+(?:with|to)|relative\s+to|versus|vs\.?|than|against|"
    r"over)\s+(?:the\s+)?[A-Z]{2,5}(?:'s)?\b")
# "net income decreased by 3.36%" states -3.36 without a minus sign; the old
# grader marked every correctly-worded negative delta wrong.
_DOWNWARD = re.compile(
    r"\b(decreas\w+|declin\w+|fell|dropped|down|shrank|shrunk|reduc\w+|"
    r"contract\w+|negative)\b", re.I)

# Percentages: 2% relative is too tight for a figure the agent recomputes from
# rounded inputs, and too loose in absolute terms near zero, so accept either.
REL_TOL = 0.02
ABS_TOL_PCT = 0.30


def numbers_in(text: str) -> list[tuple[float, bool]]:
    """Candidate figures in an answer as (value, stated_as_percent) pairs,
    excluding citation/form-name digits.

    Citations are stripped first because "[AAPL 10-K 2024, Item 8]" contributed
    "10" and "8" -- both inside the percent tolerance of real golds, so an
    answer with no figure at all could grade correct.

    Sentences that state a downward move ("decreased by 3.36%") also contribute
    the negated value, because prose is how a model most naturally reports a
    negative delta -- an ASCII minus is the exception, not the rule. The Unicode
    minus (U+2212), which PDFs and some models emit, is normalized first for the
    same reason.
    """
    text = _CITATION.sub(" ", text)
    text = _FORM_TOKENS.sub(" ", text)
    text = text.replace("−", "-")
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        downward = bool(_DOWNWARD.search(sentence))
        for m in _NUM.finditer(sentence):
            try:
                v = float(m.group(0).replace(",", ""))
            except ValueError:
                continue
            pct = is_pct_context(sentence, m.end())
            out.append((v, pct))
            if downward and v > 0:
                out.append((-v, pct))
    return out


def is_pct_context(sentence: str, end: int) -> bool:
    return bool(re.match(r"\s*(?:%|per\s?cent|percentage[- ]point|pp\b)",
                         sentence[end:end + 20], re.I))


def matches(target: float, values: list[tuple[float, bool]],
            *, unit: str | None) -> bool:
    """A gold with a percent unit only matches numbers *stated as* percentages.

    Without that constraint any stray count within 2% relative -- "8 business
    segments" against a gold of 8.02 -- scores the point, which is how the old
    grader could be satisfied by an answer containing no figure of the right
    kind at all.
    """
    want_pct = bool(unit and ("percent" in unit or "pp" in unit))
    for v, stated_pct in values:
        if want_pct and not stated_pct:
            continue
        if abs(v - target) <= (ABS_TOL_PCT if want_pct else 0):
            return True
        scale = max(abs(v), abs(target))
        if scale and abs(v - target) / scale <= REL_TOL:
            return True
    return False


def abstained(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def grade_categorical(answer: str, gold: str, other: str) -> str:
    """Which company did the answer actually name as the larger one?

    Both tickers appear in the question and usually in the answer, so presence
    is not enough. The first *verdict* sentence carrying a comparative cue is
    the claim; sentences that merely restate the question are skipped, because
    an echo ("the question asks which company reported higher net income")
    names both tickers without claiming anything. Inverse cues ("AAPL was
    lower") name the smaller company first, and a negation before the cue
    ("did not report higher") flips the claim. Deterministic, and it returns
    "ungradeable" rather than guessing when no verdict sentence exists -- a
    guess here would silently become an accuracy number.
    """
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if sentence.rstrip().endswith("?") or \
                re.search(r"\b(?:question|asks?|asked)\b", sentence, re.I):
            continue
        pos = _COMPARATIVE.search(sentence)
        inv = _COMPARATIVE_INV.search(sentence)
        cue = min((m for m in (pos, inv) if m),
                  key=lambda m: m.start(), default=None)
        if cue is None:
            continue
        # Locate the subject in the sentence with comparison objects removed;
        # the cue position is taken from the original so the negation window
        # below still looks at the right text.
        up = _COMPARISON_OBJECT.sub(" ", sentence).upper()
        gi, oi = up.find(gold), up.find(other)
        if gi < 0 and oi < 0:
            continue
        first_is_gold = oi < 0 or (0 <= gi < oi)
        claims_first_larger = cue is pos
        if re.search(r"\b(?:not|never|n't)\b[\s\w]{0,15}$",
                     sentence[:cue.start()], re.I):
            claims_first_larger = not claims_first_larger
        return "correct" if first_is_gold == claims_first_larger else "wrong"
    return "ungradeable"


def conclusion_sentence(answer: str) -> str | None:
    """The first sentence that states a percent figure -- the answer's claim.

    Strict grading scores only this sentence, so a correct figure buried in
    the supporting detail of an answer whose *headline* is wrong no longer
    earns the point. Citations are stripped first for the usual reason.
    """
    text = _FORM_TOKENS.sub(" ", _CITATION.sub(" ", answer)).replace("−", "-")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if any(pct for _, pct in numbers_in(sentence)):
            return sentence
    return None


def grade(q: dict[str, Any], answer: str, *, strict: bool = False) -> str:
    """Outcome for one question.

    The stated conclusion is graded *before* the abstain check: a complete,
    correct answer that appends an honest hedge ("segment-level figures are
    not disclosed") is an answer, not an abstention. The abstain check applies
    only when no gradeable conclusion was found.

    An empty answer is graded as its own outcome: it is an infrastructure
    failure, not a wrong answer and not a fabrication -- and grading silence
    as fabrication once turned an API bad patch into a 4/6 control-failure
    rate that the agent never earned.
    """
    if not answer.strip():
        return "empty_answer"
    if q.get("expect_abstain"):
        return "correct" if abstained(answer) else "fabricated"
    if q.get("answer_categorical"):
        tickers = re.findall(r"\b[A-Z]{2,5}\b", q["question"])
        gold = q["answer_categorical"]
        other = next((t for t in tickers if t != gold), "")
        verdict = grade_categorical(answer, gold, other) if other else "ungradeable"
        if verdict in ("correct", "wrong"):
            return verdict
        return "abstain" if abstained(answer) else verdict
    target = q.get("answer_numeric")
    if target is None:
        return "ungradeable"
    scope = answer
    if strict:
        sent = conclusion_sentence(answer)
        if sent is None:
            return "abstain" if abstained(answer) else "ungradeable"
        scope = sent
    if matches(target, numbers_in(scope), unit=q.get("answer_unit")):
        return "correct"
    return "abstain" if abstained(answer) else "wrong"


def count_figures(answer: str) -> int:
    """Numeric tokens in the answer body, citations excluded, twins not counted."""
    return len(_NUM.findall(_FORM_TOKENS.sub(" ", _CITATION.sub(" ", answer))))


def null_hit(q: dict[str, Any], answer: str, others: list[dict[str, Any]]) -> bool:
    """Would some *other* question's gold value also have matched this answer?"""
    vals = numbers_in(answer)
    for o in others:
        t = o.get("answer_numeric")
        if t is None or o["qid"] == q["qid"]:
            continue
        if abs(t - (q.get("answer_numeric") or 1e18)) < 1e-9:
            continue  # same value; not an independent decoy
        if matches(t, vals, unit=o.get("answer_unit")):
            return True
    return False


def run_config(agent: MultiHopAgent, questions: list[dict[str, Any]],
               label: str, traces_out: Path | None,
               strict: bool = False) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    nulls = 0
    null_eligible = 0
    n_numbers = 0
    conflicts_seen = 0
    traces: list[dict[str, Any]] = []
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        try:
            trace = agent.run(q["question"])
            answer = trace.answer
            conflicts_seen += len(trace.conflicts)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(questions)}] ERROR {q['qid']}: {exc}", file=sys.stderr)
            outcomes["error"] += 1
            continue

        verdict = grade(q, answer, strict=strict)
        # An abstention (or a control "correct"-by-abstention) that coincides
        # with failed extractions is not evidence of honesty -- during an API
        # outage every hop reads "not found" and the agent abstains everywhere.
        # Surface it as its own outcome instead of letting it pass.
        n_err = sum(1 for e in trace.evidence if getattr(e, "error", None))
        if n_err and (verdict == "abstain"
                      or (q.get("expect_abstain") and verdict == "correct")):
            verdict = "abstain_due_to_error"
        outcomes[verdict] += 1
        # The null control is a control *for the accuracy number*, so it is
        # computed over the same questions accuracy is: answerable ones with
        # a numeric gold. Dividing by every question -- controls and
        # categorical ones included, which cannot null-hit -- deflated the
        # rate (12/30 = 0.40) below the published, correct 12/24 = 0.50 and a
        # rerun would have republished the rejected denominator.
        if q.get("answer_numeric") is not None and not q.get("expect_abstain"):
            null_eligible += 1
            nulls += null_hit(q, answer, questions)
        n_numbers += count_figures(answer)
        traces.append({"qid": q["qid"], "type": q["type"], "verdict": verdict,
                       "gold": q.get("answer_numeric") or q.get("answer_categorical"),
                       "answer": answer, "extract_errors": n_err,
                       "verification": trace.verification,
                       "conflicts": len(trace.conflicts),
                       "unresolved": sum(1 for c in trace.conflicts if c.unresolved)})
        print(f"  [{i}/{len(questions)}] {verdict:<11} {q['qid']}")

    if traces_out:
        traces_out.parent.mkdir(parents=True, exist_ok=True)
        traces_out.write_text(json.dumps(traces, indent=2))

    # Errors are API failures, not answers; folding them into the denominator
    # deflated every rate and let an outage impersonate a well-behaved agent.
    n = max(sum(outcomes.values()) - outcomes["error"], 1)
    return {
        "config": label,
        "n": n,
        "errors": outcomes["error"],
        "outcomes": dict(outcomes),
        "accuracy": outcomes["correct"] / n,
        "wrong_rate": outcomes["wrong"] / n,
        "abstain_rate": outcomes["abstain"] / n,
        "null_hit_rate": nulls / max(null_eligible, 1),
        "null_hit_denominator": null_eligible,
        "null_hits": nulls,
        "numbers_per_answer": round(n_numbers / n, 1),
        "conflicts_detected": conflicts_seen,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../sec-rag/data/processed"))
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval/multihop.jsonl"))
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--types", default="trend,ratio,delta,compare,unanswerable")
    ap.add_argument("--grade", choices=("loose", "strict"), default="loose",
                    help="strict grades only the answer's conclusion sentence")
    ap.add_argument("--verify", choices=("off", "dual"), default="off",
                    help="dual runs two specialized post-synthesis critics")
    ap.add_argument("--qids", default=None,
                    help="comma-separated qids to run (overrides --limit; for "
                         "re-running questions lost to infrastructure failures)")
    ap.add_argument("--ablate", action="store_true",
                    help="deprecated alias for --config both")
    # The two arms are independent, so they can be run as separate processes and
    # merged afterwards. At ~7 minutes per agent run, doing them sequentially in
    # one process doubles wall-clock for no methodological benefit -- the same
    # question set is used either way, since --limit selects deterministically.
    # Default is a single arm: "both" previously required --ablate as well,
    # which silently skipped the OFF arm for anyone who took the flag at its word.
    ap.add_argument("--config", choices=("on", "off", "both"), default="on")
    ap.add_argument("--search-mode", choices=("dense", "relgrep"), default="dense",
                    help="relgrep = optional relevance-guided corpus grep "
                         "(arXiv 2607.24223); dense is the measured default")
    ap.add_argument("--retrieval-grader", action="store_true",
                    help="optional corrective retrieval (CRAG, arXiv 2401.15884): "
                         "grade each hop's passages and re-query once when they "
                         "are off-target; applies to every arm; unmeasured")
    ap.add_argument("--out", type=Path, default=Path("eval/results/agent.json"))
    args = ap.parse_args()

    wanted = set(args.types.split(","))
    all_qs = [json.loads(l) for l in args.eval_set.read_text().splitlines() if l]
    pool = [q for q in all_qs if q["type"] in wanted]
    if args.qids:
        keep = {s.strip() for s in args.qids.split(",")}
        pool = [q for q in pool if q["qid"] in keep]
        args.limit = len(pool)

    # Round-robin by type so a --limit does not silently become "all ratios".
    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in pool:
        by_type.setdefault(q["type"], []).append(q)
    questions: list[dict[str, Any]] = []
    while len(questions) < args.limit and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(questions) < args.limit:
                questions.append(by_type[t].pop(0))

    index = load_index(args.index)
    retriever = Retriever(index)
    search = make_search(retriever, mode=args.search_mode, candidates=50)

    print(f"corpus {len(index)} chunks | {len(questions)} questions "
          f"({Counter(q['type'] for q in questions)})\n")

    rows = []
    if args.config in ("on", "both"):
        print("=== contradiction check ON ===")
        rows.append(run_config(
            MultiHopAgent(search,
                          verify_answer=None if args.verify == "off" else args.verify,
                          retrieval_grader=args.retrieval_grader),
            questions, "contradiction check ON",
            Path("eval/results/traces_on.json"), strict=args.grade == "strict"))
    if args.config in ("off", "both") or args.ablate:
        print("\n=== contradiction check OFF ===")
        rows.append(run_config(MultiHopAgent(search, check_contradictions=False,
                                             retrieval_grader=args.retrieval_grader),
                               questions, "contradiction check OFF",
                               Path("eval/results/traces_off.json")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_questions": len(questions), "results": rows}, indent=2))

    print("\n| Configuration | correct | wrong | abstain | null-hit | nums/answer |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['accuracy']:.2f} | {r['wrong_rate']:.2f} | "
              f"{r['abstain_rate']:.2f} | {r['null_hit_rate']:.2f} | "
              f"{r['numbers_per_answer']} |")
    for r in rows:
        print(f"\n{r['config']}: outcomes={r['outcomes']} "
              f"conflicts={r['conflicts_detected']} ({r['seconds']}s)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
