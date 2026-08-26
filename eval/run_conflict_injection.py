#!/usr/bin/env python
"""Fault-injection validation of the contradiction guard.

The guard has never fired end-to-end: across every natural eval run,
`conflicts detected = 0`, so the ON/OFF ablation executes identical code and
measures nothing. That is a scope statement, not a validation. The standard
fix for evaluating a guardrail that never fires on natural data is to inject
the fault it exists to catch (arXiv 2504.00180) and measure whether it fires.

Injection: for the FIRST successful extraction with a parseable magnitude in
each question, a corrupted twin Evidence is added alongside it --

  - value scaled by 0.33, mimicking the observed quarterly-for-annual failure
    ($4,239M extracted where ~$12,914M was correct, a ~0.33 ratio)
  - variant entity/period spellings ("NVIDIA Corporation", "fiscal year N"),
    so the twin only groups with the original if `_norm_entity`/`_norm_period`
    do their job -- the seam the audit showed was the actual weak link
  - a different real chunk_id/citation from the same hit list, so `resolve`
    re-retrieves real passages

The twin is added at the same seam the real failure occurred (between
extraction and critique), by overriding `critique` -- the first hook that
receives the live evidence list.

Measured, per arm over the same questions:
  ON  -- detection rate (conflicts fired / injections), and whether the final
         answer survives (correct) or degrades (wrong/abstain)
  OFF -- how often the corrupted figure contaminates the final answer

The wrong->abstain-or-correct shift between arms is the guard's measured
value: the number REPORT.md 5.3 admits it does not have.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from agentic_rag.agent.contradictions import parse_magnitude  # noqa: E402
from agentic_rag.agent.loop import Evidence, MultiHopAgent  # noqa: E402
from agentic_rag.search_modes import make_search  # noqa: E402
from run_agent_eval import grade, numbers_in  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.retrieve.hybrid import Retriever  # noqa: E402

CORRUPT_FACTOR = 0.33

ENTITY_VARIANTS = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
                   "NVDA": "NVIDIA Corporation"}


def corrupt_value(value: str) -> str | None:
    mag = parse_magnitude(value)
    if mag is None:
        return None
    c = mag * CORRUPT_FACTOR
    if c >= 1e9:
        return f"${c / 1e9:,.2f} billion"
    if c >= 1e6:
        return f"${c / 1e6:,.0f} million"
    return f"{c:,.2f}"


class CorruptingAgent(MultiHopAgent):
    """MultiHopAgent that injects one corrupted twin extraction per run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._injected: Evidence | None = None
        self._pending: list[Evidence] = []

    def run(self, question: str, **kw: Any):  # type: ignore[override]
        with self._lock:
            self._injected = None
            self._pending = []
        return super().run(question, **kw)

    def extract(self, sub_question: str) -> Evidence:
        ev = super().extract(sub_question)
        if not ev.found or ev.value is None:
            return ev
        corrupted = corrupt_value(ev.value)
        if corrupted is None:
            return ev
        with self._lock:
            if self._injected is not None:
                return ev
            entity = (ev.entity or "").upper()
            variant_entity = None
            for tick, name in ENTITY_VARIANTS.items():
                if tick in entity or tick in (ev.citation or ""):
                    variant_entity = name
                    break
            period = ev.period or ""
            variant_period = (period.replace("FY", "fiscal year ")
                              if "FY" in period else period)
            twin = Evidence(
                sub_question=ev.sub_question,
                found=True,
                value=corrupted,
                quote=None,
                chunk_id=ev.chunk_id,
                citation=ev.citation,
                entity=variant_entity or ev.entity,
                metric=ev.metric,
                period=variant_period,
            )
            self._injected = twin
            self._pending.append(twin)
        return ev

    def critique(self, question: str, evidence: Sequence[Evidence]):
        # First hook that sees the live trace.evidence list (passed by
        # reference); the injection lands here, at the same point in the loop
        # where the historical $4,239M-vs-$12,914M pair coexisted.
        with self._lock:
            if self._pending and isinstance(evidence, list):
                evidence.extend(self._pending)
                self._pending = []
        return super().critique(question, evidence)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../sec-rag/data/processed"))
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval/multihop.jsonl"))
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--search-mode", choices=("dense", "relgrep"), default="dense")
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/conflict_injection.json"))
    args = ap.parse_args()

    all_qs = [json.loads(l) for l in args.eval_set.read_text().splitlines() if l]
    pool = [q for q in all_qs if q["type"] in ("ratio", "delta", "compare", "trend")]
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

    results = []
    for check_on in (True, False):
        label = f"guard {'ON' if check_on else 'OFF'} + injection"
        print(f"\n=== {label} ===")
        agent = CorruptingAgent(search, check_contradictions=check_on)
        arm = {"config": label, "questions": []}
        t0 = time.time()
        for i, q in enumerate(questions, 1):
            try:
                trace = agent.run(q["question"])
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] ERROR {q['qid']}: {exc}", file=sys.stderr)
                arm["questions"].append({"qid": q["qid"], "error": str(exc)})
                continue
            injected = agent._injected
            verdict = grade(q, trace.answer)
            corrupted_mag = parse_magnitude(injected.value) if injected else None
            contaminated = False
            if corrupted_mag is not None:
                vals = [v for v, _ in numbers_in(trace.answer)]
                for scale in (1, 1e6, 1e9):
                    if any(abs(abs(v) * scale - corrupted_mag)
                           <= 0.02 * corrupted_mag for v in vals if v):
                        contaminated = True
                        break
            row = {
                "qid": q["qid"], "verdict": verdict,
                "injected": injected is not None,
                "injected_value": injected.value if injected else None,
                "conflicts_detected": len(trace.conflicts),
                "unresolved": sum(1 for c in trace.conflicts if c.unresolved),
                "corrupted_value_in_answer": contaminated,
            }
            arm["questions"].append(row)
            print(f"  [{i}] {verdict:<9} conflicts={row['conflicts_detected']} "
                  f"contaminated={contaminated} {q['qid']}")
        arm["seconds"] = round(time.time() - t0, 1)
        qs = [r for r in arm["questions"] if "error" not in r]
        inj = [r for r in qs if r["injected"]]
        # Report the denominator every rate is over. `correct + wrong + abstain`
        # does not equal `n`: `grade` also returns "ungradeable", and the two
        # arms do not produce the same number of them, so a bare "correct 0 -> 2"
        # compares counts taken over different-sized gradeable sets. Anyone
        # reading the arms has to be able to see that.
        gradeable = [r for r in qs if r["verdict"] in ("correct", "wrong", "abstain")]
        detected = sum(1 for r in inj if r["conflicts_detected"] > 0)
        arm["summary"] = {
            "n": len(qs),
            "errors": len(arm["questions"]) - len(qs),
            "injections": len(inj),
            # None, not 0.0: with no injections the guard was never given
            # anything to catch, which is not the same as failing to catch it.
            "detection_rate": (detected / len(inj)) if inj else None,
            "detected": detected,
            "n_gradeable": len(gradeable),
            "ungradeable": len(qs) - len(gradeable),
            "correct": sum(1 for r in qs if r["verdict"] == "correct"),
            "wrong": sum(1 for r in qs if r["verdict"] == "wrong"),
            "abstain": sum(1 for r in qs if r["verdict"] == "abstain"),
            "contaminated": sum(1 for r in qs if r["corrupted_value_in_answer"]),
            # The questions are generated per (concept, year) from a shared
            # pool, so several of them ask about the same underlying figure in
            # different shapes. That makes them correlated, not independent
            # trials, and the count below is the honest sample size.
            "distinct_facts": len({r["qid"].rsplit("-", 2)[-2:][0] for r in qs}),
        }
        print(f"  summary: {arm['summary']}")
        results.append(arm)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_questions": len(questions),
                                    "corrupt_factor": CORRUPT_FACTOR,
                                    "arms": results}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
