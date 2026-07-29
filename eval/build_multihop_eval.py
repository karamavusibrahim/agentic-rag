#!/usr/bin/env python
"""Build multi-hop questions whose answers are computed, not judged.

Evaluating a multi-hop agent usually means asking a model whether the answer
looks right. That imports the failure being measured: the same class of model
that picked $4,239M over $12,914M is not a reliable judge of whether $4,239M was
the correct pick.

XBRL removes the need for a judge. Every material figure in a 10-K is also
published as a structured fact, so a *derived* quantity -- a ratio, a year-over-
year delta, a cross-company comparison -- can be computed exactly in Python from
facts the agent never sees. The question requires several retrievals to answer;
the answer requires no interpretation to grade.

Four question types, chosen because each needs a different number of hops and
fails differently:

    ratio        one company, two facts, one division      (2 hops)
    delta        one company, one metric, two years        (2 hops)
    compare      two companies, one metric, one year       (2 hops, categorical)
    trend        two companies, two metrics, two years     (4 hops)

`trend` is the shape that produced the original failure, and it is deliberately
over-represented: four hops means four chances to mis-extract, and the ratio
arithmetic hides a wrong input behind a plausible-looking output.

An `unanswerable` control set is generated too -- same phrasing, but for a
fiscal year outside the indexed filings. An agent that scores well on the real
questions and also "answers" these is pattern-matching, not retrieving.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from sec_rag.ingest.edgar import company_facts, iter_us_gaap_facts  # noqa: E402

# Each metric carries how a ratio against revenue should be *worded*. An expense
# is "spent on"; a profit line is a margin. Getting this wrong produces questions
# like "what percentage of revenue was spent on net income", which is not a
# question an analyst would ask -- and an agent that flounders on it has been
# failed by the eval, not caught by it.
METRICS: dict[str, tuple[str, str]] = {
    "ResearchAndDevelopmentExpense": ("research and development expense", "expense"),
    "SellingGeneralAndAdministrativeExpense": (
        "selling, general and administrative expense", "expense"),
    "NetIncomeLoss": ("net income", "margin"),
    "OperatingIncomeLoss": ("operating income", "margin"),
    "GrossProfit": ("gross profit", "margin"),
}


def ratio_question(ticker: str, label: str, kind: str, fy: int) -> str:
    if kind == "expense":
        return (f"What percentage of {ticker}'s total revenue in fiscal {fy} "
                f"was spent on {label}?")
    return f"What was {ticker}'s {label} as a percentage of total revenue in fiscal {fy}?"


REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
)


def load_facts(ticker: str) -> dict[tuple[str, int], float]:
    """(concept, fiscal_year) -> value, from 10-K annual facts only.

    Two filters, each guarding a different way this went wrong:

    **`annual`** excludes quarterly durations. The same concept carries 10-Q
    values, and a quarterly figure substituted for an annual one is precisely the
    error this eval exists to detect -- using unfiltered facts as truth would make
    the eval agree with the bug.

    **`fiscal_year`, not `fy`** -- see `edgar.fact_fiscal_year`. `fy` is the
    fiscal year of the *filing*, so all three comparative columns of a 10-K carry
    the same value, and `setdefault` on it returns whichever the API listed
    first: FY2022's R&D labelled FY2024. This eval scored the agent "wrong" on
    an answer of 8.02% -- which was exactly right -- because gold had been built
    from a two-year-old figure. The measurement was broken, not the agent.
    """
    out: dict[tuple[str, int], float] = {}
    for fact in iter_us_gaap_facts(company_facts(ticker)):
        if fact["form"] != "10-K":
            continue
        # `fiscal_year` is derived from the period end, NOT from `fy`. See
        # edgar.fact_fiscal_year: `fy` is the *filing's* year, so all three
        # comparative columns of a 10-K carry the same `fy`, and keying on it
        # silently returns a two-year-old figure. That bug made this eval report
        # the agent "wrong" for an answer that was arithmetically correct.
        val, fy = fact["value"], fact["fiscal_year"]
        if not isinstance(val, (int, float)) or not fy or not fact["annual"]:
            continue
        out.setdefault((fact["concept"], int(fy)), float(val))
    return out


def revenue(facts: dict[tuple[str, int], float], fy: int) -> float | None:
    for c in REVENUE_CONCEPTS:
        if (c, fy) in facts:
            return facts[(c, fy)]
    return None


def build(tickers: list[str], years: dict[str, list[int]]) -> list[dict[str, Any]]:
    facts = {t: load_facts(t) for t in tickers}
    qs: list[dict[str, Any]] = []

    for t in tickers:
        fys = sorted(years.get(t, []))
        f = facts[t]
        for fy in fys:
            rev = revenue(f, fy)
            for concept, (label, kind) in METRICS.items():
                val = f.get((concept, fy))
                if val is None:
                    continue

                if rev:
                    qs.append({
                        "qid": f"ratio-{t}-{concept}-{fy}",
                        "type": "ratio",
                        "hops": 2,
                        "question": ratio_question(t, label, kind, fy),
                        "answer_numeric": round(100.0 * val / rev, 4),
                        "answer_unit": "percent",
                        "inputs": {f"{label} FY{fy}": val, f"revenue FY{fy}": rev},
                    })

                prev = f.get((concept, fy - 1))
                if prev:
                    qs.append({
                        "qid": f"delta-{t}-{concept}-{fy}",
                        "type": "delta",
                        "hops": 2,
                        "question": f"By what percentage did {t}'s {label} change from "
                                    f"fiscal {fy - 1} to fiscal {fy}?",
                        "answer_numeric": round(100.0 * (val - prev) / abs(prev), 4),
                        "answer_unit": "percent",
                        "inputs": {f"{label} FY{fy}": val, f"{label} FY{fy - 1}": prev},
                    })

    # Cross-company, on years both companies actually filed.
    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            common = sorted(set(years.get(a, [])) & set(years.get(b, [])))
            for fy in common:
                for concept, (label, _kind) in METRICS.items():
                    va, vb = facts[a].get((concept, fy)), facts[b].get((concept, fy))
                    if va is None or vb is None:
                        continue
                    qs.append({
                        "qid": f"compare-{a}{b}-{concept}-{fy}",
                        "type": "compare",
                        "hops": 2,
                        "question": f"In fiscal {fy}, which company reported higher "
                                    f"{label}, {a} or {b}?",
                        "answer_categorical": a if va > vb else b,
                        "inputs": {f"{a} {label}": va, f"{b} {label}": vb},
                    })

                # The four-hop shape that broke the agent originally.
                ra, rb = revenue(facts[a], fy), revenue(facts[b], fy)
                rda, rdb = (facts[a].get(("ResearchAndDevelopmentExpense", fy)),
                            facts[b].get(("ResearchAndDevelopmentExpense", fy)))
                if all(x for x in (ra, rb, rda, rdb)):
                    sa, sb = 100.0 * rda / ra, 100.0 * rdb / rb
                    qs.append({
                        "qid": f"trend-{a}{b}-rd-{fy}",
                        "type": "trend",
                        "hops": 4,
                        "question": f"In fiscal {fy}, did {a} or {b} spend a larger share "
                                    f"of its revenue on research and development?",
                        "answer_categorical": a if sa > sb else b,
                        "answer_numeric": round(abs(sa - sb), 4),
                        "answer_unit": "percentage points difference",
                        "inputs": {f"{a} R&D share": round(sa, 3),
                                   f"{b} R&D share": round(sb, 3)},
                    })

    # Unanswerable controls: real phrasing, fiscal year outside the corpus.
    for t in tickers:
        fys = sorted(years.get(t, []))
        if not fys:
            continue
        for fy in (min(fys) - 6, min(fys) - 8):
            qs.append({
                "qid": f"unanswerable-{t}-{fy}",
                "type": "unanswerable",
                "hops": 2,
                "question": f"What percentage of {t}'s total revenue in fiscal {fy} "
                            f"was spent on research and development expense?",
                "answer_numeric": None,
                "expect_abstain": True,
                "inputs": {},
            })
    return qs


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=Path("../sec-rag/data/processed"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/multihop.jsonl"))
    args = ap.parse_args()

    chunks = [json.loads(l) for l in
              (args.index / "chunks.jsonl").read_text().splitlines() if l]
    years: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        y = int(c["report_date"][:4])
        if y not in years[c["ticker"]]:
            years[c["ticker"]].append(y)
    tickers = sorted(years)
    print(f"corpus covers: { {t: sorted(v) for t, v in years.items()} }")

    qs = build(tickers, dict(years))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")

    by_type: dict[str, int] = defaultdict(int)
    for q in qs:
        by_type[q["type"]] += 1
    print(f"wrote {len(qs)} questions -> {args.out}")
    print(f"  by type: {dict(by_type)}")
    for q in qs[:3]:
        print(f"  e.g. {q['question']}")
        print(f"       gold: {q.get('answer_numeric') or q.get('answer_categorical')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
