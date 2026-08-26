#!/usr/bin/env python
"""Ask a multi-hop question over the SEC corpus built by the sec-rag project.

    uv run python scripts/ask.py "Did NVDA's R&D spend as a share of revenue \
rise faster than Microsoft's between FY2024 and FY2025?"

Requires an index built by sec-rag:
    cd ../sec-rag && uv run python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from agentic_rag.agent.loop import MultiHopAgent  # noqa: E402
from agentic_rag.search_modes import make_search  # noqa: E402
from sec_rag.index.build import load as load_index  # noqa: E402
from sec_rag.retrieve.hybrid import Retriever  # noqa: E402

DEFAULT_INDEX = Path("../sec-rag/data/processed")


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--per-hop-k", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--search-mode", choices=("dense", "relgrep"), default="dense",
                    help="relgrep = optional relevance-guided corpus grep "
                         "(arXiv 2607.24223)")
    ap.add_argument("--trace", type=Path, help="write the full trace as JSON")
    args = ap.parse_args()

    question = " ".join(args.question)

    if not (args.index / "meta.json").exists():
        print(
            f"no index at {args.index}\n"
            "build one first:  cd ../sec-rag && uv run python scripts/build_index.py",
            file=sys.stderr,
        )
        return 1

    index = load_index(args.index)
    retriever = Retriever(index)
    search = make_search(retriever, mode=args.search_mode, candidates=40)

    agent = MultiHopAgent(search, per_hop_k=args.per_hop_k, max_rounds=args.max_rounds)

    print(f"Q: {question}\n{'-' * 70}")
    trace = agent.run(question, verbose=True)

    print(f"\n{'=' * 70}\nANSWER\n{'=' * 70}\n{trace.answer}")
    print(f"\n[{trace.model_calls} model calls, {len(trace.evidence)} evidence items, "
          f"{sum(1 for e in trace.evidence if e.found)} found]")

    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(trace.to_dict(), indent=2))
        print(f"trace -> {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
