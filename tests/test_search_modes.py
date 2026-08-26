"""Pure-function tests for the optional relgrep search mode.

No network: only the anchor extraction and grep filter are tested here; the
reranker-ordering step is exercised by the live eval, not unit tests (same
policy as the rest of the suite).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.search_modes import (  # noqa: E402
    company_aliases,
    extract_anchors,
    grep_chunks,
)


def chunk(ticker="AAPL", company="Apple Inc.", breadcrumb="AAPL 10-K FY2024",
          text="Total net sales were $391,035 million."):
    return {"ticker": ticker, "company": company,
            "breadcrumb": breadcrumb, "text": text}


CHUNKS = [
    chunk(),
    chunk(breadcrumb="AAPL 10-K FY2023", text="Total net sales were $383,285."),
    chunk(ticker="MSFT", company="MICROSOFT CORP",
          breadcrumb="MSFT 10-K FY2024", text="Revenue was $245,122 million."),
    chunk(ticker="NVDA", company="NVIDIA CORP",
          breadcrumb="NVDA 10-K FY2025", text="R&D expense grew 35%."),
]

ALIASES = company_aliases(CHUNKS)


class TestAliases:
    def test_ticker_and_name(self):
        assert "aapl" in ALIASES["AAPL"]
        assert "apple" in ALIASES["AAPL"]
        assert "microsoft" in ALIASES["MSFT"]
        assert "nvidia" in ALIASES["NVDA"]


class TestExtractAnchors:
    def test_company_by_name_and_year(self):
        tickers, years, soft = extract_anchors(
            "What were Apple's total net sales in FY2024?", ALIASES)
        assert tickers == {"AAPL"}
        assert years == {"2024"}
        assert "sales" in soft
        assert "apple" not in soft  # alias words are not soft terms

    def test_company_by_ticker(self):
        tickers, _, _ = extract_anchors("NVDA R&D expense FY2025", ALIASES)
        assert tickers == {"NVDA"}

    def test_no_literal_anchors(self):
        tickers, years, _ = extract_anchors(
            "Which segment grew fastest?", ALIASES)
        assert not tickers and not years

    def test_bare_year(self):
        _, years, _ = extract_anchors("revenue in 2023", ALIASES)
        assert years == {"2023"}


class TestGrepChunks:
    def test_conjunctive_company_and_year(self):
        got = grep_chunks(CHUNKS, {"AAPL"}, {"2024"}, set())
        assert len(got) == 1
        assert got[0]["breadcrumb"] == "AAPL 10-K FY2024"

    def test_year_matches_text_too(self):
        # year in chunk text (comparative column) counts, not only breadcrumb
        chunks = [chunk(breadcrumb="AAPL 10-K FY2025",
                        text="FY2024 revenue was $391,035.")]
        assert grep_chunks(chunks, {"AAPL"}, {"2024"}, set())

    def test_company_only(self):
        got = grep_chunks(CHUNKS, {"MSFT"}, set(), set())
        assert [c["ticker"] for c in got] == ["MSFT"]

    def test_wrong_company_excluded(self):
        assert grep_chunks(CHUNKS, {"MSFT"}, {"2025"}, set()) == []

    def test_soft_terms_order_large_pools(self):
        many = [chunk(text=f"filler paragraph {i}") for i in range(150)]
        many[120] = chunk(text="Total net sales and revenue details.")
        got = grep_chunks(many, {"AAPL"}, set(), {"revenue"})
        assert got[0]["text"] == "Total net sales and revenue details."
        assert len(got) == 100  # RERANK_POOL cap
