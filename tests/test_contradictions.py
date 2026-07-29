"""Regression tests for contradiction detection.

The headline case is the real failure this module was written for: the agent
extracted NVDA FY2025 R&D as $4,239M in one hop having already retrieved the
correct ~$12,914M in another, then resolved the conflict by picking the wrong
one. Detection must fire on that pair -- and must NOT fire on figures that merely
differ by filing-level rounding.
"""

from __future__ import annotations

import pytest

from agentic_rag.agent.contradictions import (
    find_conflicts,
    period_mismatch,
    period_year,
    group_key,
    parse_magnitude,
    values_agree,
)
from agentic_rag.agent.loop import Evidence


def ev(value, *, entity="NVDA", metric="research and development expense",
       period="FY2025", citation="NVDA 10-K 2025, Item 7", chunk="c1"):
    return Evidence("q", True, value, None, chunk, citation,
                    entity=entity, metric=metric, period=period)


class TestParseMagnitude:
    @pytest.mark.parametrize("text,expected", [
        ("$12,914 million", 12.914e9),
        ("12.9 billion", 12.9e9),
        ("$4,239 million", 4.239e9),
        ("60,922 million", 60.922e9),
        ("$1.2B", 1.2e9),
        ("245,122", 245122.0),
    ])
    def test_scales(self, text, expected):
        assert parse_magnitude(text) == pytest.approx(expected, rel=1e-9)

    def test_parenthesized_negative(self):
        assert parse_magnitude("(1,234)") == -1234.0

    def test_no_number(self):
        assert parse_magnitude("not a figure") is None
        assert parse_magnitude(None) is None


class TestValuesAgree:
    def test_filing_rounding_is_not_a_conflict(self):
        # 60,922 vs 60,925 for the same line item across two statements.
        assert values_agree(60.922e9, 60.925e9)

    def test_unit_normalization(self):
        assert values_agree(parse_magnitude("12.9 billion"),
                            parse_magnitude("12,900 million"))

    def test_the_real_bug_is_a_conflict(self):
        assert not values_agree(parse_magnitude("$12,914 million"),
                                parse_magnitude("$4,239 million"))

    def test_missing_value_never_agrees(self):
        assert not values_agree(None, 1.0)


class TestGrouping:
    def test_ticker_and_full_name_group_together(self):
        """The extractor writes whatever the passage said; the citation is
        machine-built, so the ticker comes from there."""
        a = group_key(ev("x", entity="NVDA"))
        b = group_key(ev("x", entity="NVIDIA Corporation"))
        assert a == b

    @pytest.mark.parametrize("period", ["FY2025", "fiscal year 2025", "2025",
                                        "fiscal 2025", "year ended 2025"])
    def test_period_spellings_collapse(self, period):
        assert group_key(ev("x", period=period))[2] == "2025"

    def test_metric_wording_collapses(self):
        a = group_key(ev("x", metric="research and development expense"))
        b = group_key(ev("x", metric="Research and development"))
        assert a == b

    def test_different_years_do_not_group(self):
        a = group_key(ev("x", period="FY2024"))
        b = group_key(ev("x", period="FY2025"))
        assert a != b

    def test_missing_fields_yield_none(self):
        assert group_key(Evidence("q", True, "1", None, "c", None)) is None


class TestFindConflicts:
    def test_detects_the_observed_failure(self):
        evidence = [
            ev("$12,914 million", entity="NVDA", chunk="c1"),
            ev("$4,239 million", entity="Nvidia", period="fiscal year 2025", chunk="c2"),
        ]
        conflicts = find_conflicts(evidence)
        assert len(conflicts) == 1
        assert {c.value for c in conflicts[0].candidates} == {
            "$12,914 million", "$4,239 million"}

    def test_rounding_difference_is_not_reported(self):
        evidence = [
            ev("60,922 million", metric="revenue"),
            ev("60,925 million", metric="revenue", chunk="c2"),
        ]
        assert find_conflicts(evidence) == []

    def test_distinct_figures_are_not_conflated(self):
        evidence = [
            ev("$12,914 million", entity="NVDA", citation="NVDA 10-K 2025"),
            ev("$245,122 million", entity="MSFT", metric="total revenue",
               period="FY2024", citation="MSFT 10-K 2024"),
        ]
        assert find_conflicts(evidence) == []

    def test_not_found_evidence_is_ignored(self):
        evidence = [
            ev("$12,914 million"),
            Evidence("q", False, None, None, None, "NVDA 10-K 2025",
                     entity="NVDA", metric="research and development expense",
                     period="FY2025"),
        ]
        assert find_conflicts(evidence) == []


class TestPeriodMismatch:
    """The failure contradiction detection cannot see.

    Asked for AAPL's fiscal 2018 ratio -- a year outside the corpus -- the agent
    returned AAPL's genuine FY2024 figures relabelled "fiscal 2018", with real
    citations to real chunks and correct arithmetic. Every extraction agreed, so
    there was nothing for a cross-check to disagree about.
    """

    def test_the_observed_fabrication_is_caught(self):
        assert period_mismatch(
            "What was AAPL's research and development expense in fiscal 2018?",
            "FY2024")

    @pytest.mark.parametrize("stated", ["FY2025", "fiscal year 2025", "2025",
                                        "year ended January 2025"])
    def test_matching_years_pass(self, stated):
        assert not period_mismatch("NVDA revenue for fiscal 2025", stated)

    def test_missing_year_is_not_a_mismatch(self):
        """Conservative: absence of evidence is not evidence of mismatch."""
        assert not period_mismatch("NVDA revenue for the most recent year", "FY2025")
        assert not period_mismatch("NVDA revenue fiscal 2025", None)
        assert not period_mismatch(None, None)

    def test_adjacent_years_still_mismatch(self):
        """Off-by-one is the common form: reading the comparative column."""
        assert period_mismatch("MSFT operating income fiscal 2024", "FY2025")

    def test_period_year_extracts_first_year(self):
        assert period_year("FY2025") == "2025"
        assert period_year("no year here") is None


class TestPeriodMismatchCompound:
    """Fixes from the 2026-07-29 audit: the guard judged period strings by
    their FIRST year only, rejecting correct extractions whenever the planner
    emitted a compound sub-question or NVDA's two-calendar-year fiscal phrasing
    appeared -- a plausible structural contributor to conflicts=0."""

    def test_compound_request_answered_for_one_of_its_years(self):
        assert not period_mismatch(
            "NVDA's R&D spending for fiscal years 2024 and 2025", "FY2025")

    def test_delta_request_answered_for_the_later_year(self):
        assert not period_mismatch(
            "change from fiscal 2023 to fiscal 2024", "FY2024")

    def test_fiscal_range_phrasing_is_not_a_mismatch(self):
        assert not period_mismatch(
            "NVDA revenue for fiscal 2025", "February 2024 to January 2025")

    def test_disjoint_years_still_fire(self):
        assert period_mismatch("fiscal 2018 R&D ratio", "FY2024")
        assert period_mismatch("fiscal years 2017 and 2018", "FY2024 and FY2025")


class TestNonNumericValuesDoNotConflict:
    def test_identical_textual_claims_are_not_a_disagreement(self):
        pair = [ev("cited supply constraints"), ev("cited supply constraints")]
        assert find_conflicts(pair) == []

    def test_numeric_disagreement_still_fires_with_a_textual_bystander(self):
        trio = [ev("$12,914 million"), ev("$4,239 million"),
                ev("management commentary")]
        assert len(find_conflicts(trio)) == 1
