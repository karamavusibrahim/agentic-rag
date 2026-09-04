"""Regression tests for the answer grader.

Every case here is a confirmed failure from the 2026-07-29 audit, each of which
made the *measurement* wrong while the agent's behavior was fine (or vice
versa). The worst: "See the 10-K filing." graded correct against a percent
gold, because "10" from "10-K" sat inside the absolute tolerance. An eval whose
point is catching fabricated answers must not be satisfiable by a citation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_agent_eval import (  # noqa: E402
    abstained,
    grade,
    grade_categorical,
    matches,
    numbers_in,
)

RATIO_Q = {"qid": "ratio-NVDA-RD-2025", "type": "ratio",
           "question": "What was NVDA's R&D as a share of revenue in fiscal 2025?",
           "answer_numeric": 9.896, "answer_unit": "percent"}
DELTA_NEG_Q = {"qid": "delta-AAPL-NI-2024", "type": "delta",
               "question": "How did AAPL's net income change from fiscal 2023 to 2024?",
               "answer_numeric": -3.36, "answer_unit": "percent"}
COMPARE_Q = {"qid": "compare-NI-2023", "type": "compare",
             "question": "Which reported higher net income in fiscal 2023, AAPL or MSFT?",
             "answer_categorical": "MSFT"}


class TestCitationTokensAreNotFigures:
    def test_bare_citation_answer_is_wrong_not_correct(self):
        assert grade(RATIO_Q, "See the 10-K filing.") == "wrong"

    def test_bracketed_citation_digits_do_not_match(self):
        assert grade(RATIO_Q, "The figure is reported "
                     "[AAPL 10-K 2024, Item 8].") == "wrong"

    def test_a_real_percent_figure_still_matches(self):
        assert grade(RATIO_Q, "R&D was 9.9% of revenue "
                     "[NVDA 10-K 2025, Item 8].") == "correct"

    def test_unqualified_counts_do_not_satisfy_percent_golds(self):
        # 8 business segments is within 2% relative of a gold of 8.02 -- but it
        # is not stated as a percentage, so it must not score.
        q = dict(RATIO_Q, answer_numeric=8.02)
        assert grade(q, "AAPL operates 8 business segments.") == "wrong"


class TestNegativeDeltas:
    def test_direction_word_phrasing_grades_correct(self):
        assert grade(DELTA_NEG_Q,
                     "Net income decreased by 3.36% year over year.") == "correct"

    def test_unicode_minus_grades_correct(self):
        assert grade(DELTA_NEG_Q, "The change was −3.36%.") == "correct"

    def test_ascii_minus_still_works(self):
        assert grade(DELTA_NEG_Q, "The change was -3.36%.") == "correct"


class TestAbstainOrdering:
    def test_complete_answer_with_hedge_grades_on_conclusion(self):
        answer = ("MSFT reported higher net income than AAPL in fiscal 2023. "
                  "Note the filings do not disclose segment-level net income.")
        assert grade(COMPARE_Q, answer) == "correct"

    def test_cannot_be_calculated_counts_as_abstention(self):
        assert abstained("The ratio cannot be calculated from the "
                         "indexed filings.")

    def test_pure_abstention_still_grades_abstain(self):
        assert grade(RATIO_Q, "This cannot be determined from the "
                     "available filings.") == "abstain"


class TestCategoricalVerdicts:
    def test_question_echo_is_not_a_verdict(self):
        answer = ("The question asks which company reported higher net income, "
                  "AAPL or MSFT. Based on the evidence, MSFT reported higher "
                  "net income.")
        assert grade_categorical(answer, "MSFT", "AAPL") == "correct"

    def test_inverse_cue_names_the_smaller_company_first(self):
        assert grade_categorical("AAPL's net income was lower than MSFT's.",
                                 "MSFT", "AAPL") == "correct"

    def test_negated_comparative_flips_the_claim(self):
        assert grade_categorical("AAPL did not report higher net income than "
                                 "MSFT.", "MSFT", "AAPL") == "correct"

    def test_no_verdict_sentence_is_ungradeable(self):
        assert grade_categorical("Both companies filed 10-Ks in 2023.",
                                 "MSFT", "AAPL") == "ungradeable"


class TestMatchesToleranceShape:
    def test_percent_gold_requires_percent_context(self):
        assert not matches(9.896, [(10.0, False)], unit="percent")
        assert matches(9.9, [(9.9, True)], unit="percent")

    def test_non_percent_gold_keeps_relative_tolerance(self):
        assert matches(31370.0, [(31370.0, False)], unit="USD millions")
        assert matches(31370.0, [(31000.0, False)], unit="USD millions")


class TestNumbersIn:
    def test_citation_and_form_digits_are_excluded(self):
        vals = [v for v, _ in numbers_in("See Item 8 of the 10-K "
                                         "[AAPL 10-K 2024].")]
        assert vals == []

    def test_downward_sentence_contributes_negated_twin(self):
        vals = [v for v, _ in numbers_in("Revenue fell by 5.2% in 2023.")]
        assert -5.2 in vals and 5.2 in vals


class TestComparisonObjects:
    """A ticker inside "compared with X" / "than X" is the object of the
    comparison, not the subject of the claim."""

    def test_compared_with_names_the_other_company_first(self):
        assert grade_categorical("Compared with MSFT, AAPL reported higher net income.",
                                 "AAPL", "MSFT") == "correct"
        assert grade_categorical("Compared with MSFT, AAPL reported higher net income.",
                                 "MSFT", "AAPL") == "wrong"

    def test_than_phrase_does_not_move_the_subject(self):
        assert grade_categorical("AAPL's net income was higher than MSFT's.",
                                 "AAPL", "MSFT") == "correct"
        assert grade_categorical("Relative to AAPL, MSFT had lower net income.",
                                 "AAPL", "MSFT") == "correct"


class TestNullDenominator:
    """The null control divides by the answerable numeric questions the
    accuracy is computed on, not by every question."""

    def test_controls_and_categorical_questions_are_excluded(self):
        from run_agent_eval import run_config

        class Trace:
            def __init__(self, answer):
                self.answer, self.evidence, self.conflicts = answer, [], []
                self.verification = None

        class Agent:
            def run(self, question):
                return Trace("The figure is 9.9%, and also 12.5% and 42.")

        qs = [
            {"qid": "a", "type": "ratio", "question": "?", "answer_numeric": 9.9,
             "answer_unit": "percent"},
            {"qid": "b", "type": "ratio", "question": "?", "answer_numeric": 12.5,
             "answer_unit": "percent"},
            {"qid": "c", "type": "unanswerable", "question": "?",
             "expect_abstain": True, "answer_numeric": 42.0},
            {"qid": "d", "type": "compare", "question": "AAPL or MSFT?",
             "answer_categorical": "AAPL"},
        ]
        r = run_config(Agent(), qs, "t", None)
        assert r["null_hit_denominator"] == 2
        assert r["null_hits"] == 2          # a matches b's gold and vice versa
        assert r["null_hit_rate"] == 1.0
        assert r["n"] == 4


class TestErrorSurfacing:
    """An extraction that failed on an API error must stay distinguishable from
    a model honestly reporting found=false; otherwise an outage-wide abstain
    sweep passes the unanswerable controls vacuously."""

    def test_evidence_carries_the_error(self):
        from agentic_rag.agent.loop import Evidence
        ev = Evidence("q", False, None, None, None, None, error="HTTP 410")
        assert ev.error == "HTTP 410"
        assert Evidence("q", False, None, None, None, None).error is None

    def test_empty_answer_is_not_wrong_and_not_fabricated(self):
        assert grade(RATIO_Q, "") == "empty_answer"
        assert grade({"qid": "u", "type": "unanswerable", "question": "x",
                      "expect_abstain": True}, "  \n") == "empty_answer"
