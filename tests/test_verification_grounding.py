"""Regression tests for the grounding of "verified" values and extractions.

The contradiction guard's whole value is that it replaces two disagreeing
figures with one that is checked against the source. Before these tests, three
paths let an *unchecked* figure through wearing that label:

  1. `resolve` accepted whatever `correct_value` the verifier returned, without
     confirming the figure appears in the passage the verifier cited -- so a
     hallucinated number entered synthesis rendered as "(verified)".
  2. `resolve` and the extractor both fell back to `hits[0]` when the model gave
     a `passage_number` outside the range shown to it, attaching a real citation
     to a value that may have come from somewhere else entirely.
  3. The extractor tested `found` for truthiness, so the string "false" -- which
     hosted models return regularly -- read as a successful extraction.
"""

from __future__ import annotations

from agentic_rag.agent.contradictions import value_in_passage

PASSAGE = (
    "Research and development expense was 12,914 for fiscal 2025 compared to "
    "8,701 for fiscal 2024 (in millions)."
)


def test_value_present_at_a_different_scale_counts_as_grounded():
    # The passage prints "12,914" under an "in millions" caption; the extractor
    # reports "$12,914 million". Same figure, different presentation.
    assert value_in_passage("$12,914 million", PASSAGE)
    assert value_in_passage("12.914 billion", PASSAGE)


def test_the_quarterly_for_annual_error_is_not_grounded():
    # $4,239M is the real observed failure -- a quarterly figure reported as
    # annual. It appears nowhere in the passage, so it must not be "verified".
    assert not value_in_passage("$4,239 million", PASSAGE)


def test_a_near_miss_is_not_grounded():
    # One digit off is the shape a hallucinated figure usually takes.
    assert not value_in_passage("$13,914 million", PASSAGE)


def test_non_numeric_values_fall_back_to_substring():
    assert value_in_passage("Americas", "Our Americas segment grew 12%.")
    assert not value_in_passage("Europe", "Our Americas segment grew 12%.")


def test_empty_value_is_never_grounded():
    assert not value_in_passage("", PASSAGE)
