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


class TestHallucinationShapes:
    """The three shapes that got past the first grounding check, pinned.

    An independent review substituted the pre-fix grounder in memory and every
    test in this repo stayed green while all three of these returned True.
    These fail against that version.
    """

    def test_zero_is_not_grounded_by_any_character_zero(self):
        assert not value_in_passage("$0", "Fiscal 2024 revenue was $100 million")
        assert value_in_passage("$0", "Net charges were $0 this year")

    def test_an_explicit_scale_is_never_rescaled(self):
        assert not value_in_passage("$12.914 trillion",
                                    "Revenue was $12.914 billion")

    def test_a_fiscal_year_is_not_a_dollar_figure(self):
        assert not value_in_passage("$2.024 million", "FY2024 revenue")
        assert not value_in_passage("$2.024 million",
                                    "fiscal 2024 revenue (in millions)")

    def test_implicit_scaling_requires_a_caption(self):
        # "100 employees" must not ground "$100 million"; the same digits under
        # an explicit caption must.
        assert not value_in_passage(
            "$100 million",
            "Revenue was $200 million; the company had 100 employees.")
        assert value_in_passage("$100 million",
                                "(in millions) Total revenue 100")

    def test_a_year_ending_a_sentence_is_still_a_year(self):
        # _NUM_RE captures "2024." from "fiscal 2024. Revenue...", and an
        # untrimmed token slipped past the year guard -- so the guard failed on
        # exactly the years that end a sentence, which is where filing prose
        # puts them.
        assert not value_in_passage("$2.024 billion",
                                    "In fiscal 2024. Revenue grew (in millions).")

    def test_a_comma_formatted_number_is_not_a_year(self):
        # "2,024" is a figure that happens to fall in 1900-2099; stripping the
        # comma before the year test rejected it under an explicit caption.
        assert value_in_passage("$2.024 billion",
                                "Revenue (in millions): 2,024")

    def test_a_caption_reaches_its_own_sentence_only(self):
        # A chunk can caption one table in thousands and the next in millions.
        # Each caption grounds the numbers in its sentence...
        assert value_in_passage(
            "$2 billion",
            "Table A (in thousands): 1. Table B (in millions): Revenue 2,000")
        # ...and must not reach across the boundary: Table A's employee count
        # scaled by Table B's caption is not three billion dollars of revenue.
        assert not value_in_passage(
            "$3 billion",
            "Table A (in thousands): Employees 3,000. "
            "Table B (in millions): Revenue 1")
        # A trailing caption still covers the numbers of its own sentence.
        assert value_in_passage("$12,914 million",
                                "R&D was 12,914 compared to 8,701 (in millions).")

    def test_a_year_shape_without_fiscal_context_is_a_figure(self):
        # "Revenue (in millions): 2024" is $2.024 billion. Rejecting every
        # 1900-2099 token threw away real values; only fiscal context makes a
        # four-digit token a date.
        assert value_in_passage("$2.024 billion", "Revenue (in millions): 2024")


class TestCaptionScope:
    """A caption governs what follows it until superseded, plus its sentence."""

    def test_a_caption_in_an_earlier_sentence_still_governs(self):
        # "Figures are in millions." then the numbers -- the standard filing
        # pattern that sentence-only scoping broke.
        assert value_in_passage("$2.024 billion",
                                "Figures are in millions. Revenue was 2,024.")

    def test_fiscal_context_survives_intervening_words(self):
        # "for the fiscal year ended 2024" and "year ending December 31, 2024"
        # are dates; the keyword sits a few tokens back from the year.
        assert not value_in_passage(
            "$2.024 billion", "For the fiscal year ended 2024 (in millions).")
        assert not value_in_passage(
            "$2.024 billion", "year ending December 31, 2024 (in millions)")


class TestScopeBoundaries:
    """The two boundary cases the sixth-pass review found, pinned."""

    def test_an_ordinary_word_gap_is_not_fiscal_context(self):
        # "For the year, revenue was 2024" is a value; only date-ish tokens
        # (ended/ending/months/day numbers) may bridge keyword and year.
        assert value_in_passage("$2.024 billion",
                                "For the year, revenue was 2024 (in millions).")
        assert not value_in_passage("$2.024 billion",
                                    "For fiscal year ended: 2024 (in millions).")

    def test_a_new_table_heading_ends_a_caption_reach(self):
        # Table B without its own caption is a new scope; Table A's scale must
        # not turn its headcount into billions.
        assert not value_in_passage(
            "$3 billion",
            "Table A (in millions): Revenue 1. Table B: Employees 3,000.")

    def test_a_document_level_caption_reaches_past_table_headings(self):
        # The seventh-pass rule ended every caption at the next heading and
        # rejected the standard "Amounts are in millions. Table 1: ..." form.
        assert value_in_passage("$2.024 billion",
                                "Amounts are in millions. Table 1: Revenue 2,024.")
        assert value_in_passage("$2.024 billion",
                                "Figures are in millions. Table B: Revenue 2,024.")

    def test_common_date_connectives_bridge_keyword_and_year(self):
        for passage in (
            "For the fiscal year ended on December 31, 2024 (in millions).",
            "For the fiscal year ended December 31st, 2024 (in millions).",
            "as of the end of the fiscal year 2024 (in millions).",
        ):
            assert not value_in_passage("$2.024 billion", passage), passage
        # ...while a non-date word still breaks the bridge:
        assert value_in_passage("$2.024 billion",
                                "For the year, revenue was 2024 (in millions).")

    def test_line_breaks_bound_a_caption_sentence(self):
        # Newline-separated tables are the common shape of a serialised
        # filing chunk; without newline boundaries the whole passage was one
        # "sentence" and every caption applied to every number.
        assert not value_in_passage(
            "$3B",
            "Table A (in thousands)\nEmployees 3,000\nTable B (in millions)\nRevenue 1")
        assert value_in_passage(
            "$2.024 billion",
            "Table A (in thousands)\nEmployees 3,000\nTable B (in millions)\nRevenue 2,024")
