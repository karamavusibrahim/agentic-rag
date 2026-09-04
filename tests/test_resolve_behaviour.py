"""Behavioural tests for the two guards, exercised through the real functions.

An earlier audit added these guards and tested only `value_in_passage`. Every
one of those tests passed with the actual fixes reverted, because none of them
called `resolve()` or the extractor. These do.

Each test here fails if its corresponding guard is removed.
"""

from __future__ import annotations

from unittest.mock import patch

from agentic_rag.agent.contradictions import Conflict, resolve

PASSAGE = (
    "Research and development expense was 12,914 for fiscal 2025 "
    "compared to 8,701 for fiscal 2024 (in millions)."
)


class Hit:
    def __init__(self, cid: str, text: str):
        self.chunk_id, self.text = cid, text

    def citation(self) -> str:
        return f"[{self.chunk_id}]"


class Cand:
    def __init__(self, value, citation="[c1]"):
        self.value, self.citation = value, citation


def run(reply: dict, hits=None) -> Conflict:
    hits = hits or [Hit("c1", PASSAGE)]
    c = Conflict(key=("NVDA", "research and development expense", "fy2025"),
                 candidates=[Cand("$12,914 million"), Cand("$4,239 million")])
    with patch("agentic_rag.agent.contradictions.chat_json_chain",
               return_value=(reply, "model")):
        return resolve(c, lambda q, k: hits, ["m"])


def test_a_grounded_value_resolves():
    c = run({"correct_value": "$12,914 million", "passage_number": 1,
             "reason": "read off the table"})
    assert c.resolved_value == "$12,914 million"
    assert c.resolved_chunk_id == "c1"
    assert not c.unresolved


def test_a_value_absent_from_the_cited_passage_stays_unresolved():
    # The real observed failure: a quarterly figure returned as annual.
    c = run({"correct_value": "$4,239 million", "passage_number": 1,
             "reason": "invented"})
    assert c.unresolved
    assert c.resolved_value is None
    assert "does not appear" in c.reason


def test_a_passage_number_outside_the_shown_range_stays_unresolved():
    c = run({"correct_value": "$12,914 million", "passage_number": 7})
    assert c.unresolved
    assert "did not cite a passage" in c.reason


def test_boolean_true_is_not_accepted_as_passage_one():
    # isinstance(True, int) is True, so this used to select hits[0] silently.
    c = run({"correct_value": "$12,914 million", "passage_number": True})
    assert c.unresolved, "boolean passage_number was accepted as an index"


def test_a_null_correct_value_stays_unresolved():
    c = run({"correct_value": None, "reason": "passages do not settle it"})
    assert c.unresolved


def test_render_only_says_verified_when_it_resolved():
    grounded = run({"correct_value": "$12,914 million", "passage_number": 1})
    assert "verified" in grounded.render()
    ungrounded = run({"correct_value": "$4,239 million", "passage_number": 1})
    assert "verified" not in ungrounded.render()
    assert "UNRESOLVED" in ungrounded.render()


# ---------------------------------------------------------------------------
# The extractor, driven through MultiHopAgent.extract rather than its helpers.
# Reverting any of the three extractor guards left the earlier tests green,
# because none of them called this method.
# ---------------------------------------------------------------------------

from agentic_rag.agent.loop import MultiHopAgent  # noqa: E402


def extract(reply: dict, text: str = PASSAGE):
    hits = [Hit("c1", text)]
    agent = MultiHopAgent(lambda q, k: hits)
    with patch("agentic_rag.agent.loop.chat_json_chain",
               return_value=(reply, "model")):
        return agent.extract("what was R&D expense")


def test_a_normal_extraction_succeeds():
    e = extract({"found": True, "value": "$12,914 million", "passage_number": 1})
    assert e.found and e.value == "$12,914 million" and e.chunk_id == "c1"


def test_the_string_false_is_not_a_successful_extraction():
    # Hosted models return "found": "false" as a string; a truthiness test read
    # that refusal as a positive extraction and rendered the value "False".
    e = extract({"found": "false", "value": "$12,914 million",
                 "passage_number": 1})
    assert not e.found


def test_a_boolean_passage_number_is_rejected():
    e = extract({"found": True, "value": "$12,914 million",
                 "passage_number": True})
    assert not e.found, "isinstance(True, int) let a boolean select hits[0]"


def test_an_out_of_range_passage_number_is_rejected():
    e = extract({"found": True, "value": "$12,914 million", "passage_number": 9})
    assert not e.found


def test_a_numeric_zero_is_a_real_value():
    # `str(value or "")` mapped 0 to empty and discarded the extraction. The
    # passage states the zero, as it must now that extractions are grounded.
    e = extract({"found": True, "value": 0, "passage_number": 1},
                text="Restructuring charges were $0 for fiscal 2025.")
    assert e.found and e.value == "0"


def test_a_figure_absent_from_the_cited_passage_is_discarded():
    """The largest hole the reviews kept open: a real passage cited for a
    figure it does not contain lent the invented figure a source."""
    e = extract({"found": True, "value": "$999 million", "passage_number": 1,
                 "entity": "NVDA", "metric": "R&D", "period": "FY2025"})
    assert not e.found
    assert e.ungrounded == "$999 million"
    assert e.entity == "NVDA" and e.period == "FY2025"


def test_a_textual_extraction_is_not_subjected_to_substring_grounding():
    # A paraphrased textual value would fail a verbatim test; grounding is
    # numeric-only, the same scope as contradiction detection.
    e = extract({"found": True, "value": "supply constraints in the data center segment",
                 "passage_number": 1},
                text="Management cited constraints on data-center supply.")
    assert e.found


def test_extraction_grounding_uses_only_the_text_the_extractor_was_shown():
    from agentic_rag.agent.loop import EXTRACT_CHARS
    text = "x" * EXTRACT_CHARS + " Research and development was 12,914 (in millions)."
    e = extract({"found": True, "value": "$12,914 million", "passage_number": 1},
                text=text)
    assert not e.found, "a figure past the truncation point counted as grounded"


def test_a_resolved_conflict_renders_its_citation():
    c = run({"correct_value": "$12,914 million", "passage_number": 1})
    assert c.resolved_citation == "[c1]"
    assert "[[c1], chunk c1]" in c.render(), c.render()


def test_an_empty_value_is_not_an_extraction():
    e = extract({"found": True, "value": "  ", "passage_number": 1})
    assert not e.found


def test_grounding_uses_only_the_text_the_verifier_was_shown():
    """A value past the truncation point was never seen by the verifier.

    `resolve` shows the model `hit.text[:VERIFY_CHARS]` but used to validate
    against the whole passage, so a figure buried beyond the cut could be
    returned and then "confirmed" by text the model never read.
    """
    from agentic_rag.agent.contradictions import VERIFY_CHARS

    buried = "$100 million " + ("filler " * 400) + " $999 million"
    assert len(buried) > VERIFY_CHARS
    c = run({"correct_value": "$999 million", "passage_number": 1},
            hits=[Hit("c1", buried)])
    assert c.unresolved, "a value beyond the shown window was accepted"
