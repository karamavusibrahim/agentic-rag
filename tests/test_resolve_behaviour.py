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
