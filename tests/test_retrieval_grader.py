"""Offline tests for the optional corrective-retrieval grader.

The grader and the re-query are injected; what is tested is the action
logic -- keep, supplement, replace, or fall back -- and that a failed or
malformed grade never costs the hop its passages.
"""

from __future__ import annotations

from unittest.mock import patch

from agentic_rag.agent.loop import MultiHopAgent
from agentic_rag.agent.retrieval_grader import correct_retrieval, grade_passages


class Hit:
    def __init__(self, cid: str, text: str = "text"):
        self.chunk_id, self.text = cid, text

    def citation(self) -> str:
        return f"[{self.chunk_id}]"


def chain_of(*replies):
    """A chat_json_chain stub returning the given replies in order."""
    queue = list(replies)

    def chain(models, messages, **kw):
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, "m"
    return chain


def run(grades, *, more=(), regrades=None, reformulated=("rewritten q",)):
    hits = [Hit(f"c{i}") for i in range(len(grades))]
    replies = [{"grades": list(grades)}]
    if regrades is not None:
        replies.append({"grades": list(regrades)})
    return correct_retrieval(
        "q", hits, models=["m"],
        search=lambda q, k: [Hit(c) for c in more],
        reformulate=lambda qs: list(reformulated),
        k=5, chat_json_chain=chain_of(*replies)), hits


class TestGradePassages:
    def test_malformed_length_or_label_is_a_failed_grade(self):
        hits = [Hit("a"), Hit("b")]
        assert grade_passages("q", hits, ["m"],
                              chat_json_chain=chain_of({"grades": ["relevant"]})) is None
        assert grade_passages("q", hits, ["m"],
                              chat_json_chain=chain_of({"grades": ["good", "bad"]})) is None
        assert grade_passages("q", hits, ["m"], chat_json_chain=chain_of(
            {"grades": ["Relevant", " irrelevant "]})) == ["relevant", "irrelevant"]

    def test_a_raising_call_is_a_failed_grade(self):
        assert grade_passages("q", [Hit("a")], ["m"],
                              chat_json_chain=chain_of(RuntimeError("410"))) is None


class TestActions:
    def test_correct_keeps_relevant_and_ambiguous_drops_irrelevant(self):
        corr, hits = run(["irrelevant", "relevant", "ambiguous"])
        assert corr.action == "correct"
        assert [h.chunk_id for h in corr.hits] == ["c1", "c2"]
        assert corr.requery is None

    def test_ambiguous_keeps_and_supplements_with_a_requery(self):
        corr, _ = run(["ambiguous", "irrelevant"], more=["x", "c0"])
        assert corr.action == "ambiguous"
        assert [h.chunk_id for h in corr.hits] == ["c0", "x"], "dedupe by chunk id"
        assert corr.requery == "rewritten q"

    def test_incorrect_replaces_with_a_graded_requery(self):
        corr, _ = run(["irrelevant", "irrelevant"], more=["x", "y"],
                      regrades=["irrelevant", "relevant"])
        assert corr.action == "incorrect"
        assert [h.chunk_id for h in corr.hits] == ["y"]

    def test_incorrect_falls_back_to_the_original_hits_when_no_better(self):
        corr, hits = run(["irrelevant"], more=["x"], regrades=["irrelevant"])
        assert corr.action == "incorrect"
        assert corr.hits == hits
        assert "kept" in corr.note

    def test_a_requery_identical_to_the_question_is_not_searched(self):
        calls = []
        corr = correct_retrieval(
            "q", [Hit("a")], models=["m"],
            search=lambda q, k: calls.append(q) or [],
            reformulate=lambda qs: ["q"], k=5,
            chat_json_chain=chain_of({"grades": ["irrelevant"]}))
        assert calls == [] and corr.hits[0].chunk_id == "a"

    def test_a_failed_grader_skips_and_keeps_everything(self):
        corr = correct_retrieval(
            "q", [Hit("a"), Hit("b")], models=["m"],
            search=lambda q, k: [], reformulate=lambda qs: [], k=5,
            chat_json_chain=chain_of(RuntimeError("down")))
        assert corr.action == "skipped" and len(corr.hits) == 2


class TestThroughTheAgent:
    PASSAGE = "Research and development was 12,914 for fiscal 2025 (in millions)."

    def test_disabled_by_default_and_recorded_when_enabled(self):
        good = Hit("good", self.PASSAGE)
        bad = Hit("bad", "Unrelated prose.")
        agent = MultiHopAgent(lambda q, k: [bad, good], retrieval_grader=True)
        # The grader is handed the loop's chat_json_chain, so one patch with
        # the full reply sequence (grade first, then extraction) covers both.
        replies = [({"grades": ["irrelevant", "relevant"]}, "m"),
                   ({"found": True, "value": "$12,914 million",
                     "passage_number": 1}, "m")]
        with patch("agentic_rag.agent.loop.chat_json_chain", side_effect=replies):
            e = agent.extract("what was R&D in fiscal 2025")
        assert e.found and e.chunk_id == "good", "passage 1 is the graded-in hit"
        assert e.retrieval_action == "correct"

        plain = MultiHopAgent(lambda q, k: [good])
        with patch("agentic_rag.agent.loop.chat_json_chain",
                   return_value=({"found": True, "value": "$12,914 million",
                                  "passage_number": 1}, "m")):
            assert plain.extract("q").retrieval_action is None
