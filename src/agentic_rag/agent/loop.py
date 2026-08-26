"""Multi-hop RAG agent: decompose, retrieve per sub-question, verify, synthesize.

Single-shot RAG answers "What was NVDA's FY2025 revenue?" fine. It fails on
"Did NVDA's R&D spend as a share of revenue rise faster than Microsoft's between
FY2024 and FY2025?" -- because no single passage contains the answer. That
question needs four retrievals, two divisions, and a comparison.

The loop:

    plan      -> decompose into atomic, independently-retrievable sub-questions
    retrieve  -> one hybrid retrieval per sub-question (parallel-safe)
    extract   -> pull the specific value/claim from each result set, with a
                 citation and an explicit "not found" option
    critique  -> check the collected evidence actually answers the question;
                 emit follow-up sub-questions if it does not
    synthesize-> compose the final answer from verified evidence only

Two design choices worth calling out:

**Extraction is separated from synthesis.** The extractor sees one sub-question
and its passages, and returns a value plus the chunk it came from. The
synthesizer sees only extracted values, never raw passages. This means the final
answer cannot silently invent a number that was not extracted, and every figure
in it is traceable to a chunk id. It costs an extra model call per hop and is
worth it.

**"Not found" is a first-class answer.** Each extractor call may return
`found: false`. The critique step then decides whether to reformulate or to
report the gap. An agent that cannot say "the filings don't state this" will
fabricate instead.

Iteration is bounded (`max_rounds`) because a self-correcting loop without a
budget is a way to spend money slowly.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..nvidia import chat, chat_json, chat_json_chain
from .contradictions import Conflict, find_conflicts, period_mismatch, resolve

# Model availability on build.nvidia.com is not stable, and the failure mode is
# abrupt: `qwen/qwen3-next-80b-a3b-instruct` and `qwen/qwen3.5-397b-a17b` both
# returned HTTP 410 "reached its end of life on 2026-07-27" -- mid-development,
# with no deprecation warning beforehand. `moonshotai/kimi-k2.6` 404s for this
# account and `google/gemma-4-31b-it` timed out at 45s. This is precisely why
# every model call here goes through a validating fallback chain rather than a
# single hard-coded id. Probe with scripts/probe_models.py before assuming.
#
# Ordered by observed JSON reliability (probed 2026-07-26).
PLANNER_MODELS = (
    "deepseek-ai/deepseek-v4-flash-0731",       # cleanest raw JSON of the live set
    "nvidia/nemotron-3-super-120b-a12b",   # strong, needs reasoning suppression
    "openai/gpt-oss-120b",                 # emits into reasoning_content
)
EXTRACTOR_MODELS = (
    "deepseek-ai/deepseek-v4-flash-0731",
    "nvidia/nemotron-3-nano-30b-a3b",      # cheap; fine for single-fact extraction
    "nvidia/nemotron-3-super-120b-a12b",
)

# Concurrency for per-hop extraction. Bounded rather than unlimited because the
# hosted API rate-limits, and a burst that trips a 429 costs more in retries than
# the parallelism saves.
MAX_HOP_WORKERS = 4


@dataclass
class Evidence:
    sub_question: str
    found: bool
    value: str | None
    quote: str | None
    chunk_id: str | None
    citation: str | None
    # Structured identity of what this figure is *about*, used to group
    # extractions for contradiction detection. String-matching sub-questions
    # would not work: two very differently worded questions can mean the same
    # thing, while two near-identical ones can differ in the single token that
    # matters (FY2024 vs FY2025).
    entity: str | None = None
    metric: str | None = None
    period: str | None = None
    # Set when the extraction FAILED (API error), as opposed to the model
    # honestly reporting found=false. The two must stay distinguishable: an
    # outage that turns every hop into "not found" makes the agent abstain
    # everywhere, and an eval that cannot see the difference scores that
    # outage as honesty.
    error: str | None = None

    def render(self) -> str:
        if not self.found:
            return f"- {self.sub_question} -> NOT FOUND in the corpus"
        return (
            f"- {self.sub_question} -> {self.value}"
            f"  [{self.citation}, chunk {self.chunk_id}]"
        )


@dataclass
class Trace:
    """Everything the agent did, for inspection and eval."""
    question: str
    rounds: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    answer: str = ""
    model_calls: int = 0
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "rounds": self.rounds,
            "evidence": [e.__dict__ for e in self.evidence],
            "conflicts": [
                {"key": list(c.key), "candidates": [x.value for x in c.candidates],
                 "resolved_value": c.resolved_value, "reason": c.reason}
                for c in self.conflicts
            ],
            "answer": self.answer,
            "model_calls": self.model_calls,
        }


PLAN_PROMPT = """You decompose financial research questions into atomic sub-questions.

Each sub-question must be answerable by retrieving ONE passage from an SEC filing.
That means: one company, one metric, one fiscal period. Do not ask for ratios,
comparisons, or changes over time -- those are computed later from the parts.

Return JSON only:
{{"sub_questions": ["...", "..."], "reasoning": "one sentence"}}

Rules:
- 2 to 6 sub-questions.
- Name the company and fiscal year explicitly in each.
- If the question needs a ratio, ask for numerator and denominator separately.

Question: {question}"""

EXTRACT_PROMPT = """Extract the answer to ONE sub-question from the passages below.

Sub-question: {sub_question}

Passages:
{passages}

Return JSON only:
{{"found": true|false, "value": "the specific figure or fact, with units",
  "quote": "verbatim text from the passage that states it, max 200 chars",
  "passage_number": <1-based index of the passage you used>,
  "entity": "the company, e.g. NVDA",
  "metric": "what is measured, e.g. research and development expense",
  "period": "the fiscal period THE PASSAGE STATES for this figure, e.g. FY2025"}}

Rules:
- Set found=false if the passages do not state the answer. Do NOT guess, infer,
  or compute. A wrong number is far worse than "not found".
- `quote` must appear verbatim in the passage you cite.
- Always fill entity/metric/period, even when found=false. They are used to
  cross-check this figure against other extractions.
- Filings print two or three years side by side. Find the column whose heading
  matches the period asked for, and take the value from that column.
- Do NOT assume the period asked for is present. If the passages cover only
  other fiscal years, set found=false -- do not return a nearby year's figure
  under the requested year's label. An earlier version of this instruction said
  "take the value for the period you were asked about", which presupposed the
  period existed and produced exactly that error.
- `period` must be the period the PASSAGE states for the figure you took, not
  the period in the sub-question. These are checked against each other."""

CRITIQUE_PROMPT = """Decide whether the evidence below is sufficient to answer the question.

Question: {question}

Evidence:
{evidence}

Return JSON only:
{{"sufficient": true|false,
  "missing": ["specific sub-questions still needed"],
  "reasoning": "one sentence"}}

Rules:
- If a needed figure is NOT FOUND, consider whether a differently-phrased
  sub-question might retrieve it; if so put that phrasing in `missing`.
- Do not request information that is merely nice to have.
- Max 3 items in `missing`."""

REFORMULATE_PROMPT = """These sub-questions retrieved nothing from a corpus of SEC filings.
Rewrite them so they match the vocabulary a filing actually uses.

Failed sub-questions:
{failed}

Return JSON only:
{{"rewritten": ["...", "..."]}}

Rules:
- Use the exact line-item wording found in financial statements
  (e.g. "Research and development", "Total revenue", "Net sales").
- Drop meta-phrasing like "as disclosed in its Form 10-K" -- it matches nothing.
- Keep the company and fiscal year explicit.
- Note that a fiscal year's figures also appear as the prior-year comparative in
  the FOLLOWING year's filing; phrase so either would match.
- One rewrite per failed question."""

SYNTH_PROMPT = """Answer the question using ONLY the evidence below.

Question: {question}

Evidence:
{evidence}

Rules:
- Every figure in your answer must come from the evidence. Never introduce a
  number that is not there.
- Show any arithmetic you perform, so it can be checked.
- Cite the source in brackets after each figure, e.g. [NVDA 10-K 2025].
- If the evidence is incomplete, say exactly what is missing and answer only the
  part you can support.
- CONTRADICTIONS: if two evidence items give different values for the same
  company, metric and period, do NOT silently choose one. Report the conflict,
  state both values with their citations, and treat that figure as unresolved.
  A confidently wrong number is the worst possible output; an explicit "these
  two passages disagree" is a useful one.
- Sanity-check magnitudes before using a figure. A company's annual R&D does not
  fall by two-thirds year over year, and a quarterly figure sitting in an annual
  column is the usual cause. If a value fails that check, flag it rather than
  computing with it.
- Be concise: lead with the answer, then the supporting figures."""

NUMBERS_CRITIC_PROMPT = """You verify FIGURES only. Check the draft answer
against the evidence: every number in the answer must appear in the evidence
verbatim or be exact arithmetic on evidence numbers. Flag any figure that is
absent, altered, mislabelled (wrong company/metric/year), or of implausible
magnitude for its label.

Question: {question}

Evidence:
{evidence}

Draft answer:
{answer}

Respond with JSON only: {{"ok": true/false, "issues": ["<one sentence per
problem figure>", ...]}}. Empty issues list if all figures check out."""

REASONING_CRITIC_PROMPT = """You verify CALCULATION AND LOGIC only, not
sourcing. Check the draft answer's arithmetic (recompute every operation),
its comparisons (does the stated winner match the stated numbers?), and
whether the conclusion actually answers the question asked (right company,
right metric, right period, right direction of change).

Question: {question}

Draft answer:
{answer}

Respond with JSON only: {{"ok": true/false, "issues": ["<one sentence per
error>", ...]}}. Empty issues list if the reasoning is sound."""

REVISE_PROMPT = """Your draft answer was reviewed and problems were found.
Rewrite it, fixing ONLY the listed problems. Keep every rule from before:
figures only from the evidence, cite in brackets, show arithmetic, report
conflicts rather than choosing.

Question: {question}

Evidence:
{evidence}

Draft answer:
{draft}

Problems found by review:
{feedback}

Write the corrected answer."""


class MultiHopAgent:
    def __init__(
        self,
        search: Callable[[str, int], Sequence[Any]],
        *,
        planner_models: Sequence[str] = PLANNER_MODELS,
        extractor_models: Sequence[str] = EXTRACTOR_MODELS,
        per_hop_k: int = 5,
        max_rounds: int = 2,
        check_contradictions: bool = True,
        verify_answer: str | None = None,
    ):
        """`search(query, k)` returns objects exposing .text, .chunk_id and
        .citation() -- the sec_rag Retriever satisfies this, but any retriever
        with that shape works."""
        self.search = search
        self.planner_models = list(planner_models)
        self.extractor_models = list(extractor_models)
        self.per_hop_k = per_hop_k
        self.max_rounds = max_rounds
        # Off only for the ablation in eval/run_agent_eval.py -- the point of the
        # check is to convert confident-wrong answers into honest abstentions,
        # and that claim is only worth anything if it can be measured against
        # the same agent with the check disabled.
        self.check_contradictions = check_contradictions
        # Optional post-synthesis verification. "dual" runs two SPECIALIZED
        # critics -- one verifying that every figure in the answer traces to
        # the extracted evidence, one verifying the calculation/reasoning --
        # and, if either objects, re-synthesizes once with their feedback.
        # Two critics rather than one general critic because the split is what
        # measured best on financial numeric QA (ICAIF'24,
        # doi:10.1145/3677052.3698686: FinQA 54.7% -> 64.1% one critic ->
        # 72.5% two specialized critics for an 8B model). Default off: it
        # costs 2-3 extra model calls per question, and published gains shrink
        # as the base model strengthens.
        self.verify_answer = verify_answer

    # -- steps ------------------------------------------------------------

    def plan(self, question: str) -> list[str]:
        data, _ = chat_json_chain(
            self.planner_models,
            [{"role": "user", "content": PLAN_PROMPT.format(question=question)}],
            # Validate shape only. An over-eager planner returning 9 sub-questions
            # is useful output to truncate, not a failure worth burning the next
            # model in the chain on -- an earlier version rejected it outright and
            # cascaded into a dead fallback.
            validate=lambda d: isinstance(d.get("sub_questions"), list)
            and len(d["sub_questions"]) >= 1,
            max_tokens=800,
        )
        return [str(s) for s in data["sub_questions"]][:6]

    def extract(self, sub_question: str) -> Evidence:
        hits = list(self.search(sub_question, self.per_hop_k))
        if not hits:
            return Evidence(sub_question, False, None, None, None, None)

        passages = "\n\n".join(
            f"[{i}] ({h.citation()})\n{h.text[:1500]}" for i, h in enumerate(hits, 1)
        )
        try:
            data, _ = chat_json_chain(
                self.extractor_models,
                [{"role": "user", "content": EXTRACT_PROMPT.format(
                    sub_question=sub_question, passages=passages)}],
                validate=lambda d: "found" in d,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    extract failed for {sub_question[:60]!r}: {exc}",
                  file=sys.stderr)
            return Evidence(sub_question, False, None, None, None, None,
                            error=str(exc))

        ent = (data.get("entity") or None)
        met = (data.get("metric") or None)
        per = (data.get("period") or None)

        # `found` has to be a real boolean. Models return the *string* "false"
        # often enough that a plain truthiness test silently converts a refusal
        # into a positive extraction -- and `str(False)` then renders the value
        # as the literal text "False", which flows into synthesis as if it were
        # a figure read off a filing.
        if data.get("found") is not True:
            return Evidence(sub_question, False, None, None, None, None,
                            entity=ent, metric=met, period=per)

        # Reject a figure that is about a different period than the one asked
        # for. The extractor is asked to report the period *the passage states*,
        # so a disagreement with the sub-question means it read the wrong column
        # -- or, worse, read the right column of a year that was never requested
        # and relabelled it. Observed: asked for fiscal 2018, which is outside
        # the corpus entirely, it returned AAPL's real FY2024 figures labelled
        # "fiscal 2018" and computed a ratio from them.
        #
        # This has to be a separate deterministic check rather than part of
        # contradiction detection, because every extraction in that run *agreed*.
        # A cross-check finds inconsistency; a uniformly mislabelled answer is
        # consistent and wrong.
        if period_mismatch(sub_question, per):
            return Evidence(sub_question, False, None, None, None, None,
                            entity=ent, metric=met, period=per)

        # An extraction with no value is not an extraction. Letting it through
        # as found=True puts "None" in front of the synthesizer with a real
        # citation attached to it.
        value = str(data.get("value") or "").strip() or None
        if value is None:
            return Evidence(sub_question, False, None, None, None, None,
                            entity=ent, metric=met, period=per)

        # Same reasoning as contradictions.resolve: if the model did not cite a
        # passage we actually showed it, we do not know where the figure came
        # from, so we must not manufacture a citation for it.
        n = data.get("passage_number")
        if not (isinstance(n, int) and 1 <= n <= len(hits)):
            return Evidence(sub_question, False, None, None, None, None,
                            entity=ent, metric=met, period=per)
        hit = hits[n - 1]
        return Evidence(
            sub_question=sub_question,
            found=True,
            value=value,
            quote=(data.get("quote") or None),
            chunk_id=hit.chunk_id,
            citation=hit.citation(),
            entity=ent,
            metric=met,
            period=per,
        )

    def critique(self, question: str, evidence: Sequence[Evidence]) -> tuple[bool, list[str]]:
        rendered = "\n".join(e.render() for e in evidence) or "(none)"
        try:
            data, _ = chat_json_chain(
                self.planner_models,
                [{"role": "user", "content": CRITIQUE_PROMPT.format(
                    question=question, evidence=rendered)}],
                validate=lambda d: "sufficient" in d,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    critique failed, failing open: {exc}", file=sys.stderr)
            return True, []  # fail open: synthesize with what we have
        return bool(data.get("sufficient")), [str(m) for m in (data.get("missing") or [])][:3]

    def reformulate(self, failed: Sequence[str]) -> list[str]:
        """Rewrite sub-questions that retrieved nothing.

        Retrieval failures are frequently vocabulary mismatches rather than
        missing data: the filing says "Research and development" in a statement
        of operations table, while the sub-question asks for it "as disclosed in
        its Form 10-K". Rephrasing toward the language a filing actually uses is
        usually enough.
        """
        try:
            data, _ = chat_json_chain(
                self.planner_models,
                [{"role": "user", "content": REFORMULATE_PROMPT.format(
                    failed="\n".join(f"- {q}" for q in failed))}],
                validate=lambda d: isinstance(d.get("rewritten"), list),
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    reformulate failed: {exc}", file=sys.stderr)
            return []
        return [str(q) for q in data["rewritten"]][: len(failed) + 1]

    def _render_evidence(self, evidence: Sequence[Evidence],
                         conflicts: Sequence[Conflict] = ()) -> str:
        # Conflicted figures are replaced by their adjudicated result (or an
        # explicit UNRESOLVED marker), so the raw disagreeing values never
        # reach the synthesizer as equal-looking options.
        conflicted_ids = {id(c) for conf in conflicts for c in conf.candidates}
        lines = [e.render() for e in evidence if id(e) not in conflicted_ids]
        lines += [c.render() for c in conflicts]
        return "\n".join(lines) or "(no evidence found)"

    def dual_critique(self, question: str, evidence: Sequence[Evidence],
                      answer: str) -> list[str]:
        """Two specialized critics: figures-vs-evidence, and reasoning.

        Returns the combined issue list (empty = both critics passed). A
        critic that fails outright contributes nothing rather than blocking --
        the verification layer must never be the reason an answer is lost.
        """
        rendered = self._render_evidence(evidence)
        issues: list[str] = []
        for prompt in (
            NUMBERS_CRITIC_PROMPT.format(question=question, evidence=rendered,
                                         answer=answer),
            REASONING_CRITIC_PROMPT.format(question=question, answer=answer),
        ):
            try:
                data, _ = chat_json_chain(
                    self.planner_models,
                    [{"role": "user", "content": prompt}],
                    validate=lambda d: "ok" in d,
                    max_tokens=400,
                )
                if not data.get("ok"):
                    issues += [str(i) for i in (data.get("issues") or [])][:3]
            except Exception as exc:  # noqa: BLE001
                print(f"    critic failed (skipping): {exc}", file=sys.stderr)
        return issues

    def synthesize(self, question: str, evidence: Sequence[Evidence],
                   conflicts: Sequence[Conflict] = (), *,
                   feedback: str | None = None, draft: str | None = None) -> str:
        rendered = self._render_evidence(evidence, conflicts)
        if feedback is not None and draft is not None:
            content = REVISE_PROMPT.format(question=question, evidence=rendered,
                                           draft=draft, feedback=feedback)
        else:
            content = SYNTH_PROMPT.format(question=question, evidence=rendered)
        # Same fallback discipline as every other model call: this was the one
        # hard-coded single-model call left, which meant a model EOL (HTTP 410)
        # would fail the run at the final step after every hop had succeeded.
        errors: list[str] = []
        for model in self.planner_models:
            try:
                answer = chat(
                    model,
                    [{"role": "user", "content": content}],
                    max_tokens=1200,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model}: {exc}")
                continue
            # An empty completion is a failure, not an answer. Observed in the
            # wild: deepseek returning '' during an API bad patch -- the chain
            # "succeeded", the agent said nothing, and the eval graded the
            # silence as wrong (answerable) or fabricated (controls). 14 of 30
            # questions in one run were lost to exactly this.
            if answer and answer.strip():
                return answer
            errors.append(f"{model}: empty completion")
        raise RuntimeError(
            "synthesize: all models failed:\n  " + "\n  ".join(errors))

    # -- driver -----------------------------------------------------------

    def run(self, question: str, *, verbose: bool = False) -> Trace:
        trace = Trace(question=question)
        pending = self.plan(question)
        trace.model_calls += 1
        if verbose:
            print(f"plan: {len(pending)} sub-questions")
            for s in pending:
                print(f"   - {s}")

        seen: set[str] = set()
        for rnd in range(self.max_rounds):
            fresh = [s for s in pending if s not in seen]

            # The critique often returns a still-missing item using the *same*
            # wording that already failed. Retrying it verbatim is guaranteed to
            # fail again, and skipping it silently turns self-correction into a
            # no-op -- so reformulate before giving up.
            if not fresh and pending:
                fresh = [q for q in self.reformulate(pending) if q not in seen]
                trace.model_calls += 1
                if verbose and fresh:
                    print(f"   reformulated {len(pending)} -> {len(fresh)} new phrasings")

            if not fresh:
                break
            seen.update(fresh)

            # Sub-questions in a hop are independent by construction -- the
            # planner is asked for atomic, independently-retrievable questions --
            # so extracting them sequentially just serialises latency that does
            # not need serialising. This matters more than it looks: observed
            # per-call latency on this API varies from 4s to 50s for identical
            # call shapes, so a four-hop question inherits the sum of four worst
            # cases instead of the max. Order is preserved so traces stay
            # comparable between runs.
            if len(fresh) > 1:
                with ThreadPoolExecutor(max_workers=min(len(fresh), MAX_HOP_WORKERS)) as pool:
                    hop_evidence = list(pool.map(self.extract, fresh))
            else:
                hop_evidence = [self.extract(s) for s in fresh]
            trace.model_calls += len(fresh)
            trace.evidence.extend(hop_evidence)
            if verbose:
                for e in hop_evidence:
                    print(f"   {'OK ' if e.found else 'MISS'} {e.sub_question[:60]} -> {e.value}")

            sufficient, missing = self.critique(question, trace.evidence)
            trace.model_calls += 1
            trace.rounds.append({
                "round": rnd,
                "sub_questions": fresh,
                "found": sum(1 for e in hop_evidence if e.found),
                "sufficient": sufficient,
                "missing": missing,
            })
            if verbose:
                print(f"   critique: sufficient={sufficient} missing={len(missing)}")
            if sufficient or not missing:
                break
            pending = missing

        # Cross-check before synthesis. Two extractions that disagree about the
        # same figure are resolved against the source passages here, so the
        # synthesizer is never put in the position of choosing between them --
        # a job it has no grounds to do well, and which it previously did wrong.
        conflicts = find_conflicts(trace.evidence) if self.check_contradictions else []
        for c in conflicts:
            resolve(c, self.search, self.planner_models)
            trace.model_calls += 1
        trace.conflicts = conflicts
        if verbose and conflicts:
            for c in conflicts:
                status = "resolved" if not c.unresolved else "UNRESOLVED"
                print(f"   conflict [{status}] {' '.join(c.key)}: "
                      f"{[x.value for x in c.candidates]} -> {c.resolved_value}")

        trace.answer = self.synthesize(question, trace.evidence, conflicts)
        trace.model_calls += 1

        if self.verify_answer == "dual" and trace.answer.strip():
            issues = self.dual_critique(question, trace.evidence, trace.answer)
            trace.model_calls += 2
            trace.verification = {"issues": issues, "revised": False}
            if issues:
                feedback = "\n".join(f"- {i}" for i in issues)
                revised = self.synthesize(
                    question, trace.evidence, conflicts,
                    feedback=feedback, draft=trace.answer)
                trace.model_calls += 1
                if revised.strip():
                    trace.answer = revised
                    trace.verification["revised"] = True
                if verbose:
                    print(f"   critics raised {len(issues)} issue(s); revised")
        return trace
