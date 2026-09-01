# agentic-rag

A multi-hop RAG agent that decomposes a question, retrieves per sub-question,
verifies its own evidence, and refuses to answer what it cannot support.

Runs on the SEC corpus built by the sibling [`sec-rag`](../sec-rag) project,
using NVIDIA NIM models for planning, extraction and synthesis.

## The problem

Single-shot RAG answers *"What was NVDA's FY2025 revenue?"* fine. It fails on:

> *Did NVDA's R&D spending as a share of revenue rise faster than Microsoft's
> between fiscal 2024 and fiscal 2025?*

No passage contains that answer. It needs four retrievals, two divisions and a
comparison — and the failure mode when a model attempts it in one shot is a
confident, fabricated number.

## The loop

```
plan         decompose into atomic, independently-retrievable sub-questions
retrieve     one hybrid retrieval per sub-question
extract      pull one value per sub-question, with a citation and a "not found" option
critique     does the evidence answer the question? if not, what is missing?
reformulate  rewrite failed sub-questions toward filing vocabulary
synthesize   compose from verified evidence only
```

Two design decisions carry most of the reliability:

**Extraction is separated from synthesis.** The extractor sees one sub-question
and its passages, and returns a value plus the chunk it came from. The
synthesizer sees only extracted values — never raw passages. So the final answer
cannot introduce a number that was not extracted, and every figure traces to a
chunk id. It costs one extra model call per hop.

**"Not found" is a first-class answer.** Each extraction may return
`found: false`; the critique step then chooses between reformulating and
reporting the gap. An agent that cannot say *"the filings don't state this"* will
fabricate instead.

## What actually happened when it ran

Real output, first version:

```
plan: 4 sub-questions
   OK   NVDA R&D + revenue FY2024   -> $8,675M / $60,925M
   MISS NVDA R&D + revenue FY2025   -> None
   OK   MSFT R&D + revenue FY2024   -> $29,510M / $245,122M
   OK   MSFT R&D + revenue FY2025   -> $32,488M / $281,724M
   critique: sufficient=False missing=1
```

It **correctly refused** to invent NVDA's missing FY2025 figures, computed
Microsoft's ratios accurately (12.04% → 11.53%), and stated exactly what was
missing. That is the behaviour the design exists for.

Two bugs surfaced, and both are recorded here rather than quietly fixed:

**1. Self-correction silently no-opped.** The critique flagged the gap, then the
loop stopped. The round logic skipped any sub-question already attempted, and the
critique had returned the *same phrasing* that had just failed — so the fresh
list was empty and the loop broke. Fixed by adding a `reformulate` step:
retrieval failures here are usually vocabulary mismatches ("as disclosed in its
Form 10-K" matches nothing; "Research and development" matches the statement of
operations).

**2. It produced a confidently wrong number.** After that fix, the agent
extracted NVDA FY2025 R&D as **$4,239M** — wrong; the correct figure is
~$12,914M, which it had *already retrieved* in an earlier hop. It noticed the
contradiction, said so in the answer, and then resolved it by picking the wrong
value.

That is the real failure mode of agentic RAG: not hallucination from nothing, but
mis-selection among genuinely retrieved candidates, compounded by a synthesizer
that treats "pick one" as its job.

## The fix: contradiction detection before synthesis

`agent/contradictions.py` takes the choice away from the synthesizer.
Extractions are grouped by what they are *about* — `(entity, metric, period)` —
and any group whose values disagree is re-verified against the source passages
before synthesis sees it. Unresolvable conflicts are passed through as an
explicit `UNRESOLVED CONFLICT` marker rather than as two equal-looking options.

Three things had to be right, and each was a bug caught by a unit test first:

**Values must compare by magnitude, not string.** `"$12.9 billion"` and
`"12,914 million"` are the same figure; `"$12,914 million"` and `"$4,239 million"`
are not. A 1% relative tolerance absorbs the rounding that filings do between
statements (60,922 vs 60,925 for the same line item) without hiding the real bug.

**Entity comes from the citation, not the model.** The extractor writes whatever
the passage said — `NVDA`, `Nvidia`, `NVIDIA Corporation` — none of which group.
The citation is machine-built by the retriever (`"NVDA 10-K 2025, Item 7"`), so
its leading token is a reliable ticker.

**Period normalization by word-stripping was silently broken.** Removing
`\b(fiscal|year|fy)\b` leaves `"FY2025"` untouched — there is no word boundary
between `fy` and `2025` — so it never grouped with `"2025"`. Extracting the
4-digit year outright removes the whole class of problem.

```
$ uv run pytest tests/ -q
57 passed
```

The headline test asserts detection fires on the exact observed failure, and
*doesn't* fire on filing-level rounding.

Remaining honesty: the resolver is an LLM adjudicating against re-retrieved
passages, so it can still be wrong — but it now fails to an explicit "cannot
determine" rather than to a confident wrong number.

## The failure contradiction detection cannot catch

The end-to-end eval asked for AAPL's R&D-to-revenue ratio in fiscal **2018** — a
year entirely outside the indexed corpus. The agent answered:

```
The percentage ... in fiscal 2018 is approximately 8.02%.
- R&D expense for fiscal 2018: $31,370   [AAPL 10-K 2024, Item 7, chunk ...#0078]
- Net sales for fiscal 2018:  $391,035   [AAPL 10-K 2024, Item 8, chunk ...#0101]
```

Real chunks, real citations, correct arithmetic — and AAPL's **FY2024** figures
relabelled fiscal 2018.

**Cross-checking is structurally blind to this.** Every extraction agreed with
every other, because they were uniformly wrong. Contradiction detection finds
*inconsistency*; this is perfectly consistent and entirely false.

The extractor prompt caused it. It said *"take the value from the column for the
period you were asked about"* — which presupposes that column exists. Asked for a
year the filing does not contain, the model obeyed and took the nearest one.

The fix is deterministic and separate: the extractor now reports the period **the
passage states**, and `period_mismatch` rejects any extraction whose year set
shares nothing with the years asked for. Year *sets*, not first years: the
planner emits compound sub-questions ("fiscal years 2024 and 2025") and NVDA's
fiscal phrasing spans two calendar years ("February 2024 to January 2025") —
first-year comparison rejected correct extractions in both cases (see REPORT
§5.2–5.4 for the audit that caught this).

```
unanswerable controls, before:  1/1 fabricated
unanswerable controls, after:   2/2 abstained,  1.0 numbers per answer
answerable questions:           verdicts unchanged on this run
```

## End-to-end eval: the measurement was wrong before the agent was

Answers are graded against quantities **computed in Python from XBRL**, never by
a judge model — the same class of model that picked $4,239M over $12,914M is not
a reliable judge of that pick.

The first run scored 3 correct / 2 wrong. Both "wrong" answers turned out to be
right:

```
($31,370M R&D ÷ $391,035M revenue) × 100 = 8.02%     <- the agent
gold: 6.66%                                          <- wrong
```

Gold had been keyed on the SEC API's `fy` field, which is the fiscal year of the
**filing**, not of the fact. A 10-K prints three comparative years all tagged
`fy=2024`, so the eval had quietly used FY2022's R&D as FY2024's. Deriving the
year from the period `end` date instead:

| | before fix | after fix |
|---|---|---|
| correct | 3 | **4** |
| wrong | 2 | **0** |
| abstain | 0 | 1 |

**At n=30 (after the audit's grader rewrite): answerable accuracy 20/24 =
0.83, Wilson95 [0.64, 0.93]; unanswerable controls 6/6 abstained.** Re-grading
the committed traces offline puts the null control at **12/24 = 0.50** on the
same denominator the accuracy uses — all twelve null hits fall on answerable
questions, since an abstaining control emits no numbers to collide. (Quoting it
as 12/30 = 0.40 mixes denominators and flatters the margin.) That is well above
the ~0.2–0.3 previously written here: the margin over the null is 0.83 vs 0.50,
real but modest, and the null is this high because gold values collide across
questions — a property of the question set, worth fixing before the accuracy
number is leaned on. Two things had to be fixed to
get an honest number, and both are the interesting part:

- *The grader could be satisfied by a citation.* `_NUM` read "10" out of
  "10-K" and "8" out of "Item 8", so `"See the 10-K filing."` graded correct
  against a percent gold — and the null control read 0.60 for the same
  reason. Rewritten (citations stripped, percent golds require percent-stated
  numbers, "decreased by 3.36%" understood as −3.36, conclusion graded before
  the abstain check), pinned by regression tests.
- *An empty completion counted as an answer.* During an API bad patch,
  deepseek returned empty strings; the fallback chain only advanced on
  exceptions, and the grader scored the silence as wrong/fabricated — 14 of
  30 questions in one run. The chain now advances on empty output and the
  grader has an `empty_answer` outcome (see REPORT §5.5).

The "matching figure anywhere in the answer" looseness was then tested
directly: a strict mode (`--grade strict`) grades only the conclusion
sentence, and re-grading all 30 answers **changed zero verdicts** — the
caveat retired with data (REPORT §5.7).

**Optional: dual specialized critics** (`--verify dual`, off by default) — a
figures-vs-evidence critic plus a calculation critic with one revision round,
after the ICAIF '24 result that two specialized critics beat one general
critic on FinQA-class tasks. A/B on 10 questions: 9/10 → 10/10 at ~40% extra
latency, with the flip attributable to either the critics or run-to-run
nondeterminism — see REPORT §5.6 for the honest read.

**Optional: relevance-guided corpus grep** (`--search-mode relgrep`, off by
default) — after "A New Role for Relevance: Guiding Corpus Interaction in
Agentic Search" (arXiv 2607.24223): when a hop query names a company and a
fiscal year, take the exact-match subset of the corpus (a conjunctive filter
no bag-of-words retriever provides) and order it coarse-to-fine with the
hosted reranker; anchor-free queries fall back to the dense default
untouched. Implemented as a shared `make_search` factory
(`src/agentic_rag/search_modes.py`) that also de-duplicates the three
formerly copy-pasted search closures. Grep and anchor logic are covered by
network-free unit tests (`tests/test_search_modes.py`).

**The contradiction guard fires when the fault is injected — on a sample too
small to call it validated.** Across every natural run, `conflicts detected = 0`
(mostly-successful 2-hop questions never extract the same figure twice), so the
guard was exercised by injecting the failure it exists to catch: a corrupted
twin extraction per question (value ×0.33, variant entity/period spellings so
the grouping normalizers are genuinely exercised, real chunk ids). Result —
detection **5/5**; the corrupted figure reached the final answer **3/5 with the
guard off vs 1/5 with it on** (and that one was an abstention quoting the
conflict, not an assertion) — `eval/run_conflict_injection.py`, REPORT §5.3.

Read that with the denominators visible, because they are small and they are
not equal:

- **n = 5 questions, and 4 of them ask about R&D FY2024** in different shapes
  (ratio / delta / compare / trend). Together the five questions draw on seven
  atomic source figures, but four of them are shapes over the same R&D pair —
  correlated questions, not five independent trials.
- **`grade` returns "ungradeable" too**, and the arms produce different amounts
  of it — 2 ungradeable with the guard on, 3 with it off. So the "correct
  answers 0 → 2" figure previously quoted here compares 3 gradeable answers
  against 2. It is a direction, not an effect size, and it is no longer quoted
  as one.
- **Detection 5/5 vs 0/5 is definitional on the OFF arm**: the detector is
  disabled there, so its zero is a tautology rather than a measurement. The
  informative half is the ON arm's 5/5 and the contamination gap.

The arms also run the agent twice rather than replaying one trajectory, so
extraction is re-sampled per arm and the injected twin is not guaranteed to
attach at the same seam. What this run recorded is that the guard
*fired on every injection and the corrupted figure reached fewer answers with
it on* — in two arms that were not given identical corruptions, so even that is
an observation, not an identified effect; a controlled effect size needs
recorded trajectories replayed with the guard toggled.

The audit had also found and fixed two reasons conflicts could silently fail to
form: first-year-only period keys and non-numeric values that always
"conflicted".

```bash
uv run python eval/build_multihop_eval.py
uv run python eval/run_agent_eval.py --limit 5 --types ratio,delta,compare,trend
```

## Model availability is not stable

Mid-development, two models went from working to dead:

```
qwen/qwen3-next-80b-a3b-instruct  ->  HTTP 410 "end of life on 2026-07-27"
qwen/qwen3.5-397b-a17b            ->  HTTP 410 "end of life on 2026-07-27"
moonshotai/kimi-k2.6              ->  404 for this account
google/gemma-4-31b-it             ->  timeout at 45s
deepseek-ai/deepseek-v4-flash     ->  HTTP 410 "end of life on 2026-08-07"
                                      (chains updated to -0731 same week)
```

No deprecation warning; the model simply began returning 410. This is why every
model call goes through `chat_json_chain` — a validating fallback chain — rather
than a hard-coded id. Probe before trusting anything:

```bash
uv run python scripts/probe_models.py
```

Current chain, ordered by observed JSON reliability:

| role | models |
|---|---|
| planner / critique | `deepseek-ai/deepseek-v4-flash`, `nvidia/nemotron-3-super-120b-a12b`, `openai/gpt-oss-120b` |
| extractor | `deepseek-ai/deepseek-v4-flash`, `nvidia/nemotron-3-nano-30b-a3b`, `nemotron-3-super` |

Hosted NIM endpoints do not document `guided_json` / `nvext` constrained decoding
— that exists only for self-hosted NIM containers — so JSON is parsed
defensively and validated per call, with the chain moving on when a model
produces an unusable shape.

A related trap, inherited from prior work on this stack: several hosted models
emit chain-of-thought that pollutes JSON output, and the suppression incantation
differs per family (`/no_think` for Qwen; `chat_template_kwargs.thinking=False`
**plus** a "detailed thinking off" directive for Nemotron, because the directive
alone silently stops working on long prompts). Some models put their actual
answer in `reasoning_content` rather than `content`, so the client accumulates
both.

## Setup

```bash
# 1. Build the corpus in the sibling project first
cd ../sec-rag && uv run python scripts/build_index.py

# 2. Then here
cd ../agentic-rag
uv sync
cp .env.example .env    # NVIDIA_API_KEY=nvapi-...
```

`sec-rag` is a path dependency, so retrieval is reused rather than
reimplemented — the agent only needs an object exposing `.text`, `.chunk_id` and
`.citation()`.

## Usage

```bash
uv run python scripts/ask.py "Did NVDA's R&D spending as a share of revenue \
rise faster than Microsoft's between fiscal 2024 and fiscal 2025?" \
  --trace data/trace.json
```

`--trace` writes the full decision record: every sub-question, what was found or
missed, each critique verdict, and the model-call count.

Retrieval is configured as **dense + rerank, no BM25** — the measured best
configuration on this corpus per `sec-rag`'s ablation, where hybrid fusion
actively hurt.

## Layout

```
src/agentic_rag/
  nvidia.py               NIM client with retry + validating model-fallback chains
  agent/loop.py           plan / extract / critique / reformulate / synthesize
  agent/contradictions.py magnitude parsing, grouping, conflict detection,
                          resolution, and period-mismatch rejection
eval/
  build_multihop_eval.py  XBRL-computed answers for 2- and 4-hop questions
  run_agent_eval.py       three-way grading + null-model control + ablation switch
tests/
  test_contradictions.py  33 tests, incl. both observed failures as regressions
scripts/
  ask.py                  CLI entry point
  probe_models.py         which hosted models are alive and return usable JSON
```

## Limitations

- The conflict resolver is itself an LLM call. It fails to "cannot determine"
  rather than to a wrong number, which is the right failure direction, but it is
  not a guarantee. It is now **checked rather than trusted**: a resolved value
  is only rendered to the synthesizer as "verified" if the figure actually
  occurs in the passage the verifier cited, at any of the scales filings use
  (`value_in_passage`). Previously the resolver could return a figure appearing
  in no passage at all and the synthesizer would receive it labelled verified —
  carrying more authority than the two conflicting extractions it replaced.
  A verifier that cites a passage index outside the set it was shown now leaves
  the conflict unresolved instead of silently borrowing the first passage's
  citation.
- Contradiction detection only covers **numeric** values — non-numeric claims
  ("management cited supply constraints") are grouped but not compared.
- Bounded at `max_rounds=2`. A self-correcting loop without a budget is a way to
  spend money slowly.
- **The contradiction check has no natural end-to-end validation.** Zero
  conflicts fired across every eval run, so the ON/OFF ablation on natural data
  is uninformative. Fault injection shows the guard fires and reduces
  contamination when the fault is present, but over 5 correlated questions —
  they draw on seven atomic source figures, four of them shapes over the same
  R&D pair — with unequal gradeable denominators per arm (3 vs 2). Direction,
  not effect size. The saved run also predates this branch's resolver changes,
  so it describes the guard as it was, not as it is.
- **Answer accuracy is an upper bound**, not a measurement — at the n=30
  headline, the recomputed null-model hit rate is 0.50 over the same 24
  answerable questions the 0.83 accuracy is computed on (the earlier
  5-question framing, 0.60 vs 0.80, is superseded). Grading needs to target the stated
  conclusion rather than any number in the answer.
- Extraction now runs in parallel across sub-questions (bounded at 4 workers),
  which matters because observed per-call latency on this API ranges from 4s to
  50s for identical call shapes.

## License

MIT.
