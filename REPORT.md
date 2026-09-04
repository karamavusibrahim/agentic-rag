# agentic-rag — technical report

**Question.** What actually breaks when a RAG system has to answer a question no
single passage contains?

**Answer.** Not hallucination from nothing — that is the failure the literature
prepares you for, and the one this agent's design already prevents. The real
failure is **mis-selection among genuinely retrieved candidates**: the agent had
the correct figure in hand from an earlier hop, noticed it conflicted with a
later extraction, said so explicitly in its answer, and then resolved the
conflict by choosing the wrong one.

A second, nastier shape emerged from the eval: **uniform mislabelling**, where
every extraction agrees on a false premise. Asked about a fiscal year absent from
the corpus, the agent returned genuine FY2024 figures under a "fiscal 2018"
label, with real citations and correct arithmetic. No cross-check can see that,
because there is no disagreement (§5.2).

And the eval's own first verdict on the agent was wrong (§5.1).

| | |
|---|---|
| Retrieval | `sec-rag` as a path dependency — dense + rerank, no BM25 |
| Models | validating fallback chains, `deepseek-v4-flash` first |
| Loop | plan → retrieve → extract → critique → reformulate → synthesize |
| Guards | contradiction detection + period-mismatch rejection (57 unit tests) |
| Eval | XBRL-computed multi-hop answers, three-way outcome, null-model control |

---

## 1. Design

### 1.1 Two decisions carry most of the reliability

**Extraction is separated from synthesis.** The extractor sees one sub-question
and its passages, and returns a value plus the chunk it came from. The
synthesizer sees only extracted values — never raw passages. The final answer
therefore *cannot* introduce a number that was not extracted, and every figure
traces to a chunk id. Cost: one extra model call per hop.

**"Not found" is a first-class answer.** Each extraction may return
`found: false`; the critique step then chooses between reformulating and
reporting the gap. An agent with no way to say *"the filings don't state this"*
will fabricate instead.

### 1.2 What happened on the first real run

```
plan: 4 sub-questions
   OK   NVDA R&D + revenue FY2024   -> $8,675M / $60,925M
   MISS NVDA R&D + revenue FY2025   -> None
   OK   MSFT R&D + revenue FY2024   -> $29,510M / $245,122M
   OK   MSFT R&D + revenue FY2025   -> $32,488M / $281,724M
   critique: sufficient=False missing=1
```

It correctly refused to invent NVDA's missing figures, computed Microsoft's
ratios accurately (12.04% → 11.53%), and stated exactly what was missing. That is
the design working.

Two bugs followed, and both are more interesting than the success.

---

## 2. Bug 1 — self-correction that silently did nothing

The critique flagged the gap, and then the loop stopped.

The round logic skipped any sub-question already attempted. The critique had
returned the *same phrasing* that had just failed, so the "fresh" list was empty
and the loop broke. Self-correction was structurally present and functionally
absent — it emitted the right diagnosis into a code path that discarded it.

The fix was a `reformulate` step. Retrieval failures here are almost always
vocabulary mismatches: *"as disclosed in its Form 10-K"* matches nothing, while
*"Research and development"* matches the statement of operations directly.

This is a failure mode worth naming: a self-correcting loop whose correction is
unreachable looks identical, in logs, to one that had nothing to correct.

---

## 3. Bug 2 — the confidently wrong number

After that fix, the agent extracted NVDA FY2025 R&D as **$4,239M**. The correct
figure is ~$12,914M — which it had *already retrieved* in an earlier hop. It
noticed the contradiction, mentioned it in the answer, and resolved it wrongly.

The likely proximate cause is a quarterly figure read as annual, or a prior-year
comparative column read as the current year. But the structural cause is the
prompt: a synthesizer asked to "compose an answer" treats choosing between
candidates as part of its job, and it has no grounds to choose well. It sees two
numbers and no evidence about which is right, because the passages were
deliberately withheld from it (§1.1).

### 3.1 The fix: take the choice away

`agent/contradictions.py` groups extractions by what they are *about* —
`(entity, metric, period)` — and any group whose values disagree is re-verified
against re-retrieved passages before synthesis sees it. Unresolvable conflicts
pass through as an explicit `UNRESOLVED CONFLICT` marker rather than as two
equal-looking options.

Grouping needs structured fields rather than string similarity, because two
sub-questions can be worded very differently and mean the same thing (*"NVDA R&D
FY2025"* vs *"Nvidia research and development expense for fiscal year 2025"*),
while two nearly identical strings can differ in the one token that matters
(FY2024 vs FY2025). So the extractor emits entity/metric/period explicitly.

### 3.2 Three bugs the unit tests caught first

**Values must compare by magnitude, not string.** `"$12.9 billion"` and
`"12,914 million"` are the same figure. A 1% relative tolerance absorbs the
rounding filings do between statements (60,922 vs 60,925 for the same line item)
without hiding the real bug.

**Entity must come from the citation, not the model.** The extractor writes
whatever the passage said — `NVDA`, `Nvidia`, `NVIDIA Corporation` — none of
which group. The citation is machine-built by the retriever
(`"NVDA 10-K 2025, Item 7"`), so its leading token is a reliable ticker.

**Period normalisation by word-stripping was silently broken.** Removing
`\b(fiscal|year|fy)\b` leaves `"FY2025"` untouched — there is no word boundary
between `fy` and `2025` — so it never grouped with `"2025"`. The first
implementation detected **zero** conflicts on the exact bug it was written for.
Extracting the 4-digit year outright removes the whole class of problem.

```
$ uv run pytest tests/ -q
57 passed
```

The headline test asserts detection fires on the observed $12,914M vs $4,239M
pair and does *not* fire on filing-level rounding.

---

## 4. Measuring whether the fix earns its cost

The check adds a model call per conflict, so it has to be worth something. The
claim being tested is deliberately narrow: it should convert *confidently wrong*
answers into *honest abstentions*. It is **not** expected to raise accuracy much —
an agent that mis-extracts a figure in every hop has nothing to cross-check
against.

### 4.1 Ground truth without a judge

Asking a model whether an answer looks right imports the failure being measured:
the same class of model that picked $4,239M over $12,914M is not a reliable judge
of whether that pick was correct.

XBRL removes the need for a judge. A *derived* quantity — a ratio, a
year-over-year delta, a cross-company comparison — can be computed exactly in
Python from facts the agent never sees. 83 questions in four shapes:

```
ratio      one company, two facts, one division      (2 hops)
delta      one company, one metric, two years        (2 hops)
compare    two companies, one metric, one year       (2 hops, categorical)
trend      two companies, two metrics, two years     (4 hops)
```

Plus an `unanswerable` control set: identical phrasing, fiscal years outside the
indexed filings. Anything other than abstention there is fabrication.

Ground truth is restricted to 10-K facts with an **annual duration**, because the
same concepts also carry quarterly values — and a quarterly figure substituted
for an annual one is precisely the error being hunted. Using unfiltered facts as
truth would have made the eval *agree with the bug*.

The fiscal year is derived from each fact's period `end` date, **not** from the
API's `fy` field. That distinction is not cosmetic; getting it wrong is what
produced §5.1.

### 4.2 Outcomes are three-way, not binary

```
correct   the computed gold value appears in the answer
wrong     an answer was given, and it is not the gold value
abstain   the agent stated it could not determine the figure
```

Scoring on accuracy alone would understate the fix; scoring on abstention alone
would let a broken agent that abstains on everything look perfect. The number
that matters is the **wrong → abstain** shift.

### 4.3 Two controls

**Null-model hit rate.** "The gold number appears in the answer" is a weak test:
an answer listing many figures will match by luck. So for every question, every
*other* question's gold value is also checked against the answer. A real hit rate
only means something to the extent it exceeds the null rate.

This control immediately earned its place — see §6.3, where it says the primary
metric is not yet trustworthy.

**Categorical grading returns "ungradeable" rather than guessing.** Both tickers
appear in a comparison question and usually in its answer, so presence proves
nothing. The first sentence carrying a comparative cue states the verdict, and
whichever ticker appears earlier in it is the claim. Where no such sentence
exists the question is excluded — a guess there would silently become an accuracy
number.

---

## 5. What the eval actually found

### 5.1 The eval was wrong before the agent was

First run, 6 questions: 3 correct, 2 wrong, 1 fabricated. Inspecting the "wrong"
answers changed the conclusion entirely:

```
Approximately 8.02% of AAPL's total revenue in fiscal 2024 was spent on R&D.
  ($31,370M R&D  ÷  $391,035M total revenue) × 100 = 8.02%
```

Those are AAPL's real FY2024 figures, correctly cited and correctly divided. The
**gold value was wrong** — 6.66%, built from XBRL by keying on the `fy` field,
which is the fiscal year of the *filing*, not of the fact. A 10-K carries three
comparative years all tagged with the same `fy`, so the eval had silently used
FY2022's R&D as FY2024's.

Corrected (deriving the year from the period `end` date, and requiring an annual
duration), the same 5 answerable questions score:

| | before fix | after fix |
|---|---|---|
| correct | 3 | **4** |
| wrong | 2 | **0** |
| abstain | 0 | 1 |

The agent had been right and the measurement had been wrong. This is the third
metric bug across these three projects, and the most consequential: it would have
reported a 40% error rate that did not exist.

### 5.2 A fabrication the contradiction check cannot catch

The `unanswerable` control asked for AAPL's fiscal **2018** ratio — a year
entirely outside the indexed corpus. The agent returned:

```
The percentage ... in fiscal 2018 is approximately 8.02%.
- R&D expense for fiscal 2018: $31,370   [AAPL 10-K 2024, Item 7, chunk ...#0078]
- Net sales for fiscal 2018:  $391,035   [AAPL 10-K 2024, Item 8, chunk ...#0101]
```

Real chunks, real citations, correct arithmetic — and AAPL's **FY2024** figures
relabelled as fiscal 2018.

**Contradiction detection is structurally blind to this.** Every extraction
agreed with every other, because they were uniformly mislabelled. Cross-checking
finds *inconsistency*; this failure is perfectly consistent and entirely wrong.
A guard against disagreement cannot catch agreement on a false premise.

Worse, the extractor prompt caused it. It said:

> *Take the value from the column for the period you were asked about.*

which presupposes the period exists. Asked for a year not in the filing, the
model did as instructed and took the nearest column.

The fix is deterministic and separate from the contradiction machinery: the
extractor now reports the period **the passage states**, and `period_mismatch`
rejects any extraction whose year disagrees with the year requested. It is
conservative — it fires only when both sides name at least one year and the
year *sets* share none. The set comparison matters: the first version compared
first-years only, which rejected correct extractions whenever the planner
emitted a compound sub-question ("fiscal years 2024 and 2025" answered for
FY2025) or NVDA's fiscal phrasing spanned two calendar years ("February 2024 to
January 2025" for FY2025). Both cases occur in real traces.

```
unanswerable controls, before:  1/1 fabricated
unanswerable controls, after:   2/2 abstained,  1.0 numbers per answer
answerable questions:           verdicts unchanged on this run
```

(An earlier version of this table attributed "20.2 numbers per answer" to the
fabricated control; that figure was the mean over the whole 6-question mixed
run, not the control answer. "Verdicts unchanged" is likewise a single-run
observation, not proof no valid extraction can be rejected — the first-year
bug above was a concrete mechanism by which they were.)

### 5.3 The contradiction ablation could not be run

Across every eval run, `conflicts detected = 0`. The guard never fired, so the
ON/OFF arms execute identical code and comparing them would measure only
run-to-run variance.

That is a real result about the guard's scope rather than a gap in the
experiment. Conflicts require the same `(entity, metric, period)` to be extracted
twice with different values, which happens when the planner emits overlapping
sub-questions or when a second round re-extracts something round one already got.
On mostly-successful 2-hop questions, neither occurs. The guard is a safety net
for the harder path — which is where the original $4,239M failure lived — and a
sample weighted toward 4-hop questions with partial first-round failures is what
would exercise it.

For natural data that remains true — the eval has never produced a conflict on
its own. What changed is that the guard is no longer *unvalidated*: see the
fault-injection results below.

A 2026-07-29 audit added two structural reasons the count may read zero even
when disagreements exist, both now fixed: compound extractions
("FY2024 and FY2025") were keyed under their first year only, so the historical
$12,914M-vs-$4,239M pair could land in *different* groups and never be
compared; and any two non-numeric values always registered as a conflict, which
was masked only because groups never formed.

**The guard has now been validated by fault injection**
(`eval/run_conflict_injection.py`, the standard approach for guardrails that
never fire on natural data — arXiv 2504.00180). For one extraction per
question, a corrupted twin Evidence is injected at the extraction/critique
seam: value ×0.33 (mimicking the observed quarterly-for-annual failure),
**variant entity and period spellings** ("NVIDIA Corporation", "fiscal year
2024") so detection only fires if the normalizers actually group them, and a
real chunk id so `resolve` re-reads real passages. Same 5 questions, guard ON
vs OFF:

| | guard ON | guard OFF |
|---|---|---|
| conflicts detected | **5/5** | 0 (disabled) |
| corrupted figure in final answer | **1/5** | **3/5** |
| correct | 2 | 0 |
| wrong | 0 | 0 |
| abstain | 1 | 2 |
| ungradeable (compare echoes) | 2 | 3 |

(The OFF arm's ungradeable count previously read 2 here; the committed artifact
has 3, which is also why the arms' gradeable denominators differ — 3 vs 2.
These numbers describe the run saved in `conflict_injection.json`, which
predates this branch's resolver changes; the guard's behaviour under current
code has not been re-measured, as the rerun is API-bound.)

Three readings. First, **detection recall through the full production path is
5/5** — grouping survived the variant spellings, which is precisely the seam
the audit had flagged as the weak link. Second, the guard's measured value is
the contamination line: without it the fabricated figure reaches the user 3
times in 5; with it, once — and that once is an abstention *quoting* the
unresolved conflict rather than asserting the figure. Third, the OFF arm shows
what an unguarded synthesizer does with two equal-looking values: zero correct
answers — it either propagates the corruption or collapses into a hedge.

Honesty: n=5 per arm, one synthetic corruption pattern (×0.33), and two
compare questions graded "ungradeable" in both arms (the synthesis echoed the
comparison rather than stating a verdict). This validates the mechanism, not
a rate.

### 5.4 The grader had the same disease as the thing it graded

The same audit found the answer grader could be satisfied **by a citation
alone**: `_NUM` extracted "10" from "10-K" and "8" from "Item 8", and both sat
inside the percent tolerance of real golds — `"See the 10-K filing."` graded
*correct* against a gold of 9.896%. Also confirmed: a correct negative delta
phrased as "decreased by 3.36%" (or with a Unicode minus) graded wrong — only
an ASCII minus scored; a complete, correct comparison with an honest hedge
graded abstain because the abstain check ran first; and the comparative-cue
scan was blind to "lower/less/smaller" and to sentences that merely echo the
question.

The grader was rewritten (citations and form tokens stripped before number
extraction; percent golds only match numbers stated as percentages;
direction-word sentences contribute the negated value; conclusion graded before
the abstain check; inverse and negated comparatives handled) and pinned with 18
regression tests, each replaying a confirmed failure.

Re-grading the stored answers offline:

| | old grader | fixed grader |
|---|---|---|
| verdicts (5 answerable) | 4 correct / 1 abstain | **unchanged** |
| null-model hit rate | 0.60 | **0.20** |
| numbers per answer | 21.8 | 6.6 |

The verdicts survived because the agent's real answers contain genuine
percent-stated figures — the holes were exploitable, not exploited. The
important movement is the null rate: 0.60 was mostly the grader matching
citation digits against other questions' golds. Under the fixed grader the
margin over null is 0.80 vs 0.20, which is the first version of this metric
that means something.

### 5.5 Scaling to n=30 — and the infrastructure bug it flushed out

The eval was then run at n=30 (24 answerable across all four shapes + 6
unanswerable controls). The first pass produced a shocking-looking table —
7 wrong, 4 of 6 controls fabricated — that dissolved on inspection: **14 of
30 answers were empty strings.** During an API bad patch, deepseek returned
empty completions; the synthesize fallback chain only advanced on
*exceptions*, so an empty "success" became the final answer; and the grader
scored silence as wrong (answerable) or fabricated (controls). The agent
never said the things it was being marked down for.

Two fixes (an empty completion now advances the chain; an empty answer now
grades as `empty_answer`, never wrong/fabricated), and the 14 lost questions
re-run:

```
answerable (n=24):  20 correct   2 wrong   2 abstain
                    accuracy 0.83, Wilson95 [0.64, 0.93]
controls   (n=6):   6/6 abstained, Wilson95 [0.61, 1.00]
conflicts on natural data: still 0
```

The two genuine wrongs: one 4-hop trend question, and one comparison where
the agent failed to retrieve AAPL's operating income and said a comparison
could not be made (arguably an abstention the grader's phrasing list does not
recognise). Accuracy at 0.83 sits above the null control, but by less than
first written here: recomputed offline on the same 24 answerable questions the
accuracy uses, the null is **12/24 = 0.50** — all null hits fall on answerable
questions, since an abstaining control emits no numbers to collide, so the
earlier ~0.2–0.3 (and any 12/30 = 0.40 phrasing) understates it. 0.83 vs 0.50
is a real margin, not a comfortable one; the null is this high because gold
values collide across questions. Every control abstained — the period guard held at 6/6 rather than the
2/2 previously reported.

The episode is the project's thesis in miniature: the dramatic result was an
artifact, the boring result survived, and the difference was only visible
because empty answers were *inspected* rather than trusted as verdicts.

### 5.6 Optional: dual specialized critics (deep-research technique, A/B'd)

Deep-research (2026-07-30) surfaced a strongly quantified critic-agent result
on financial numeric QA (ICAIF '24, doi:10.1145/3677052.3698686): a critic
reviewing reasoning and answer adds +15% accuracy for an 8B model / +5% for a
70B, and **two specialized critics — one verifying figures against evidence,
one verifying calculation — beat one general critic** (FinQA 54.7% → 64.1% →
72.5% for the 8B). The split maps directly onto this agent's architecture, so
it was implemented as an optional post-synthesis pass
(`MultiHopAgent(verify_answer="dual")`, eval flag `--verify dual`): two
critics, and one revision round if either objects. Default off — it costs 2–3
extra model calls per question, and the paper's own data says gains shrink as
the base gets stronger.

A/B on the 10 answerable questions (same qids as the n=30 baseline subset):

| | baseline | dual critics |
|---|---|---|
| correct | 9/10 | **10/10** |
| wrong | 1/10 | 0/10 |
| seconds | ~600 | 864 |

The flipped question is the baseline's honest-miss comparison (failed AAPL
retrieval → "cannot compare"). Read this with the discipline the rest of the
report demands: single runs of a nondeterministic agent, n=10, and the flip is
equally consistent with retrieval simply succeeding on the rerun. What the A/B
does establish: the critics cost ~40% latency, broke nothing (no correct
answer was argued into a wrong revision — the published failure mode of
critique loops), and the ceiling at this baseline (0.83–0.90) leaves little
headroom, exactly as the paper's strong-model trend predicts. The mode is
worth its place as an option; the evidence does not support making it the
default.

### 5.7 Strict conclusion-sentence grading: a load-bearing null result

The last standing metric caveat was "a matching figure *anywhere* in the
answer counts". A strict mode was added (`--grade strict`): only the answer's
conclusion sentence — the first sentence stating a percent figure — is
eligible to match. Re-grading all 30 stored answers:

```
loose : 26 correct  2 wrong  2 abstain
strict: 26 correct  2 wrong  2 abstain    (zero verdicts changed)
```

The looseness existed but was never being exploited: this agent's synthesis
prompt leads with the answer, so the conclusion sentence and "anywhere in the
answer" coincide in practice. The caveat retires with data rather than with a
promise, and strict mode remains available for agents with chattier outputs.

---

### 5.8 Extraction grounding, and an optional corrective-retrieval hop

Two changes to the agent itself, one measured in the only sense available
offline (it fails a reproduction that used to pass) and one deliberately
unmeasured.

**Every numeric extraction is grounded.** §5.2's fabrication was consistent
and therefore invisible to contradiction detection; the guard for it was
`period_mismatch`. A second, simpler fabrication had no guard at all: the
extractor returns `{"found": true, "value": "$999 million", "passage_number":
1}` against a passage that says $100 million, and the citation to passage 1
lends the invented figure a source. The resolver has refused that shape since
§5.4 (`value_in_passage`, over exactly the text it showed the model); ordinary
extractions now go through the same check, over exactly the 1,500 characters
the extractor was shown. A rejected figure becomes "not found" for synthesis
and is kept on the trace as `ungrounded`, so the next run can report how often
it happened. Numeric values only — a textual value would face a verbatim
substring test, and rejecting every paraphrase is a different error. The
saved n=30 traces predate this rule; the rates in §5.5 are what the agent
did without it.

**Optional corrective retrieval** (`--retrieval-grader`; Corrective RAG,
Yan et al., arXiv 2401.15884). A small model grades each hop's passages as
relevant / ambiguous / irrelevant for the sub-question before extraction.
Relevant: keep the relevant and ambiguous passages, drop the rest. Ambiguous
only: keep them and add one reformulated query's hits. Nothing usable:
re-query once with the agent's own filing-vocabulary reformulation and grade
that; if it is no better, hand the extractor the original passages, so the
grader narrows and never starves a hop. No web fallback — the paper's — since
the eval's premise is that the answer is in the corpus or the agent must say
so. It costs one small call per hop, is tested offline through injected
calls, and has **not** been run ON/OFF against the hosted API; the trace
records the action per hop (`retrieval_action`) so that comparison can be
stratified when it is made. No number is claimed for it.

## 6. Engineering notes

**Model availability is not stable.** Mid-development, two models went from
working to dead with no deprecation warning:

```
qwen/qwen3-next-80b-a3b-instruct  ->  HTTP 410 "end of life on 2026-07-27"
qwen/qwen3.5-397b-a17b            ->  HTTP 410 "end of life on 2026-07-27"
moonshotai/kimi-k2.6              ->  404 for this account
google/gemma-4-31b-it             ->  timeout at 45s
```

This is why every model call goes through `chat_json_chain`, a validating
fallback chain, rather than a hard-coded id. (Almost every call: the final
synthesis step turned out to be the one hard-coded single-model call left,
meaning a model EOL would fail the run at the last step after every hop had
succeeded. Found in audit, now on the same chain.)

**Hosted NIM has no constrained decoding.** `guided_json` / `nvext` exist only
for self-hosted NIM containers, so JSON is parsed defensively and validated per
call, with the chain advancing when a model produces an unusable shape.

**Reasoning suppression differs per model family.** `/no_think` for Qwen;
`chat_template_kwargs.thinking=False` **plus** a "detailed thinking off" directive
for Nemotron, because the directive alone silently stops working on long prompts.
Some models put their actual answer in `reasoning_content` rather than `content`,
so the client accumulates both.

**A validator that is too strict cascades.** An early version rejected any plan
with more than 6 sub-questions outright, which burned the next model in the chain
on output that was merely over-eager, and eventually fell through to a dead
fallback. It now validates shape only and truncates.

**Per-hop extraction is parallelised.** Sub-questions in a hop are atomic and
independent by construction, so extracting them serially just sums latency.
This matters more than it looks: observed per-call latency on this API varies
from 4s to 50s for identical call shapes, so a four-hop question was inheriting
the sum of four worst cases rather than the max. Bounded at 4 workers, because a
burst that trips a 429 costs more in retries than the parallelism saves.

**A stalled SSE stream never trips a read timeout.** httpx's timeout is
per-chunk; a stream trickling keepalives with no content blocks indefinitely and
presents as a hang with no error and no output. Batch jobs over this API need a
wall-clock deadline per item.

---

## 7. Limitations

- **The conflict resolver is itself an LLM call.** It fails to "cannot determine"
  rather than to a wrong number, which is the right failure direction, but it is
  not a guarantee.
- **Contradiction detection only covers numeric values.** Non-numeric claims
  ("management cited supply constraints") are grouped but not compared.
- **Bounded at `max_rounds=2`.** A self-correcting loop without a budget is a way
  to spend money slowly.
- **The eval is small.** The headline is n=30 (24 answerable + 6 controls);
  the original 5 answerable + 2 controls survives only as the grader-fix
  story in §5.4. The harness is the durable artifact; the numbers are a first
  measurement, not a benchmark.
- **The primary metric improved but remains an upper bound.** The original
  null-model hit rate of 0.60 against accuracy 0.80 turned out to be mostly the
  grader's own citation-digit bug (§5.4); under the fixed grader the null was
  0.20 on those five questions, and is **0.50 over the n=30 headline's 24
  answerable questions** — the honest comparison for the 0.83. The remaining looseness: grading still accepts a matching figure
  anywhere in the answer rather than only in the headline sentence, and several
  golds sit within tolerance of each other (two NVDA deltas differ by 1.46%
  relative, inside the 2% window), so correct/wrong is not distinguishable for
  those pairs. Tightening to the stated conclusion is still the right next
  step.
- **The headline sample is now n=30** (§5.5): answerable accuracy 0.83 with a
  Wilson interval of [0.64, 0.93], controls 6/6. Still one corpus, one run per
  question of a nondeterministic agent, and 53 unused questions in the eval
  set.
- **An API outage was indistinguishable from honesty.** Failed extractions
  were swallowed into "not found" with no logging, so during an outage the
  agent abstained everywhere and the unanswerable controls passed vacuously.
  Extraction errors now travel on the evidence (`Evidence.error`) and the
  grader reports them as their own outcomes (`abstain_due_to_error`,
  `empty_answer`) rather than folding them into abstention; the n=30 headline
  was graded under that rule.
- **The contradiction check is validated only under synthetic faults** (§5.3):
  detection 5/5 and contamination 3/5 → 1/5 under injection, but zero conflicts
  have ever fired on natural data, and one corruption pattern (×0.33) is not a
  distribution of real failure modes.
- **Grading rewards the right number appearing, not the right reasoning.** An
  answer that reaches the correct ratio through two compensating errors scores
  correct.

## 8. Conclusion

The useful finding is about where to spend defensive effort. Guarding against
fabrication-from-nothing is well covered by separating extraction from synthesis
and making "not found" a first-class answer — and this agent got that right on
its first real run.

What that design does *not* protect against is the agent retrieving the right
evidence and then using it wrongly. Two distinct shapes showed up, and they need
different defences:

**Disagreement** — the same figure extracted twice with different values, then
arbitrated badly. Caught by grouping extractions on `(entity, metric, period)`
and refusing to let the synthesizer choose. Unit-tested; not yet observed
end-to-end.

**Uniform mislabelling** — every extraction agreeing on a false premise, such as
FY2024 figures returned under a fiscal-2018 label. *Invisible* to any
cross-check, because there is nothing to disagree with. Caught only by comparing
each extraction against the question's own stated period.

Both look like competence from the outside: real citations, real chunks, correct
arithmetic. The distinguishing feature of the second is that it was *encouraged
by the prompt* — "take the value from the column for the period you were asked
about" presupposes that column exists. An instruction that assumes the answer is
present will manufacture one.

And the most useful thing the eval did was catch itself. Its first verdict on
this agent — 40% wrong — was an artifact of the SEC API's `fy` field, not of the
agent. A measurement built to catch a model reading the wrong year was itself
reading the wrong year.
