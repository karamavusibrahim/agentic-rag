# Audit — evaluation claims and verification grounding

A pass over what this repository *claims* against what its code and committed
artifacts actually support. Every finding below was reproduced offline from
files in the repo; nothing here required a new API call, and no measured result
was deleted or re-run to make it look better.

## Fixed

### 1. A resolved conflict could be labelled "verified" without being verified

`src/agentic_rag/agent/contradictions.py` — `resolve()`

The resolver asks a model which of two disagreeing figures is correct, then
renders the winner to the synthesizer as `-> {value} (verified; conflict
resolved)`. It accepted whatever `correct_value` came back **without checking
that the figure appears in the passage the verifier cited**. A hallucinated
number therefore entered synthesis wearing a stronger label than the two real
extractions it replaced — the exact failure the guard exists to prevent,
reintroduced one layer down.

Two paths, both closed:

- `passage_number` outside the range shown to the model fell through to
  `hits[0]`, attaching a real citation to a value that may have come from
  anywhere. It now leaves the conflict unresolved.
- The value is now required to occur in the cited passage, compared by
  magnitude rather than string so that `"$12,914 million"` still matches a
  table printing `12,914` under an "in millions" caption (`value_in_passage`).
  Non-numeric values fall back to a substring test.

### 2. The extractor treated the string `"false"` as a successful extraction

`src/agentic_rag/agent/loop.py`

`if not data.get("found")` is a truthiness test. Hosted models return
`"found": "false"` as a *string* often enough to matter, and a non-empty string
is truthy — so a refusal became a positive extraction, and `str(False)` then
rendered the value as the literal text `"False"` into the synthesis prompt.
Now requires `is True`.

Two adjacent holes closed at the same time:

- `found=True` with an empty value was representable, putting `None` in front
  of the synthesizer with a real citation attached. Now returns not-found.
- The same out-of-range `passage_number -> hits[0]` fallback as above.

### 3. Fault-injection denominators hid the real sample size

`eval/run_conflict_injection.py`, `README.md`

`grade()` also returns `"ungradeable"`, which the arm summary never counted, so
`correct + wrong + abstain` did not equal `n` and the two arms had **different
gradeable denominators** — 3 with the guard on, 2 with it off. The README
quoted "correct answers 0 → 2" off those unequal bases.

Also: 4 of the 5 questions ask about R&D FY2024 in different shapes
(ratio/delta/compare/trend), so they are variants of one fact rather than five
independent trials; and `detection_rate` divided by `max(len(inj), 1)`, which
reports `0.0` — indistinguishable from "the guard failed" — when nothing was
actually injected.

The summary now reports `n_gradeable`, `ungradeable`, `errors`, `detected`,
`distinct_facts`, and `None` (not `0.0`) for a detection rate over zero
injections. The README states the direction rather than the effect size.

## Examined and left alone

- **The injection design itself is sound.** Both arms run the same questions
  through the same retriever with the same corruption; only
  `check_contradictions` toggles. The weakness is not the arm construction, it
  is that the agent is re-run per arm rather than replayed, so extraction is
  re-sampled and the twin is not guaranteed to attach at the same seam. Noted
  in the README; fixing it means recording and replaying trajectories.
- **Detection 5/5 vs 0/5** is honest for the ON arm and definitional for the
  OFF arm, where the detector is disabled. Labelled as such rather than removed.
- Test counts are stated in the third-pass section below; earlier figures in
  this file were wrong.

## Not fixed — needs a run this audit did not make

- The claimed null-model margin could not be reproduced from the committed
  traces. Re-grading `eval/results/traces_n30_merged.json` offline gives
  26 correct / 2 wrong / 2 abstain. Confirming or retracting the margin needs
  the null-model arm re-run, which is an API-bound job.


---

## Second pass — corrections to this audit

The fixes above were reviewed independently against `main`. That review found
the central guard was itself exploitable, and that the tests could not have
caught it. Both are fixed here.

### `value_in_passage` accepted three classes of hallucination

The first version special-cased zero by searching for the character `"0"`, and
multiplied every passage number by 1e3/1e6/1e9 to guess an implicit
"in millions" caption. All three of these returned **True**:

```python
value_in_passage("$0", "Fiscal 2024 revenue was $100 million")
value_in_passage("$12.914 trillion", "Revenue was $12.914 billion")
value_in_passage("$2.024 million", "FY2024 revenue")
```

The last is the clearest: a fiscal *year* scaled by 1e3 becomes a plausible
dollar figure. A guard written to stop a hallucinated number from being
labelled "verified" would have verified one.

Now: implicit scaling applies only to a passage token carrying **no** scale word
of its own (so "billion" is never re-scaled into "trillion"), never to a bare
four-digit year, and zero is matched as a parsed numeric token rather than a
character. A genuine `$0` in a passage still grounds a `$0` claim.

### `passage_number: true` selected passage 1

`isinstance(True, int)` is `True` in Python, so a boolean passed the range check
in both `resolve()` and the extractor — restoring the silent `hits[0]` fallback
the guard was written to remove. Both now reject booleans explicitly.

### A legitimate `value: 0` was thrown away

`str(data.get("value") or "")` maps `0` to `""`, so an extraction reporting zero
became not-found. Zero is a real answer. Only `None`, blank strings, and
booleans are missing values now.

### The tests could not have caught any of this

All five original tests called `value_in_passage` directly. Reverting the actual
`resolve()` and extractor changes left every one of them green — they tested a
helper, not the path that runs. `tests/test_resolve_behaviour.py` now drives
`resolve()` through a stubbed model, and **3 of its 6 tests fail if the fixes
are reverted** (verified, not assumed).

## Findings from that review left open

Recorded rather than quietly dropped:

- **The injection arms are worse than "not paired".** The two arms received
  *different* corruptions ($10.35B ON vs $129.04B OFF on the ratio question),
  because extraction is re-sampled per arm and the twin attaches to whichever
  extraction returns first. Fixing it means recording and replaying
  trajectories; the README states the direction only.
- **"Four variants of one fact" was wrong.** The five questions draw on at
  least five distinct atomic figures. They share one AAPL R&D value; they are
  correlated, not duplicates.
- **`main` had 59 tests, not 69.** The earlier count credited this branch's own
  additions to the baseline. The suite is now **87**: 59 inherited plus 28
  added across this branch.
- **The null-model margin is computable offline** (12/30 = 0.40 over the merged
  n=30 traces). The claim earlier in this file that it needed an API rerun is
  withdrawn.
- **Ordinary extraction is still ungrounded.** Only conflict *resolution* checks
  its value against the cited passage. The far more common path does not, and
  that is the larger remaining hole.


---

## Third pass

### A value the verifier never saw could still be "verified"

`resolve` shows the model `hit.text[:1200]` and validated the answer against
the **full** passage. A figure sitting past the truncation point could be
returned and then confirmed by text the model never read — a coincidence
dressed up as verification. The window is now one constant, `VERIFY_CHARS`,
used for both, and a test buries `$999 million` past the cut and asserts the
conflict stays unresolved.

### Implicit scaling still let an employee count stand in for a revenue figure

Reading a bare `12,914` as `$12,914 million` is only justified when the passage
says it reports in thousands or millions. Applied unconditionally, `"the
company had 100 employees"` grounded a claim of `$100 million` — the same shape
as the bug this guard exists to stop. Implicit scaling now requires an explicit
"in millions"/"in thousands" caption in the passage, and uses that caption's
scale rather than trying all of them.

### The extractor still had no test of its own

Reverting the extractor's truthiness, boolean-index or zero handling left every
test green, because all of them called helpers. `test_resolve_behaviour.py` now
drives `MultiHopAgent.extract` through a stubbed model for all six cases.

Suite: **87 tests** (59 inherited from `main`, 28 added on this branch).

## Still open after three passes

- **Ordinary extraction is ungrounded.** `resolve` checks its value against the
  cited passage; `extract` does not, and it runs far more often. A model can
  return `$999 million` against a passage reading `$100 million` and have it
  cited. This is the largest remaining hole and needs the same treatment.
- **The injection arms are not paired** and the committed artifact predates the
  summary fields this branch added, so `conflict_injection.json` does not carry
  `n_gradeable` / `ungradeable` / `distinct_facts`. Regenerating needs API access.
- `distinct_facts` counts QID tokens, giving 3 where the five questions draw on
  seven atomic figures. README still says "~2 facts" and REPORT still says the
  OFF arm had 2 ungradeable rows where the artifact shows 3.
- README publishes a null-hit rate of ~0.2–0.3; recomputing over the same 30
  traces gives 12/30 = 0.40.
- README and REPORT still carry the superseded 5-question framing beside the
  n=30 headline.
- `relgrep` truncates to 100 chunks before reranking and does not fall back,
  so the only answer-bearing chunk can be discarded permanently.
- `Conflict.render()` omits `resolved_chunk_id`, so the synthesizer gets a
  "verified" figure with no source to cite.
- Compare-question grading uses mention order, so "Compared with MSFT, AAPL
  reported higher net income" grades wrong when the gold answer is AAPL.
