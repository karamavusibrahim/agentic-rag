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
- The existing 69 tests pass unchanged; 5 were added for the grounding fixes.

## Not fixed — needs a run this audit did not make

- The claimed null-model margin could not be reproduced from the committed
  traces. Re-grading `eval/results/traces_n30_merged.json` offline gives
  26 correct / 2 wrong / 2 abstain. Confirming or retracting the margin needs
  the null-model arm re-run, which is an API-bound job.
