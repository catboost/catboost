# S34-PROBE-F-LITE Verdict

**Date**: 2026-04-25
**Branch**: `mlx/sprint-34-probe-f-lite`
**Sprint scope**: investigate per-side mask validity at `csv_train.cpp:1698` (one-hot branch of `FindBestSplit`)
**Outcome**: math-first analysis closed the question without empirical probe (T1 not run)

## Verdict

**Per-side mask is WRONG for one-hot Cosine. Leave `csv_train.cpp:1698` as joint-skip.**

The de-facto state after the in-session S33-S1 patch revert is correct. CR-S33-S1 and SA-N1's L1698 reroute are resolved as **no-fix-needed**.

## Convergence

T0a (mathematician, `math-derivation.md`) and T0b (ml-engineer, `code-reading.md`) ran in parallel and converged independently on the same answer:

- **Cosine has no parent-term subtraction anywhere.** Verified in MLX (`csv_train.cpp:1715-1717, 2004-2005`, finalized at `:1731 / :2052` by `cosNum_d / sqrt(cosDen_d)`) and CPU (`score_calcers.h:58`, `short_vector_ops.h:155-175`). Inline comment at `csv_train.cpp:1568-1569` is explicit: "No parent-node subtraction (unlike L2). Cosine is an absolute score, not a differential gain."
- **The advisory board's per-partition / per-(p,k) / neither trichotomy collapses to "neither"** because Cosine is structurally parentless.
- **Per-side mask at L1698 would inject** `totalSum_p² / (totalWeight_p + λ)` into `cosNum_d` (and analogous into `cosDen_d`) for every (p, k) cell where the candidate's bin is absent from partition p — a real signal-attribution change, not a parent-term accounting bug.
- **The injection is bin-dependent**: rare categories accumulate larger baseline-mass contributions than common ones → **structural argmax bias toward rare-category bins** → directional regression.
- **One-hot is structurally degenerate-prone**: at depth ≥ 1 a one-hot category is sparse across partitions, so degenerate (p, k) cells are common, amplifying the bias.

This mechanism predicts a directional regression. The S33-S1 in-session smoke (8k synthetic anchor, single seed) showed loss 0.479101 → 0.493401 = +3% post-fix, exactly the predicted direction.

## Why the L1980 (ordinal) fix was correct but L1698 (one-hot) is not

The shipped DEC-042 fix at L1980 covered the **ordinal-feature branch** of `FindBestSplit`. That branch has different degeneracy semantics:

- Ordinal degeneracy is a **prefix transient** — wL = 0 only when the running prefix is empty. The non-empty side carries real partition content.
- One-hot degeneracy is a **static fact** — wR = 0 means "no docs in partition p have feat == k", i.e., the candidate is the trivial split for this partition. The non-empty side has the full partition mass — but the candidate isn't actually splitting anything in p, so contributing its full mass to the joint score biases argmax.

The same `continue` line, two different bug classes. The advisory board's @visionary frame was correct: DEC-042 covered the ordinal mechanism; the one-hot mechanism (had it been a bug) would have been its own DEC. Since the joint-skip behavior at L1698 already matches CPU's effective behavior at degenerate (p, k) for one-hot Cosine, **there is no bug to fix**.

## Confidence

**High.** Cosine's parent-termless structure is verified in 4 places. The bin-dependence of the bias is exact algebra (T0a §5.4-5.5). The mechanism is monotone in `|D(bin)|` (number of degenerate partitions for bin) and one-hot has structurally large `|D(bin)|`.

To upgrade to **certain**, two open questions remain:

- **Q1** — Does CPU's one-hot Cosine path filter degenerate candidates upstream (before reaching `UpdateScoreBinKernelPlain`)? Walk `CalcScoresForLeaf` in CPU.
- **Q2** — Pre-S33 parity report for one-hot Cosine: was the gate bit-equal? If yes, that implies CPU=joint-skip behaviorally and the verdict is certain.

Neither Q1 nor Q2 changes the recommendation. Both would only confirm the "leave it" verdict's confidence level.

## Side finding (separate ticket)

`math-derivation.md` §8.2 Q4 flags an unrelated potential bug:

> MLX's L2 path subtracts `totalSum² / (totalWeight + λ)` per-(p, k) at `csv_train.cpp:1704` (one-hot) and `:1973` (ordinal); CPU's L2 path (`score_calcers.cpp:20-33`) subtracts no parent term.

At depth=0 with numPartitions=1, this is a constant offset (argmax-invariant). At depth ≥ 1 or numPartitions > 1, it becomes per-(feat, bin) variable and may bias argmax in MLX vs CPU.

Currently papered over by the 18-config L2 parity gate (G4d 18/18 in [0.9991, 1.0008]). Papered over because the parity gate measures aggregate RMSE drift, not split-selection agreement. A split-selection harness (S31-T3b style) at depth ≥ 1 would surface or refute this.

**Opens as new ticket #128 — S35-Q4-L2-PARENT-TERM** (or wherever it lands; depends on sprint planning).

## Resolves

- **CR-S33-S1** (`docs/sprint33/sprint-close/cr-report.md` Major Should-fix): resolved as no-fix-needed. The L1698 site is structurally identified but the per-side mask transformation does not apply because Cosine is parentless. Joint-skip is the correct behavior at this site.
- **SA-N1-S33** (`docs/sprint33/sprint-close/sa-report.md` Info): the asymmetry between `FindBestSplit` (one-hot) and the shipped fix is intentional, not an oversight. SA-N1 closed.
- **DEC-042 ordinal-scope footnote** (added in S33-S2): the footnote correctly documents that DEC-042 closure scope is the ordinal branch. The one-hot branch was investigated under S34 and found to be already correct under joint-skip. No change to DEC-042 status.
- **TODO #127 S34-PROBE-F-LITE**: closed.

## Does NOT resolve

- **Q1 / Q2** above — open questions for a future sanity probe if needed
- **Q4 (L2 parent-term divergence)** — opens as new ticket
- **Pre-S33 one-hot Cosine parity gate** — was never run; if needed, opens as a separate verification task

## Files in this verdict

- `docs/sprint34/probe-f-lite/math-derivation.md` (T0a, mathematician, 465 lines) — formal derivation
- `docs/sprint34/probe-f-lite/code-reading.md` (T0b, ml-engineer, 307 lines) — code structure verification
- `docs/sprint34/probe-f-lite/verdict.md` (this file)

## What ships

S34-PROBE-F-LITE closes with **no code change** — the math-first analysis showed the in-session revert was correct and no further fix is warranted at L1698. The single commit is docs-only, capturing the analysis for future agents who might re-encounter the L1698 `continue` line and consider applying the L1980 pattern.

## Lessons

1. **Bug-class identity is a property of the math, not the syntax.** Two `continue` lines in adjacent branches of the same function can encode different bug classes if the surrounding mathematical structure (parent-term subtraction in this case) differs. CR-S33-S1's syntactic argument was a reasonable starting point but the math-first analysis was needed to determine fix shape.
2. **Empirical regressions on synthetic anchors are real signals.** The 3% loss bump on the in-session smoke was not noise; it was the predicted directional regression from a structurally biased argmax. The advisory board's "Fix Properly Always means investigate to determine the proper fix" framing was the correct response.
3. **Math-first can close probe questions cheaply.** S34's full PROBE-F-LITE plan budgeted 1-2 working sessions for T0+T1+T2+T3. T0 alone (math-derivation + code-reading, ~2 hours total agent time) closed the question. T1 (3-anchor empirical sweep with CPU reference) was not needed.
