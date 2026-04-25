# S33-S1 Rescope — One-hot per-side mask requires investigation, not direct patch

**Date**: 2026-04-25
**Branch**: `mlx/sprint-33-iter2-scaffold`
**Source review**: `docs/sprint33/sprint-close/cr-report.md` (S33-S1)
**Source SA item**: `docs/sprint33/sprint-close/sa-report.md` (SA-N1, after CR re-scoping)

## Summary

The S33 code review (`cr-report.md`) flagged that the per-side mask fix shipped at
`csv_train.cpp:1941/1980` for the ordinal branch of `FindBestSplit` was missed at
the structurally similar one-hot branch at `csv_train.cpp:1698`. The original
recommendation was "apply the same pattern."

In-session validation (2026-04-25) applied the patch and ran a synthetic 8k one-hot
anchor under ST+Cosine. MLX-pre-fix vs MLX-post-fix:

| | iter=49 loss |
|---|---|
| MLX pre-fix (joint-skip) | 0.479101 |
| MLX post-fix (per-side mask) | 0.493401 |

Loss regressed ~3% post-fix on this anchor. No CPU CatBoost reference was available
to disambiguate "fix is correct, MLX-pre-fix was lucky" vs "fix is wrong for one-hot."
Patch was reverted. Working tree is clean.

## Why the syntactic argument is not sufficient

The one-hot and ordinal branches differ in the **semantics of degeneracy**:

- **Ordinal** (`csv_train.cpp:1980+`): bin k partitions partition p into a Left
  prefix (running sum up through bin k) and a Right suffix. Degeneracy
  (`weightLeft = 0`) is a *prefix transient* — eventually wL becomes positive as
  k sweeps. The non-empty side represents a real partition split with content.
- **One-hot** (`csv_train.cpp:1698+`): bin k means "split on feat == k". Left =
  exactly the docs in histogram bin k. Degeneracy (`weightRight = 0`) is the
  *static* fact "no docs in partition p have feat ≠ k" → all docs in p have
  feat == k → the candidate is the **trivial split**, not a real partition.

When `weightRight = 0` in one-hot, `weightLeft = totalWeight_p` and `sumLeft =
totalSum_p` (Left has full mass). The per-side mask injects
`totalSum_p² / (totalWeight_p + λ)` into cosNum for that (partition, k). Whether
this matches CPU depends on whether CPU's joint-cosine reducer subtracts a parent
term per partition (cancels exactly) or per (partition, k) (inflates joint sum).
This question requires inspecting `ComputeCosineGainKDim` and CPU's
`UpdateScoreBinKernelPlain` interaction in the cat-feature dispatch path. It was
not done in S33.

## What changed vs the original cr-report.md recommendation

cr-report.md S33-S1 recommended applying the per-side mask pattern at L1698. The
empirical regression falsifies the assumption that the syntactic argument carries
full force. The recommendation is updated to: **investigate fix shape via
S34-PROBE-F-LITE before any patch lands at L1698.**

## What S33's existing closure does and does not claim

- S33 closes DEC-036 / DEC-042 for the **ordinal branch** of `FindBestSplit`. This
  closure is supported by PROBE-E mechanism, four-gate parity validation
  (G4a–G4e), 1941× drift improvement, and CPU per-side mask reference match.
- S33 does NOT claim closure for the one-hot branch. The L1698 site is identified
  as a structurally similar `continue` rule in the same function but with
  different degeneracy semantics. Whether it carries the same DEC-042 bug class
  or a related-but-distinct class is the S34 question.
- DEC-042 status remains FULLY CLOSED with the ordinal-scope footnote added in
  S33-S2.
- The S28 guards (S28-ST-GUARD, S28-LG-GUARD) remain removed. They were
  measurement-tested at G4a–G4e on the ordinal path. One-hot Cosine has been
  shipping on the joint-skip rule throughout S28–S33; if PROBE-F-LITE finds a
  real divergence vs CPU, that may motivate a re-guard, which would be a
  separate ticket.

## S34-PROBE-F-LITE — scoped follow-up

Branch: cut `mlx/sprint-34-probe-f-lite` from S33 close commit (whichever S33-S1-RESCOPE
ends up at).

### Pre-work (must complete before any code change)

- **T0a (mathematician)** — Derive cosNum/cosDen per-(partition, k) contribution at
  `csv_train.cpp:1698` with explicit treatment of parent-term subtraction.
  Identify whether parent subtraction occurs inside or outside the per-bin loop.
  Write up at `docs/sprint34/probe-f-lite/math-derivation.md`.
- **T0b (ml-engineer)** — Read `csv_train.cpp:1690-1722` and `ComputeCosineGainKDim`;
  confirm or refute the per-bin parent-subtraction question raised in T0a.

### Probe (only if T0a/T0b leave the fix question open)

- **T1 (research-scientist)** — Build PROBE-F-LITE harness:
  - 3 anchors × 5 seeds × 2 builds (MLX-pre-fix, MLX-post-fix-per-side-mask), plus
    a third arm `MLX-post-fix-exclude-trivial` if T0 motivates it.
  - Anchor 1 (cosine-relevant): 50k docs, 5 cat features × {2,4,4,8,8} levels +
    5 numeric, ST+Cosine, 50 iters, depth=6, bins=128, lr=0.05, seeds={42, 1337,
    7, 17, 9999}.
  - Anchor 2 (degenerate-prone): 8k docs, 1 cat × 4 levels + 2 numeric (the
    in-session anchor), ST+Cosine, 50 iters, same seeds.
  - Anchor 3 (LG cosine): 50k docs, 4 cat × {4,4,8,8} + 4 numeric, LG+Cosine,
    same seeds.
  - CPU CatBoost reference (Python `catboost` package) on each anchor with
    matched config.
- **T2 (research-scientist)** — Per-partition counterfactual capture at L1698
  mirroring PROBE-E methodology, on at least one tree where one-hot degeneracy
  fires.
- **T3 (research-scientist + mathematician)** — Verdict at
  `docs/sprint34/probe-f-lite/verdict.md`: which of {per-side mask, exclude
  trivial, no-op} is correct, with evidence.

### Gate criteria

- **G-PFL-1** — At least one fix arm has MLX vs CPU drift within 2% on ≥4/5 seeds
  on Anchor 1 → that arm is the chosen fix.
- **G-PFL-2** — Per-partition counterfactual capture confirms the chosen arm
  matches CPU per-(partition, k) within fp32 precision floor.
- **G-PFL-3** — 18-config L2 sweep no-regression preserved (the existing G4d
  envelope must hold).

### Kill-switches

- **K-PFL-1** — All three fix arms regress on Anchor 1 vs CPU → escalate to full
  PROBE-F with reducer-level instrumentation; reopen DEC-042 with a "one-hot
  subclass" section; do NOT ship a fix to L1698 until subclass is named.
- **K-PFL-2** — CPU disagrees with itself across seeds (>2% drift CPU-vs-CPU on
  Anchor 1) → CPU harness configuration is wrong; halt and rebuild.
- **K-PFL-3** — Probe wall-clock exceeds 3 working sessions without verdict →
  escalate scope to a dedicated S34 sprint instead of probe-only.

### Out of scope

- Reverting Commits 1, 1.5, 2, 3a, 3b (those fix the ordinal path correctly).
- Reopening DEC-036 closure (closure stands for ordinal; one-hot is a separate
  finding).
- Removing S28 guards retroactively for one-hot (guards are already removed).

### Estimated cost

1–2 working sessions for T0+T1+T2+T3 if probe lands clean.

## Lesson recorded

The transferable lesson from this rescope: **syntactic-class arguments in adjacent
code paths are not sufficient evidence for fix-shape transfer**. PROBE-E proved
the mechanism for the ordinal path; the same `continue` line in the one-hot path
sits in a different mathematical context (different degeneracy semantics, possibly
different parent-subtraction structure) and required independent validation. This
will be appended to `LESSONS-LEARNED.md` if/when that file's location and
maintenance pattern is settled.
