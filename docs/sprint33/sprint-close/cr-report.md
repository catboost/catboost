# Sprint 33 Sprint-Close Code Review

**Branch**: `mlx/sprint-33-iter2-scaffold`
**Tip**: `c511549eeb` (HANDOFF.md says `d599e5b033` — see S33-S2)
**Date**: 2026-04-25
**Reviewer**: code-reviewer

## Verdict

**APPROVE-WITH-COMMENTS**

Counts — Must-fix: **0** / Should-fix: **3** / Minor: **5** / Affirmations: **6**.

The per-side mask fix at `csv_train.cpp:1941-2045` correctly implements CPU's `UpdateScoreBinKernelPlain` semantics for the FindBestSplit ordinal path under both L2 and Cosine. The kernel md5 invariant (`9edaef45b99b9db3e2717da93800e76f`) is preserved — this is a pure host-side fix. PROBE-E instrumentation is compile-gated cleanly. Guard removal is complete and symmetrical across all three layers (Python core, C++ nanobind, CLI). The DEC-012 atomicity discipline across the 5 commits is exemplary.

The fix as committed is correct and ships safely. The Should-fix items below address (1) one structurally-similar callsite that was not covered by the gate suite, (2) stale state-file statuses, and (3) doc bodies that still assert the invalidated DEC-041 quantization hypothesis. None block the merge of the structural fix; all should land before S33 is declared archived.

## Critical (blocking)

None.

## Major (Should-fix)

### S33-S1 — Per-side mask not applied to FindBestSplit one-hot path (`csv_train.cpp:~L1698`)

The fix at L1941-2045 correctly replaces the joint `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` rule with the per-side mask in the ordinal-feature branch of `FindBestSplit`. However, the one-hot/binary-feature branch at L1698 of the same `FindBestSplit` function still uses the old joint-skip rule. Because the bug class (DEC-042) is "degenerate-child skip in the per-partition update of FindBestSplit," and L1698 sits in the same outer per-partition loop with the same parent-term-subtraction structure, it has the **same cross-partition bug**.

The four formal gates exercise SymmetricTree+Cosine on adult/higgs (G4a/G4b/G4e) and the 18-config L2 sweep (G4d), all of which route through the **ordinal** branch under the bin-quantized layout used in `csv_train.cpp`. The one-hot branch is structurally **uncovered** by the gate suite.

For comparison, `FindBestSplitPerPartition` at L2293 (one-hot) and L2377 (ordinal) are **structurally immune**: each call evaluates a single partition in isolation, so a degenerate child only zeroes out that partition's own contribution — there is no cross-partition running sum to corrupt. The L1698 site does not have that immunity.

**Note**: This contradicts the security audit's SA-N1-S33 finding which flagged L2293/L2377 as the asymmetry site. CR analysis (single-partition evaluation, no cross-partition running sum) is the structural truth; SA's flag should be re-scoped to L1698.

**Suggestion.** Apply the same per-side mask pattern to L1698 and add a one-hot Cosine-on-ST counterfactual to the gate suite. Until that lands, ship S33 with a `KNOWN-LIMITATIONS.md` note flagging the L1698 path.

### S33-S2 — Stale state-file statuses

- `HANDOFF.md:7` — tip pointer `d599e5b033`; actual HEAD is `c511549eeb`.
- `HANDOFF.md:18` — "DEC-036 OPEN", "DEC-032 PARTIALLY CLOSED pending guard removal" — guards have been removed.
- `HANDOFF.md:29` — "DEC-036 ROOT-CAUSED... Status remains OPEN" — DEC-036 is now CLOSED via DEC-042.
- `DECISIONS.md:1387` — DEC-036 status block still reads "OPEN — investigation-phase". Contradicts DEC-042's "FULLY CLOSED 2026-04-25" at L1875.
- `DECISIONS.md:1661` — DEC-040 status still "OPEN — investigation phase".
- `DECISIONS.md:986` — DEC-032 status text still references "until S30-COSINE-KAHAN lands".
- `DECISIONS.md:1246` — DEC-035 status still asserts guards remain.

**Suggestion.** Single state-files-only commit:

1. Update HANDOFF.md tip to `c511549eeb` and rewrite the "Open Decisions" / "In Progress" sections.
2. Mark DEC-036 as **CLOSED — superseded by DEC-042**.
3. Mark DEC-040 as **CLOSED — investigation concluded; mechanism captured in DEC-042**.
4. Update DEC-032 and DEC-035 status footers to reflect post-guard-removal state.

### S33-S3 — Sprint-close and L4-fix verdict docs assert invalidated DEC-041 mechanism

`docs/sprint33/sprint-close.md` and `docs/sprint33/l4-fix/verdict.md` still describe the DEC-036 mechanism as **static-vs-dynamic quantization border divergence (DEC-041)**. DEC-041 is INVALIDATED: PROBE-E (2026-04-25) identified the actual class as the per-partition degenerate-child skip in FindBestSplit, captured under DEC-042. The retraction lives in the commit message at `9a9ba59a52` but the doc bodies were never updated.

A reader landing on `sprint-close.md` from the index will conclude that the shipped fix changed quantization, which is materially false — the fix is host-side per-side mask in score accumulation, with kernel md5 unchanged.

**Suggestion.** Rewrite both documents in place to describe DEC-042 as the closing mechanism, with an explicit "DEC-041 invalidated — see PROBE-E" callout. (The technical-writer agent has already drafted this rewrite for `sprint-close.md`; verify the new body lands.)

## Minor / Nit

- **S33-N1** — HANDOFF tip pointer one commit behind (covered under S33-S2; called out separately because it is the first read in the session protocol).
- **S33-N2** — Redundant `import os` in `tests/test_cli_guards.py` (each inverted test has `import os` inside the function body despite a module-level `import os`).
- **S33-N3** — P13 debug-print formula at `csv_train.cpp:L2174-2179` still uses joint-skip rule; printed values for partitions where exactly one side is degenerate will diverge from runtime accumulation. Debug-only path.
- **S33-N4** — `mlx_skipped` PROBE-E CSV column emitted as int rather than bool.
- **S33-N5** — DEC-040 *header* line in `DECISIONS.md` (the title row that index-views render) still reads "investigation phase". Update header alongside body.

## Affirmations

- **PROBE-E counterfactual instrument quality.** Per-(feat, bin, partition) capture comparing MLX's actual rule against CPU's reference rule, compile-gated under `PROBE_E_INSTRUMENT`, is the right tool for a class-of-bug that pure end-to-end gates could not isolate.
- **DEC-012 atomicity discipline across the 5-commit landing.** Each commit carries exactly one structural change. Exemplary.
- **Four-gate report reproducibility.** `docs/sprint33/commit2-gates/REPORT.md` documents G4a/G4b/G4c/G4d/G4e with seeds, configs, and observed numbers.
- **K4 fp64 widening preserved correctly across the fix.** `cosNum_d` / `cosDen_d` widening intact in the rewritten accumulation block.
- **Comprehensive test inversion.** All four `test_cli_guards.py` cases inverted from rejection to acceptance; module docstring updated.
- **Kernel md5 invariance maintained.** Pure host-side fix; kernel md5 `9edaef45b99b9db3e2717da93800e76f` unchanged.

## Report-back

- **Verdict**: APPROVE-WITH-COMMENTS
- **Critical-issue count**: 0
- **Top-3 highlights**:
  1. **S33-S1** — Per-side mask not applied to the one-hot branch of `FindBestSplit` at `csv_train.cpp:~L1698`; same cross-partition bug class as DEC-042, structurally uncovered by the gate suite. SA's L2293/L2377 flag is incorrect (those sites are single-partition and immune); the real uncovered site is L1698.
  2. **S33-S2** — HANDOFF.md tip pointer is `d599e5b033`, actual HEAD is `c511549eeb`; DEC-036/DEC-032/DEC-035/DEC-040 statuses across state files still reflect pre-closure state.
  3. **S33-S3** — `docs/sprint33/sprint-close.md` and `docs/sprint33/l4-fix/verdict.md` still describe the DEC-036 mechanism as the invalidated DEC-041 quantization-border hypothesis (the technical-writer's rewrite of `sprint-close.md` should resolve half of this).

## Addendum (2026-04-25, post-empirical)

S33-S1's recommendation in this report ("apply the same per-side mask pattern to L1698") was based on the syntactic argument alone. A subsequent in-session smoke test applied the patch and ran a synthetic 8k one-hot anchor (1 cat × 4 levels + 2 numeric, ST+Cosine, 50 iters, depth=6, bins=32, lr=0.05, l2=3, seed=42, max-onehot-size=8). MLX-pre-fix iter=49 loss = 0.479101; MLX-post-fix iter=49 loss = 0.493401. Loss went up ~3% post-fix on this anchor. The patch was reverted; the structural argument is no longer sufficient evidence for the L1698 fix. The advisory board (math + research-scientist + strategist + visionary) recommended Path B: investigate before fix. The recommendation is amended from "apply same pattern" to **"investigate via S34-PROBE-F-LITE; do not patch L1698 until math-first analysis (parent-term subtraction structure) and CPU CatBoost reference probe land verdict."** SA-N1's reroute to L1698 is correspondingly amended to "L1698 and the related one-hot per-bin parent-subtraction logic, pending PROBE-F-LITE verdict."
