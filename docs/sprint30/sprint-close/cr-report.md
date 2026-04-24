# Sprint 30 Close-out — Code Review Report

**Branch:** `mlx/sprint-30-cosine-kahan`
**Tip:** `24a0e829b8`
**Reviewer:** code-reviewer (senior staff engineer)
**Date:** 2026-04-23

## Verdict: APPROVE (PASS-WITH-NITS)

The two code-change commits under review (K4 fp64 widening bundled inside
`afdfbc0af8`, and Fix 2 `90a0cb4475`) are correct, well-scoped, well-commented,
and respect DEC-012 atomicity to a practically acceptable degree. The
instrumentation in `c1f1ff0e26` is cleanly gated behind
`COSINE_RESIDUAL_INSTRUMENT` / `COSINE_D2_INSTRUMENT` / `COSINE_T3_MEASURE`
macros with zero runtime impact on shipping release builds. No must-fix
findings. Five should-fix nits and minor observations are listed below for
follow-up.

---

## Scope reviewed

Code commits on `mlx/sprint-30-cosine-kahan ^master`:

| Commit | Purpose | Lines (ins/del) |
|---|---|---|
| `c1f1ff0e26` | T1 instrumentation (compile-time gated) | +318 / -0 |
| `afdfbc0af8` | T2-KAHAN Neumaier + K4 fp64 widening of `cosNum_d`/`cosDen_d` | +108 / -60 |
| `90a0cb4475` | Fix 2 — widen gain scalar and argmax to fp64 | +53 / -37 |

All code changes are confined to `catboost/mlx/tests/csv_train.cpp`
(standalone CLI driver). The production path (`catboost/mlx/train_api.cpp`,
`catboost/mlx/gpu_data/gpu_structures.h`, `python/catboost_mlx/core.py`) is
untouched. Metal kernels, per the task scope, were not reviewed.

## Guard posture (confirmation per deliverable)

All three S29-scope guards remain in place and un-weakened:

| Guard site | File:line | Status |
|---|---|---|
| Python `_validate_params` ST+Cosine | `python/catboost_mlx/core.py:638–647` | unchanged |
| Python `_validate_params` LG+Cosine | `python/catboost_mlx/core.py:628–636` | unchanged |
| C++ `TrainConfigToInternal` ST+Cosine | `catboost/mlx/train_api.cpp:41–54` | unchanged |
| C++ `TrainConfigToInternal` LG+Cosine | `catboost/mlx/train_api.cpp:24–35` | unchanged |
| CLI `ParseArgs` ST+Cosine | `catboost/mlx/tests/csv_train.cpp:398–410` | unchanged (gated bypass only under `COSINE_RESIDUAL_INSTRUMENT` or `COSINE_T3_MEASURE`) |
| CLI `ParseArgs` LG+Cosine | `catboost/mlx/tests/csv_train.cpp:380–391` | unchanged (gated bypass only under `COSINE_T3_MEASURE`) |

`git diff master...mlx/sprint-30-cosine-kahan -- python/catboost_mlx/core.py catboost/mlx/train_api.cpp` returns empty — confirmed no production-guard diffs. The S30 close decision to keep the guards is respected.

## Must-fix findings

**None.**

## Should-fix nits

1. **`csv_train.cpp:3104–3112` — snapshot replay narrows `Gain` to fp32.**
   After Fix 2, `TBestSplitProperties::Gain` is `double`, and the tree JSON
   serialiser writes it with `%.8g` at line 2693 and `%.10g` at line 2933.
   The snapshot *reader* lambda at line 3104 is declared
   `-> float`, so `prop.Gain = extractPropField("gain")` narrows back to fp32
   even though the struct field is now fp64. Round-tripping a saved snapshot
   therefore still produces `Gain` values with ~1.2e-5 ULP, defeating the
   entire point of the Fix 2 widening on the resume path.
   - *Scope of impact:* snapshot-resumed training runs only. The deserialised
     `Gain` is written to `TRecord::SplitProps` and used solely for
     feature-importance aggregation (lines 2721, 5403) and JSON output
     (lines 2693, 2933) — it does **not** drive split selection on resumed
     iterations (those are re-computed from histograms). Hence not a
     correctness regression, but the stated invariant ("argmax in double
     end-to-end, matching CPU CatBoost") is violated on the resume path.
   - *Suggested fix:* change the lambda return to `double` (e.g.
     `auto extractPropField = [&](const std::string& k) -> double`) and use
     `std::strtod` instead of `std::atof` with explicit null-check, **and**
     bump the JSON `fprintf` precision to `%.17g` at lines 2693 and 2933 so
     the double is losslessly representable in ASCII.

2. **`csv_train.cpp:1545` — non-standard `std::sqrtf`.**
   C++ standard library has `std::sqrt` overloaded for `float`; `std::sqrtf`
   is a C99 library function that libc++ / libstdc++ expose under
   `::sqrtf` but not universally under `std::sqrtf`. The call compiles
   today on Apple-clang, but it is an extension and would break portability
   under a stricter toolchain. This site is inside
   `#ifdef COSINE_RESIDUAL_INSTRUMENT` so it does not affect release builds,
   but the fix is trivial: `std::sqrt(cosDen_f32_shadow)` (the `float`
   overload resolves identically).

3. **`csv_train.cpp:1285–1286` — stale comment after Fix 2.**
   The narrative comment in the one-hot branch still reads "Convert back to
   float only at `ComputeCosineGainKDim` call", which described the K4
   behaviour *before* Fix 2 removed the `static_cast<float>`. The adjacent
   comment on the ordinal branch (line 1428) was correctly updated
   ("Fix 2 removes the cast at finalization"). Apply the same update to
   the one-hot comment block so the two branches are narratively symmetric.

4. **`csv_train.cpp:1286–1287` — memory-cost note scope.**
   The comment states the K4 overhead is "2 doubles per bin candidate
   (16 bytes) vs 2 floats (8 bytes) — negligible at 2540 bins". In practice
   `cosNum_d`/`cosDen_d` are stack scalars re-declared per bin iteration;
   the "2540 bins" framing is fine but the accumulators are single doubles,
   not arrays. Minor clarity issue — the narrative could simply say "two
   additional fp64 stack scalars per bin iteration".

5. **`c1f1ff0e26` + `afdfbc0af8` — DEC-012 atomicity borderline.**
   The T2-Kahan commit (`afdfbc0af8`) bundles two structurally distinct
   changes: (a) Neumaier compensation on float32 cosNum/cosDen (the original
   T2 deliverable), and (b) K4 fp64 widening of the same accumulators
   (applied when Neumaier fell short of the G2 gate). The commit message
   is fully transparent about this ("K4 fp64 fallback invoked
   (pre-authorized per DEC-035)"), and both sub-changes target the same
   four call sites with a single semantic goal — close the cosNum/cosDen
   precision gap. Acceptable under DEC-012 given the pre-authorisation and
   the shared structural change (accumulator precision), but worth flagging
   so future bisection is not surprised by two fp-widening edits landing in
   one commit.

## Observations (no action required)

- **`FindBestSplitPerPartition` stores `results[p].Gain = perturbedGain`
  (lines 1755, 1837)**, while `FindBestSplit` stores
  `bestSplit.Gain = totalGain` (lines 1361, 1571) — i.e. per-partition
  records the perturbed value, oblivious records the clean value. This is
  a pre-existing asymmetry from S26 (`925529d20e`), not an S30 regression,
  but worth a future normalisation pass.
- **`.Score` is write-only** across the file — no consumer reads it. The
  `static_cast<float>(-totalGain)` / `static_cast<float>(-perturbedGain)`
  casts added by Fix 2 are technically correct but store into a field no
  caller inspects. Candidate for future removal, not this sprint.
- **`TLeafCandidate::Gain` widening to `double` (line 3726)** correctly
  propagates through the `std::priority_queue<TLeafCandidate>` comparator
  (`operator<` at line 3729 compares `Gain < o.Gain`) — ordering is now in
  fp64 and matches `TBestSplitProperties::Gain`. No stale fp32 comparator.
- **`perturbedGain` widening preserves DEC-028 semantics.** The noise term
  `noiseScale * noiseDist(*rng)` stays float (both operands float) and is
  explicitly promoted via `static_cast<double>` before the add — the
  RandomStrength × gradRms formula is byte-identical to pre-Fix-2 at the
  addition boundary; only the accumulation target widens. gradRms itself
  remains `float` per DEC-028, which is the documented contract.
- **Instrumentation singleton (`g_cosInstr`) is zero-cost in release
  builds** — the struct, singleton, and all write sites are behind
  `#ifdef COSINE_RESIDUAL_INSTRUMENT`. No dead dispatch, no stranded
  includes on the shipping path. The `<filesystem>` and `<cassert>`
  includes inside the gate (lines 111–112) are duplicates of existing
  conditional includes but are harmless (header guards).

## Style / naming

- `cosNum_d` / `cosDen_d` — lowercase-with-suffix is consistent with the
  existing `invL` / `invR` / `sumLeft` convention in this file. Passes
  project style for local variables.
- `TCosineResidualInstrument`, `TBinRecord`, `TLeafCandidate` follow
  CatBoost's `T`-prefix PascalCase convention.
- `EScoreFunction` and switch dispatch patterns match the S28 precedent.
- Comments consistently tag provenance (`S30-T2-KAHAN K4:`,
  `Fix 2 (S30-COSINE-KAHAN):`) making bisection painless.

## Closing

Ship it. Nit #1 (snapshot truncation) is the most substantive follow-up —
worth scheduling into an S31 housekeeping ticket. Nits #2–#5 are cosmetic
and can be rolled into any future touch of `csv_train.cpp`.
