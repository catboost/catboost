# Sprint 24 D0 — Off-by-One Diagnostic: Cascade Retest

**Branch**: `mlx/sprint-24-dec023-fix`
**Date**: 2026-04-21
**Status**: KILL-SWITCH FIRED — off-by-one not present in code; Part B/C/D measurements not taken

---

## §1 Task

Verify whether a claimed off-by-one between the scoring kernel (suffix-sum semantics) and
`ApplySplitToPartitions` (split application) is present in `bench_boosting.cpp`, and if so,
fix it and re-measure config #8 bimodality at current T2 production.

The hypothesis: config #8's 105 ULP bimodal cascade might be an artifact of suboptimal splits
being applied one bin off from the scored optimum, rather than (or in addition to) the DEC-023
atomic-float race.

---

## §2 Part A — Code Audit

### §2.1 Scoring path (suffix-sum → GAIN computation)

**Suffix-sum kernel** (`kSuffixSumSource`, `kernel_sources.h`):

The kernel initializes `scanBuf[t] = histogram[firstFold + (folds-1-t)]` (reversed order) and
runs a Hillis-Steele inclusive prefix scan. Thread t >= 1 writes the result to bin `folds-1-t`
in the output.

After the transform, for bin index `b` (0 ≤ b ≤ folds-2):

```
transformedHist[firstFold + b] = Σ histogram[firstFold + b .. firstFold + folds-1]
```

The histogram convention (`kHistOneByteSource` writeback, confirmed in `kT2AccumSource` comment):

```
histogram[firstFold + k] = sum of stats for docs with RAW BIN = k + 1
```

Therefore:

```
transformedHist[firstFold + b] = sum for raw bins {b+1, b+2, ..., folds}
                                = sum for docs with raw_bin > b
```

**Score kernel** (`kScoreSplitsLookupSource`):

```metal
float sumRight = histogram[histBase + firstFold + binInFeature];  // uses transformedHist
```

For split at `binInFeature = b`, `sumRight` = sum for docs with `raw_bin > b`. The kernel
scores the partition: right = {docs with raw_bin > b}, left = {remaining docs}.

**Returned `BinId`** = `binInFeature = b` (the histogram slot index, 0-indexed).

### §2.2 Application path (`ApplySplitToPartitions`)

```cpp
uint32_t featureVal = (dataPtr[d * numUi32PerDoc + col] >> shift) & mask;
// Right child if value > threshold (CatBoost ordinal convention)
bool goRight = (featureVal > binThreshold);   // binThreshold = best.BinId = b
```

`featureVal` is the raw bin value extracted directly from the compressed data (0-indexed, same
unit as `raw_bin` above). The split criterion is `raw_bin > b`.

### §2.3 Consistency check

| Path | Criterion (expressed in raw bin) | Right partition |
|------|----------------------------------|----------------|
| Scoring (suffix-sum at slot b) | raw_bin > b | {raw_bin = b+1, b+2, ..., folds} |
| Apply (`featureVal > binThreshold = b`) | raw_bin > b | {raw_bin = b+1, b+2, ..., folds} |

**The two paths are consistent.** Both encode the same partition boundary: `raw_bin > b` for the
right child, where `b = BinId` (the histogram slot index returned by the scoring kernel).

### §2.4 CatBoost CPU reference confirmation

`catboost/libs/model/split.h` line 12-13:

```cpp
template <typename TBucketType>
inline bool IsTrueHistogram(TBucketType bucket, TBucketType splitIdx) {
    return bucket > splitIdx;
}
```

CatBoost's authoritative split criterion is `bucket > splitIdx` — strict greater than. This
matches both the bench_boosting scoring and apply paths. The comment on line 829 of
`bench_boosting.cpp` ("CatBoost ordinal convention") is correct.

### §2.5 Kill-switch condition

> **Off-by-one isn't actually present in the code** (Path X agent may have misread the source)
> → halt, report. Describe what you actually see in the scoring/apply paths, and we'll reassess.

**Kill-switch fires.** The described off-by-one is NOT present. The scoring and applying paths
use the same strict-greater-than convention, consistent with CatBoost's canonical
`IsTrueHistogram` definition.

---

## §3 Part B — Not Measured

No build or runs were performed. The kill-switch fired at Part A (code audit). Proceeding to
measure would require a code change that does not exist (there is nothing to fix) and would
change the semantics to `bin >= b` (inclusive), which would make `ApplySplitToPartitions`
INCONSISTENT with the scoring kernel — introducing a real bug.

---

## §4 Part C — Distribution: Not Measured

No distribution data collected. Existing S23 D0 data (from `d0_bimodality_verification.md §A`)
remains authoritative: 20 runs at config #8 show 10× 0.48231599 / 10× 0.48231912, 105 ULP
spread.

---

## §5 Part D — Verdict

**Case 3 (pre-diagnosed): the proposed off-by-one was not present.**

The Path X diagnostic agent appears to have used inconsistent indexing terminology when
describing the split semantics:

- "bin ≥ b goes right" (scoring) — this refers to **histogram slot b** in the suffix-sum output,
  which equals sum for raw bins b+1..folds (i.e., raw_bin > b).
- "bin > b goes right" (applying) — this refers to **raw bin value b** in the data, which is
  the same condition: raw_bin > b.

The two statements describe the same partition boundary, expressed in different coordinate
systems (histogram slot index vs raw bin value). There is no functional mismatch.

---

## §6 Why the bimodality is not an artifact of off-by-one

If the splits were applied one bin off, the per-partition counts would diverge from the scoring
target by roughly `docs_in_bin_b` docs per partition per split. At config #8 (10k docs, 128 bins,
depth=6), that is approximately 10000/128/64 ≈ 1.2 docs per partition per split — a substantial
divergence that would affect ALL 18 configs systematically, not just config #8.

The bimodality is config-specific (exactly 1 of 18 configs fires), which is inconsistent with a
systematic off-by-one in the apply path. The DEC-023 root cause (features 1-3 atomic-float
non-associativity in T2-accum) remains the most credible single-config-specific explanation, as
documented in `d0_bimodality_verification.md`.

---

## §7 Recommendation

1. **Do not change `ApplySplitToPartitions`.** The `featureVal > binThreshold` (strict greater
   than) is correct per CatBoost's canonical convention. Changing it to `>=` would introduce a
   real off-by-one inconsistency with the scoring kernel.

2. **Proceed with the original DEC-023 fix plan**: Option A (Path 4 — pure int-atomic scatter
   with workaround for MLX 64-bit atomic limitation) or re-examine Option 1 from the S24 README
   (TG-local reduce + single-thread commit for features 1-3). The bimodality root cause remains
   the atomic-float race in `kT2AccumSource` features 1-3.

3. **Discard the Path X off-by-one hypothesis.** It does not hold up to the code audit.

---

## §8 Files Read (Audit Evidence)

- `catboost/mlx/tests/bench_boosting.cpp` — `ApplySplitToPartitions` (lines 803-835),
  `FindBestSplitGPU` scoring path (lines 514-659)
- `catboost/mlx/kernels/kernel_sources.h` — `kSuffixSumSource` (lines 345-392),
  `kScoreSplitsLookupSource` (lines 508-590), `kHistOneByteSource` (lines 107-281),
  `kT2AccumSource` bin convention comment (lines 1103-1105, 1160-1162)
- `catboost/libs/model/split.h` — `IsTrueHistogram` (lines 11-14)
- `docs/sprint23/d0_bimodality_verification.md` — footprint, mechanism, DEC-023 scope
- `docs/sprint24/d0_dec023_fix.md` — Path 5 falsification history
- `.claude/state/KNOWN_BUGS.md` — BUG-T2-001 DEC-023 status

**No commits made.** Standing order DEC-012 one-change-per-commit was not triggered since no
valid fix was identified.
