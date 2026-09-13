# Greedy PairLogit work in progress

Source checkpoint: 2026-09-13. This extension is included for continued
development on `metal-m3`; it has not passed release acceptance.

## Connected code

- Depthwise, Lossguide, and Region training use the original supplied pair
  edges for structure derivatives, leaf estimation, backtracking, and loss.
- Original incident pair mass supplies document weights. Extra object/group
  weights do not multiply the supplied edge weights.
- Each permutation dataset retains its own predictions and leaf estimates.
  Final leaf values use CUDA's unweighted zero-average convention.
- A counted pair-session C ABI, the native GPU trainer, and the standalone
  greedy ranker/lifecycle are connected. Snapshot identity includes the
  supplied training and validation pairs.

The source mapping follows CUDA's querywise non-symmetric/region trainer
registrations, `InitPairLogit`, and `NeedZeroAverage` leaf finalization.

## Current evidence

Both native extension variants and the CLI compiled successfully on M3 Pro.
An initial private GPU run of `tests/test_greedy_pairwise.py` produced
**265 passed, 4 failed**. The passing cases include independent edge equations
for root scores, leaf estimates, incident weights, loss, and centering, plus
invalid-input/count checks. The failing cases are:

- Bayesian, seven datasets, Region.
- Bernoulli, four datasets, Depthwise.
- Bernoulli, seven datasets, Depthwise.
- Bernoulli, seven datasets, Region.

These fail independent permutation-cursor comparisons with four Armijo leaf
attempts. A follow-up diagnostic that starts each oracle step from the actual
GPU cursor still observes a maximum difference of approximately 1.386e-4;
the same diagnostic with three leaf attempts differs by at most 1.193e-7.
This narrows the investigation but does not establish the cause. Test
tolerances have not been changed to hide the failures.

To reproduce the initial private suite on Apple Silicon with the standalone
dependencies installed:

```sh
PYTHONPATH=catboost/metal/python python -m pytest -q catboost/metal/tests/test_greedy_pairwise.py
```

The preceding installed local checkpoint, `20260913T155047Z`, passed 10,560
tests plus 16 subtests, 3,507 alternate-extension tests, 685 installed GPU
paths, 153 CLI variants, and 194 preceding-snapshot recoveries. Those counts
overlap and describe the build before this PairLogit extension.

## Remaining acceptance

- Diagnose the four numerical mismatches and verify exact continuation for
  the affected datasets.
- Complete native and standalone public acceptance, including categorical
  permutations, literal pair weights, unlabeled Pools, model readers,
  validation metrics, and snapshots.
- Update obsolete unsupported-mode expectations as support is validated.
- Run the coherent regression/alternate-extension/CLI/recovery matrix, then
  package and test a matching wheel before replacing the installed checkpoint.

No CPU CatBoost training or live NVIDIA comparison was used for this work.
