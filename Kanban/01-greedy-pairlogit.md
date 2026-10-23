---
title: Finish greedy PairLogit
status: done
completed: 2026-09-13
priority: 1
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Finish greedy PairLogit

[Board](README.md)

Completed and installed as checkpoint `20260913T195124Z`. The four original
Armijo failures are resolved without widening tolerances. A separate native
PairAccuracy check now supports unlabeled supplied-pair Pools.

Validation: **11,099 tests + 16 subtests**, **3,744 alternate-extension tests**,
**237 installed tests + 18 GPU smoke cases**, **171 CLI configurations**, and
**202 exact prior-snapshot recoveries**. Counts overlap and are not summed.

[Detailed report](../catboost/metal/GREEDY_PAIRLOGIT_PORT.md)
· [Release manifest](../catboost/metal/.build/releases/20260913T195124Z/manifest.json)

- [x] Diagnose and resolve Bayesian P7 Region cursor mismatch.
- [x] Diagnose and resolve Bernoulli P4/P7 Depthwise cursor mismatches.
- [x] Diagnose and resolve Bernoulli P7 Region cursor mismatch.
- [x] Verify exact snapshot continuation for affected datasets.
- [x] Complete native and standalone public acceptance: categorical permutations, pair weights, unlabeled Pools, model readers, validation metrics and snapshots.
- [x] Update unsupported-mode expectations when support is validated.
- [x] Pass coherent full regression, alternate-extension, CLI and snapshot-recovery checks.
- [x] Package, install and validate a wheel matching the accepted sources.

The numerical failures are resolved and the matching installed package passes
the complete acceptance matrix.

[Source evidence](../catboost/metal/GREEDY_PAIRLOGIT_WIP.md)
