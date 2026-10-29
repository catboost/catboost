---
title: Complete categorical combinations
status: done
priority: 2
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Complete categorical combinations

[Board](README.md)

Completed 2026-09-13 in installed checkpoint `20260913T220233Z`.
Native scalar symmetric Plain/Ordered FeatureParallel training supports compound
CTRs with Sample/Group histories, retained P1/P4 grids, exact snapshots and
standard CBM/JSON exports. Explicit `max_ctr_complexity=2` or `3` selects the
FeatureParallel path by default; existing complexity-one behavior is preserved.

- [x] Connect the dynamic categorical tensor scheduler and feature grids to incremental tree search.
- [x] Support combinations of categorical features and applicable numeric splits.
- [x] Preserve per-permutation histories and retained feature grids as trees grow.
- [x] Connect compound CTR state to snapshots and exact continuation.
- [x] Validate final CTR tables, standard model export and GPU prediction.
- [x] Validate supported CUDA configurations with `max_ctr_complexity > 1`.

Done when supported compound CTR configurations train, resume and export
correctly through the native interface.

Validation: 11,323 tests plus 16 subtests; 3,892 alternate-extension cases;
41 C++ checks; 385 installed tests plus 34 GPU smoke configurations;
187 CLI configurations and 214 exact prior snapshot recoveries. Counts overlap.

[Acceptance report](../catboost/metal/COMPOUND_CTR_PORT.md).
Query/ranking FeatureParallel integration remains with [training modes](03-training-modes.md).

[Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Current categorical restrictions](../catboost/metal/train_lib/categorical.cpp)
· [Upstream CTR settings](https://catboost.ai/docs/en/references/training-parameters/ctr)
