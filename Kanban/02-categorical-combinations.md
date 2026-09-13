---
title: Complete categorical combinations
status: backlog
priority: 2
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Complete categorical combinations

[Board](README.md)

Simple CTRs work. Training currently restricts `max_ctr_complexity` to 1;
compound and tree-dependent CTR scheduling still needs integration.

- [ ] Connect the dynamic categorical tensor scheduler and feature grids to incremental tree search.
- [ ] Support combinations of categorical features and applicable numeric splits.
- [ ] Preserve per-permutation histories and retained feature grids as trees grow.
- [ ] Connect compound CTR state to snapshots and exact continuation.
- [ ] Validate final CTR tables, standard model export and GPU prediction.
- [ ] Validate supported CUDA configurations with `max_ctr_complexity > 1`.

Done when supported compound CTR configurations train, resume and export
correctly through the native interface.

[Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Current categorical restrictions](../catboost/metal/train_lib/categorical.cpp)
· [Upstream CTR settings](https://catboost.ai/docs/en/references/training-parameters/ctr)
