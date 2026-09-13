---
title: Finish remaining training modes
status: backlog
priority: 3
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Finish remaining training modes

[Board](README.md)

- [ ] Port CUDA-registered greedy YetiRank for Depthwise, Lossguide and Region.
- [ ] Port CUDA-supported Ordered query objectives.
- [ ] Connect CUDA-supported query/ranking FeatureParallel training, including compound CTRs through the categorical machinery completed in card 2.
- [ ] Integrate Combination losses.
- [ ] Integrate custom objectives through native Metal training.
- [ ] Verify supported combinations of objectives, tree policies, scores, samplers and leaf estimators against CUDA registrations.
- [ ] Validate metrics, baselines, initial models, snapshots, callbacks and model export for newly connected modes.

Greedy PairLogit is tracked separately in [card 1](01-greedy-pairlogit.md).
Done when each newly supported mode passes native and applicable standalone
acceptance without CPU-training fallback.

[Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Native training restrictions](../catboost/metal/train_lib/train.cpp)
