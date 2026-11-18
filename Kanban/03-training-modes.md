---
title: Finish remaining training modes
status: done
priority: 3
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Finish remaining training modes

[Board](README.md)

- [x] Port CUDA-registered greedy YetiRank for Depthwise, Lossguide and Region.
- [x] Port CUDA-supported Ordered query objectives.
- [x] Connect CUDA-supported query/ranking FeatureParallel training, including compound CTRs through the categorical machinery completed in card 2.
- [x] Integrate Combination losses.
- [x] Integrate custom objectives through native Metal training.
- [x] Verify supported combinations of objectives, tree policies, scores, samplers and leaf estimators against CUDA registrations.
- [x] Validate metrics, baselines, initial models, snapshots, callbacks and model export for newly connected modes.

Greedy PairLogit is tracked separately in [card 1](01-greedy-pairlogit.md).
Done when each newly supported mode passes native and applicable standalone
acceptance without CPU-training fallback.

Completed and installed as `20260914T000929Z`: 12,745 tests +16 subtests,
4,901 alternate, 1,267 installed, 134 GPU smoke configurations, 287 CLI
configurations and 244 exact preceding snapshot recoveries. Counts overlap.
The registration audit keeps greedy Simple leaf estimation and standalone
Plain FeatureParallel/compound frontends in [card 4](04-api-and-options.md).

[Training modes release report](../catboost/metal/TRAINING_MODES_PORT.md)

[Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Native training restrictions](../catboost/metal/train_lib/train.cpp)
