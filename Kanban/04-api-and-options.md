---
title: Close API and option gaps
status: backlog
priority: 4
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Close API and option gaps

[Board](README.md)

- [ ] Integrate shared text and embedding estimated-feature pipelines and model finalization.
- [ ] Support native cross-validation.
- [ ] Support Plain boosting with FeatureParallel partitioning.
- [ ] Support fixed binary splits.
- [ ] Support feature subsampling where CUDA supports it.
- [ ] Support automatic CTR prior estimation.
- [ ] Audit remaining training options, defaults and feature penalties against CUDA-supported behavior.
- [ ] Add acceptance coverage for newly supported options and explicit rejection of unsupported combinations.

Done when the audited CUDA API/options surface is implemented and validated.
Check each option against upstream GPU restrictions; CPU-only capabilities
are outside this CUDA parity backlog.

[Native restrictions](../catboost/metal/train_lib/train.cpp)
· [Categorical restrictions](../catboost/metal/train_lib/categorical.cpp)
