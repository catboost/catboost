---
title: Close API and option gaps
status: done
completed: 2026-09-13
priority: 4
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Close API and option gaps

[Board](README.md)

- [x] Integrate shared text/embedding estimators and model finalization.
- [x] Support shared native cross-validation with metrics-only and returned-model modes.
- [x] Support native scalar/query Plain FeatureParallel, including Combination/custom (card 3).
- [x] Expose compound CTR/Plain FeatureParallel and newly enabled explicit symmetric scalar/vector/diagonal-query Simple through the standalone frontend.
- [x] Connect registered scalar/query, six symmetric vector and greedy Simple leaf estimation.
- [x] Support fixed binary splits with source manager identities and policy restrictions.
- [x] Support packed feature subsampling for the three CUDA-registered full-matrix objectives.
- [x] Support automatic simple Borders CTR priors and Full frequency counters.
- [x] Connect feature weights and the native 256-value one-hot boundary.
- [x] Connect source-specific normalization, ridge, Meta-L2 and Langevin consumers.
- [x] Preserve evaluation predictions through copying, standalone conversion and pickle.
- [x] Audit defaults, source no-ops, CPU-only penalties and explicit unsupported combinations.
- [x] Pass coherent native/standalone, alternate, installed, CLI/smoke, host and immutable snapshot gates.

Completed and installed as `20260914T025201Z`: 14,376 tests plus 16 subtests in the full matrix; 6,118 tests alternate; 2,431 tests installed; 348 CLI; 203 preinstall smoke; 203 installed smoke; 350 exact preceding snapshot recoveries.

Counts overlap and are not summed. The report records binary identities,
separate gate counts, source contracts and deliberate corrections to automatic
prior initialization and dynamic feature-weight aliasing. Published defaults
remain compatible; registered explicit options are available. Existing
full-matrix Simple keeps its direct runtime unless another new option selects
native routing, preserving established configurations and snapshots.

CPU-only options are outside this CUDA backlog. Full device RNG/numerical
agreement remains [card 5](05-numerical-agreement.md), scale/performance remains
[card 6](06-scale-and-performance.md), and separate model-based feature analysis
remains [card 7](07-model-based-feature-analysis.md). No full CUDA parity or
NVIDIA execution is claimed.

[API and options release report](../catboost/metal/API_OPTIONS_PORT.md)

[Implementation status](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Native restrictions](../catboost/metal/train_lib/train.cpp)
· [Categorical restrictions](../catboost/metal/train_lib/categorical.cpp)
