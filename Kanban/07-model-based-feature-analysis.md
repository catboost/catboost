---
title: Support model-based feature analysis
status: backlog
priority: 7
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Support model-based feature analysis

[Board](README.md)

Discovered during the card 4 training-options audit. This is the separate
feature-ablation analysis workflow exposed by `model-based-eval`.

- [ ] Port the CUDA trainer's `ModelBasedEval` workflow and feature-set controls.
- [ ] Reuse the registered Metal training paths for baseline and feature-set runs.
- [ ] Validate fold histories, baselines, output artifacts and supported CLI options.
- [ ] Test explicit rejection of unsupported configurations and interrupted runs.

The native Metal trainer currently rejects this operation. Ordinary training,
prediction, metrics and cross-validation do not establish support for it.

[Metal entry point](../catboost/metal/train_lib/train.cpp)
· [CUDA implementation](../catboost/cuda/train_lib/train.cpp)
