---
title: Establish stronger numerical agreement
status: backlog
priority: 5
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Establish stronger numerical agreement

[Board](README.md)

Several saved CUDA validation histories agree around 1e-7 or better.
Remaining maximum history differences are approximately 0.000563444 for
Logloss/L2 and 0.01676455 for MultiClass/L2.

- [ ] Resolve general tied/empty-child split semantics behind Logloss/L2 divergence.
- [ ] Resolve equivalent CTR alias choices and their later feature-penalty effects in MultiClass/L2.
- [ ] Complete CUDA GPU random-buffer state and consumption semantics.
- [ ] Compare training behavior across multiple seeds and representative datasets.
- [ ] Extend saved-reference checks with configuration-matched live NVIDIA comparisons when hardware is available.
- [ ] Define and document justified numerical and model-quality acceptance tolerances.

Done when remaining divergences are resolved or explained within independently
justified acceptance criteria. Do not force fixture-specific split choices or
widen tolerances to conceal defects. CUDA GPU training is itself
nondeterministic, so universal bitwise equality is not the acceptance rule.

[Current comparison evidence](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Upstream GPU behavior](https://catboost.ai/docs/en/features/training-on-gpu)
