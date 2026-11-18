---
title: Establish stronger numerical agreement
status: deferred
priority: 5
created: 2026-09-13
tags: [catboost, metal, parity]
---

# Establish stronger numerical agreement

[Board](README.md)

Paused on 2026-09-13 at the user's request. Unfinished source changes and a
resume note are preserved locally on `codex/parked-card-5-numerical-agreement`
at commit `701ef858a7ebd5e11b2388ec54b86544eb47f7a7`.
The resume note is `catboost/metal/NUMERICAL_AGREEMENT_WIP.md` on that branch.
This work has not passed coherent release acceptance and was neither installed
nor pushed. At deferral, `metal-m3` and the installed package retained accepted
card 4. Card 7 subsequently completed on `metal-m3` as installed checkpoint
`20260914T052913Z`, without importing the parked card 5 work.

Several saved CUDA validation histories agree around 1e-7 or better.
Remaining maximum history differences are approximately 0.000563444 for
Logloss/L2 and 0.01676455 for MultiClass/L2.

- [ ] Resolve general tied/empty-child split semantics behind Logloss/L2 divergence.
- [ ] Resolve equivalent CTR alias choices and their later feature-penalty effects in MultiClass/L2.
- [ ] Reconcile existing greedy QueryRMSE/QuerySoftMax/PairLogit Gradient/Newton weak-score denominator dispatch with CUDA; new Simple leaves follow the source mapping.
- [ ] Complete CUDA GPU random-buffer state and consumption semantics.
- [ ] Carry shared random state across model-based feature experiments while keeping each experiment's model age local; the current workflow reuses Metal's existing per-fit random-state helpers.
- [ ] Reconcile vector continuous cursor updates (`base + step * raw`) with replay of rounded stored leaf increments; a one-ULP prefix difference can change later tied CTR choices. Card 7 preserves saved-model replay semantics and does not change the accepted training arithmetic.
- [ ] Compare training behavior across multiple seeds and representative datasets.
- [ ] Extend saved-reference checks with configuration-matched live NVIDIA comparisons when hardware is available.
- [ ] Define and document justified numerical and model-quality acceptance tolerances.

Done when remaining divergences are resolved or explained within independently
justified acceptance criteria. Do not force fixture-specific split choices or
widen tolerances to conceal defects. CUDA GPU training is itself
nondeterministic, so universal bitwise equality is not the acceptance rule.

[Current comparison evidence](../catboost/metal/IMPLEMENTATION_STATUS.md)
· [Upstream GPU behavior](https://catboost.ai/docs/en/features/training-on-gpu)
