# Additional CUDA vector score calculators

The shared Metal vector tree search implements SolarL2, LOOL2, and SatL2 for
MultiClass, MultiClassOneVsAll, MultiRMSE, RMSEWithUncertainty, MultiLogloss, and
MultiCrossEntropy. Their shared score IDs are 4, 5, and 6. The implementation is
in `native/metal_multiclass_scores.h`; the actual-GPU numerical and tree-winner
oracles are in `tests/test_multiclass_scores.py`.

The source calculators are in `cuda/methods/kernel/score_calcers.cuh`. Given a
child's negative-gradient sum G and observation weight W, their contributions
are:

| Calculator | Contribution and explicit float intermediates |
| --- | --- |
| SolarL2 | `−G²(1+2 log(W+1))/W` when `W > float32(1e−20)`, else zero |
| LOOL2 | `a=float32(W/(W−1))` when W>1, else zero; then `a=float32(a*a)` and `−a G²/W` for positive W |
| SatL2 | `a=float32(W(W−2)/(W²−3W+1))` when W>2, else zero; then `−a G²/W` for positive W |

Each CUDA calculator widens its child inputs and evaluates the leaf expression
in double, but its Score field is float and rounds after every contribution.
None uses L2 regularization or random score noise. L2 still regularizes the
separate leaf estimator. Metal uses compensated float pairs for wide gradient
sums and products, division residuals, and the final float accumulation.

SatL2's multiplier is negative on `(2, (3+sqrt(5))/2)` and positive above the pole.
That sign is preserved. Around the pole, the implementation keeps the quadratic
residual with fused products and float pairs; for large W it uses the equivalent
normalized rational expression to avoid intermediate `W*W` overflow. Tests cover
adjacent float32 weights on both sides of the pole, all three strict thresholds,
and representable scores with weights up to 1e30.

The product also normalizes exponents before arithmetic. For example, LOOL2 at
the first float32 weight above one and G=1e−24 has a representable score near
−7.0368754e−35, even though G² alone underflows float32. A corresponding SatL2
case next to the pole yields −2.0379301e−38 from G=1e−22. Both are permanent GPU
regressions; the final accumulator is still rounded to float after each leaf.

CUDA's symmetric greedy dispatch visits each parent, each active output's left
and right child, then MultiClass's missing output. The latter gradient is the
negative sum of the active gradients, retaining wider parent/selected sums.
The Metal implementation preserves that order. The empty pre-split calculator
has zero score, so gain is `float32(raw_score * feature_multiplier)`. It does not
subtract the previous depth's score. CTR multipliers update each depth and the
winning CTR becomes used before testing the raw-score growth condition. SatL2
can therefore select a positive raw score and correctly produce a depth-zero
tree even when its gain has been multiplied to zero.

SolarL2 and LOOL2 are publicly routed to these CUDA vector objectives. SatL2 has
a CUDA calculator and all greedy score dispatch cases, but upstream
`private/libs/options/enum_helpers.cpp::IsSecondOrderScoreFunction` omits it and
throws before greedy search. Exposing the implemented SatL2 calculator in Metal
is an intentional dispatcher correction. NewtonL2 and NewtonCosine remain
rejected: `multiclass_targets.cpp::StochasticDer` does not support second
derivatives as structure weights. These extra calculators require Plain
boosting; CUDA's `IsPlainOnlyModeScoreFunction` rejects Ordered use.

The tests compare GPU results with the checked-in CUDA equations and independent
split-score references. They do not run CPU fitting or establish end-to-end
NVIDIA performance or quality parity.
