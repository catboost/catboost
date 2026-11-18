# Multiclass sampling and score randomness

The multiclass CUDA trainer uses the greedy subset searcher. Its sampling and
score statistics differ from the scalar oblivious-tree path. The new Metal
helpers in [`metal_multiclass_bootstrap.h`](native/metal_multiclass_bootstrap.h)
passed 44 actual GPU tests on September 12, 2026. A further 18 dedicated
session tests passed after sampling/noise integration. These focused results
are separate from the full multiclass/native acceptance matrix.

## CUDA behavior

[`train_lib/multiclass.cpp`](../cuda/train_lib/multiclass.cpp) registers the
greedy trainer. Its
[`weak_objective_impl.h`](../cuda/methods/greedy_subsets_searcher/weak_objective_impl.h)
generates one bootstrap weight per object, then
[`multiclass_targets.cpp`](../cuda/targets/multiclass_targets.cpp) forms sampled
effective weights and class derivatives. Bayesian, Bernoulli, Poisson, and No
therefore share one object draw across every class. Original effective weights
remain in use for final leaf estimation.

Effective weights already include object weight, group weight, and target
class weight, from
[`MakeClassificationWeights`](../private/libs/target/data_providers.cpp).
The multiclass derivative is `w*(onehot-probability)`: there is no extra class
weight, class-count divisor, or learning-rate factor to apply. MultiClass
stores `C-1` derivatives; MultiClassOneVsAll stores all `C` derivatives.

**CUDA explicitly rejects multiclass MVS** in
[`catboost_options.cpp`](../private/libs/options/catboost_options.cpp), with
matching assertions in the greedy weak objective and searcher. There is no
CUDA multiclass vector-MVS magnitude or automatic-lambda formula to translate.
Adding one would be an extension, requiring its own design and validation.

The actual noise statistic is
[`ComputeTargetVarianceImpl`](../cuda/methods/greedy_subsets_searcher/kernel/compute_scores.cu),
called by
[`ComputeTargetStdDev`](../cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp)
**after bootstrap**. For sampled gradients `g` and sampled weights `w`:

```text
eligible rows: w > 1e-15
MultiClass: reconstruct g_last = -sum(stored class gradients)
numerator = sum_over_eligible_rows(sum_over_all_classes(g*g) / w)
denominator = sum_over_eligible_rows(w)
stddev = sqrt(numerator / denominator), or zero for an empty sampled target
```

There is no mean subtraction, no division by row count or class count, and no
scalar `ZeroAwareDivide` tiny-gradient rule. OneVsAll does not reconstruct a
missing class. `random_score_helper.h` belongs to the scalar path and must not
be reused for this reduction.

The per-tree scale is
`random_strength * logistic(log(original_rows)-absolute_iteration*learning_rate) * stddev`.
The boosting model-length multiplier is passed through
[`greedy_subsets_searcher.h`](../cuda/methods/greedy_subsets_searcher.h). The
original row count controls this multiplier even when bootstrap excludes rows.

## Symmetric-tree score selection

CUDA symmetric greedy search selects a **gain**. In
[`ComputeOptimalSplits`](../cuda/methods/greedy_subsets_searcher/kernel/compute_scores.cu),
it copies an empty score calculator immediately after selecting the feature,
accumulates candidate leaves into the first calculator, then subtracts the
empty calculator's score. Both Cosine calculators generate the same feature
noise from the same seed:

```text
candidate_score = raw_cosine_score + feature_noise
baseline_score = feature_noise
selection_gain = candidate_score - baseline_score
```

The noise therefore cancels apart from floating-point rounding. CUDA compares
`Gain`, while retaining the noisy `Score` in split properties. A Metal winner
kernel that compares `raw_cosine_score + feature_noise` directly would implement
different behavior. L2 ignores the noise entirely. All thresholds of a feature
share the same draw.

After selecting the best gain, CUDA permits growth only when that candidate's
noisy raw `Score` is negative. Thus noise can prevent a split even when its
contribution cancels from gain, and the search does not fall back to a worse
gain with a negative score. Selection and stopping must preserve both values.

The existing Metal MWC/Box-Muller helper supplies those draws with documented
per-feature/per-depth seed adaptation. CUDA launch-owned seeds are not
bit-identical to Metal's resumable absolute-iteration streams. The subtraction
above must remain two rounded float operations when matching the CUDA gain.

## GPU helper contract

Concatenate `CBMMetalMulticlassBootstrapSource` after
`CBMMetalBootstrapSource`. Its independent 16-byte structure is:

```cpp
uint32_t rows, dimensions, multi_logit, reserved;
```

`dimensions=C-1, multi_logit=1` for MultiClass; `dimensions=C, multi_logit=0`
for OneVsAll. Gradients are class-major `[dimensions, rows]`.

| Kernel | Bindings | Dispatch |
| --- | --- | --- |
| `ApplyMulticlassBootstrap` | 0 in/out gradients; 1 original weights; 2 object multipliers; 3 output sampled structure weights; 4 parameters | `rows*dimensions` threads |
| `ReduceMulticlassScoreStatistics` | 0 sampled gradients; 1 sampled structure weights; 2 output `float2[groups]`; 3 parameters | Any positive count of full 256-thread groups |

The reduction outputs partial numerator divided by original row count in `.x`
and partial accepted weight divided by original row count in `.y`. This common
scaling limits float32 overflow and leaves their ratio unchanged. Sum both
columns in host double, then compute the ratio and square root. The eligible
weight threshold applies to the **sampled**, unscaled weight. Reject nonfinite
final statistics through the normal runtime error path.

Reuse `GenerateScoreFeatureNoise` with `BootstrapParams.rows=features`, the
absolute tree iteration, `stream=depth+1`, and the fixed tree scale. That helper
is supplied by [`metal_score_noise_kernels.h`](native/metal_score_noise_kernels.h).

The Metal helper multiplies the already-weighted original gradients by each
object draw. CUDA computes derivatives using already-sampled effective weights;
the algebra is the same, with possible float32 multiplication-order differences.
Original class probabilities do not change during object sampling.

## Validation

[`test_multiclass_bootstrap_kernels.py`](tests/test_multiclass_bootstrap_kernels.py)
compiles [`multiclass_bootstrap_probe.mm`](tests/multiclass_bootstrap_probe.mm)
and tests actual Metal execution against independent equations. Coverage
includes real GPU Bayesian/Bernoulli/Poisson draws, 1–63 stored dimensions,
missing-class reconstruction, zero and threshold weights, tiny gradients,
all-excluded targets, original-weight preservation, uniform sample-scale
invariance, alternative missing-class choices, 65,539-row reductions with
different launch counts, and finite weights around `1e30`. No CPU model is fit.

[`test_multiclass_bootstrap_training.py`](tests/test_multiclass_bootstrap_training.py)
adds 18 connected-session cases. Moderate noise preserves a separated winning
gain; huge finite noise exercises the two rounded score operations. With
nonzero initial cursors, deliberately chosen fixtures select different
candidates if the runtime uses pre-bootstrap variance or omits the missing
class's energy; those fixtures keep both competing raw scores negative.
Separate huge-noise cases verify selecting by gain and then stopping when the
selected noisy raw score is nonnegative. The correct CUDA equations match all
actual session outcomes.
The file also verifies the MVS rejection. The multiclass runtime owner's
separate suite owns original-weight leaves and seeded continuation checks.
