# Custom objectives on native Metal

The native Metal trainer accepts a scalar custom objective whose
`calc_ders_range_metal()` method returns a Metal Shading Language function body.
The body is compiled into the training session's GPU kernels. It receives three
`float` arguments named `approx`, `target` and `weight`, and returns:

```cpp
float3(maximized_objective_value, negative_loss_derivative, positive_curvature)
```

All three outputs must already include the observation weight. They must be
finite, and curvature must be nonnegative for every row. Zero-weight rows
contribute zero without invoking the body. The first component is the optimizer's
maximized value; a separate `eval_metric` supplies the reported metric.

This RMSE example follows CatBoost's CUDA custom-objective convention, including
the objective value's full squared-error scale:

```python
from catboost import CatBoostRegressor

class MetalRMSE:
    def calc_ders_range_metal(self):
        return """
            const float residual = target - approx;
            return float3(-weight * residual * residual,
                          weight * residual, weight);
        """

model = CatBoostRegressor(
    task_type="GPU",
    loss_function=MetalRMSE(),
    eval_metric="RMSE",
    iterations=100,
    leaf_estimation_method="Newton",
    boost_from_average=False,
)
model.fit(X, y)
```

The body can use Metal's scalar math functions and embed objective parameters as
numeric constants. It must return a nonempty UTF-8 string of at most 64 KiB,
without NUL characters. Compiler errors are surfaced as CatBoost errors. It is
scalar per-object code; query coupling and multidimensional objectives require
their corresponding native implementations.

The Metal method has its own GPU source contract. Python `calc_ders_range`
callbacks and CUDA `calc_ders_range_gpu` kernels do not supply Metal shader code
and are rejected if the Metal method is absent. Existing CPU and CUDA descriptor
paths keep their own callback interfaces.

The native adapter fingerprints the exact source bytes for snapshot identity.
Changing the body or embedded constants requires a new snapshot. Exported CBM
and JSON models contain the fitted trees and can be evaluated without the custom
objective object. Baselines and `init_model` use the existing native lifecycle;
an initial model starts a new segment, while a snapshot resumes its saved state.

See [training modes acceptance](../TRAINING_MODES_PORT.md) for the supported
partition, boosting, leaf and lifecycle matrix and its validation status.
