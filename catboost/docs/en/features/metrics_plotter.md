# Metrics Plotter for Jupyter

CatBoost library has a tool for plotting arbitrary learning curves. This tool triggers interactive widget in Jupyter Notebook / JupyterLab interface.

[Additional packages for data visualization support](../installation/python-installation-additional-data-visualization-packages.md) must be installed to plot charts in [Jupyter Notebook](http://jupyter.org/).

## Features

- Clickable plots with learning curves
- Remaining learning time estimation based on total iteration count
- Graceful learning stop by `KeyboardInterruption` without error message in the notebook 
- Tracking of iteration number with best metric value

## Usage

```python
from catboost import MetricsPlotter

train_metrics = ["MSE", "MAE", "R2"]
test_metrics = ["MSE", "R2"]
iter_count = 10

with MetricPlotter(train_metrics, test_metrics, iter_count) as plotter:
    for i in range(iter_count):

        # TRAIN CODE: <...>

        plotter.log(
            i, train=True,
            metrics={
                "MAE": value1,
                "MSE": value2,
                "R2": value3,
            }
        )

        # EVALUATION CODE: <...>

        plotter.log(
            i, train=False,
            metrics={
                "MSE": value1,
                "R2": value2,
            }
        )
```

Note that metrics should be passed to class successively, in a natural way described above.

To save plot in the notebook, use `Widget` → `Save Widget State` menu option in Jupyter Notebook interface.

`MetricsPlotter` constructor arguments:

Aguments | Type | Description
----- |  ----- | -----
`train_metrics` | `list` of `str` or `list` of `dict` | List of train metrics to be tracked. </br> Each item in list can be either string with metric name or dict with the fields `name` and `best_value`, where the latter can be one of the following: `Max`, `Min`, `Undefined`.
`test_metrics` | `list` of `str` or `list` of `dict`, optional (default=`None`) | List of test metrics to be tracked. </br> Has the same format as `train_metrics`. Equals to `train_metrics`, if not defined
`total_iterations` | `int`, optional (default=`None`) | Total number of iterations, allows for remaining time estimation.