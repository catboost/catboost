### callbacks

#### Description

A list of user-defined callback objects that are invoked at the end of each training iteration.

Each callback must implement the `after_iteration(self, info)` method. The `info` argument has the following attributes:

- `iteration` — the number of completed iterations (the first call receives `1`).
- `metrics` — a dictionary with the values of all metrics computed so far. The keys of the outer dictionary are the dataset names (`learn`, `validation`, or `validation_0`, `validation_1`, ... if several evaluation datasets are set). Each value is a dictionary that maps the metric name to the list of its values, one value per completed iteration (the same structure is returned by the [get_evals_result](../../../concepts/python-reference_catboost_get_evals_result.md) method).

The `after_iteration` method must return a boolean value:

- `True` — continue training.
- `False` — stop training. Callbacks are invoked in the order they are listed, so the callbacks after the one that returned `False` are not invoked on this iteration.

{% note info %}

Metric values are recorded only on the iterations on which they are calculated. Use the default value of the `metric_period` parameter to receive them on every iteration.

{% endnote %}

See the [usage example](../../../concepts/python-usages-examples.md#callbacks).

**Possible types**

{{ python-type--list }} of Python objects that implement the `after_iteration(self, info)` method

**Default value**

None (no callbacks are used)

**Supported processing units**

{{ calcer_type__cpu }}
