### data

#### Description

A file or matrix with the input dataset.

**Possible values**

{{ python-type--pool }}

**Default value**

{{ python--required }}

### metrics

#### Description

The list of metrics to be calculated.
[Supported metrics](../../../references/custom-metric__supported-metrics.md)
For example, if the {{ error-function--AUC }} and {{ error-function--Logit }} metrics should be calculated, use the following construction:

```python
['Logloss', 'AUC']
```

**Possible values**

{{ python-type--list-of-strings }}

**Default value**

{{ python--required }}

### ntree_start

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to`[ntree_start; ntree_end)`.

{% include [eval-start-end-ntree_start__short-param-desc](../reusage-common-phrases/ntree_start__short-param-desc.md) %}

**Possible values**

{{ python-type--int }}

**Default value**

{{ fit--ntree_start }}


### ntree_end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to`[ntree_start; ntree_end)` and the step of the trees to use to`eval_period`.

{% include [eval-start-end-ntree_end__short-param-desc](../reusage-common-phrases/ntree_end__short-param-desc.md) %}

**Possible values**

{{ python-type--int }}

**Default value**

{{ fit--ntree_end }}

### eval_period

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the step of the trees to use to`eval_period`.


{% include [eval-start-end-python_r__eval__period__desc](../reusage-common-phrases/python_r__eval__period__desc.md) %}

**Possible values**

{{ python-type--int }}

**Default value**

{{ fit--staged-predict-eval-period }}


### thread_count

#### Description

{% include [reusage-thread-count-short-desc](thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible values**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

{% include [python__log-params](../../../_includes/work_src/reusage-python/python__log-params.md) %}
