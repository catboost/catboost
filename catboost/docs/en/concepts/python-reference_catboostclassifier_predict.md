# predict

{% include [sections-with-methods-desc-predict__purpose__full-with-note__div](../_includes/work_src/reusage/predict__purpose__full-with-note__div.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
predict(data,
        prediction_type='{{ prediction-type--Class }}',
        ntree_start={{ fit--ntree_start }},
        ntree_end=0,
        thread_count=-1,
        verbose=None)
```

## {{ dl--parameters }} {#parameters}

### data

#### Description

Feature values data.

The format depends on the number of input objects:

- Multiple — Matrix-like data of shape `(object_count, feature_count)`
- Single — An array

**Possible types**

For multiple objects:

- {{ python-type--pool }}
- {{ python-type__list_of_lists }}
- {{ python-type__np_ndarray }} of shape `(object_count, feature_count)`
- {{ python-type--pandasDataFrame }}
- {{ python_type__pandas-SparseDataFrame }}
- {{ python-type--pandasSeries }}
- [{{ python-type__FeaturesData }}](../concepts/python-features-data__desc.md)
- {% include [libsvm-scipy-except-dia](../_includes/work_src/reusage-formats/scipy-except-dia.md) %}


For a single object:

- {{ python-type--list }} of feature values
- one-dimensional {{ python-type__np_ndarray }} with feature values


**Default value**

{{ python--required }}

### prediction_type

#### Description

The required prediction type.

- Supported prediction types:
- Probability
- Class
- RawFormulaVal
- Exponent
- LogProbability

**Possible types**

{{ python-type--string }}

**Default value**

Class

### ntree_start

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the `eval_period` parameter to _k_ to calculate metrics on every _k_-th iteration.

{% include [eval-start-end-ntree_start__short-param-desc](../_includes/work_src/reusage-common-phrases/ntree_start__short-param-desc.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit--ntree_start }}

### ntree_end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the `eval_period` parameter to _k_ to calculate metrics on every _k_-th iteration.

{% include [eval-start-end-ntree_end__short-param-desc](../_includes/work_src/reusage-common-phrases/ntree_end__short-param-desc.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit--ntree_end }}

### thread_count

#### Description

The number of threads to use.
{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

### verbose

#### Description

Output the measured evaluation metric to stderr.

**Possible types**

{{ python-type--bool }}

**Default value**

None

## {{ dl__return-value }} {#output-format}

{% include [sections-with-methods-desc-python__predict-returned-value](../_includes/work_src/reusage/python__predict-returned-value.md) %}


