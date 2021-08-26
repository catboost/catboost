# staged_predict_proba

{% include [sections-with-methods-desc-staged-predict-proba__purpose__full-with-note__div](../_includes/work_src/reusage/staged-predict-proba__purpose__full-with-note__div.md) %}

## {{ dl--invoke-format }} {#call-format}

```python
staged_predict_proba(data, 
                     ntree_start={{ fit--ntree_start }}, 
                     ntree_end=0,
                     eval_period=1,
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

### eval_period
 
#### Description
 
To reduce the number of trees to use when the model is applied or the metrics are calculated, setthe range of the tree indices to`[ntree_start; ntree_end)` and the `eval_period` parameter to _k_ to calculate metrics on every _k_-th iteration.
 
{% include [eval-start-end-python_r__eval__period__desc__no-example](../_includes/work_src/reusage-common-phrases/python_r__eval__period__desc__no-example.md) %}
 
 
In this case, the metrics are calculated for the following tree ranges: `[0, 2)`, `[0, 4)`, ... , `[0, N)`
 
 
**Possible types** 
 
{{ python-type--int }}
 
**Default value** 
 
{{ fit--staged-predict-eval-period }}
 
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

{% include [sections-with-methods-desc-python__staged_predict__output-type__intro](../_includes/work_src/reusage/python__staged_predict__output-type__intro.md) %}


{% include [sections-with-methods-desc-predict_proba__type-of-returned-value__div](../_includes/work_src/reusage/predict_proba__type-of-returned-value__div.md) %}

