# staged_predict

{% include [sections-with-methods-desc-staged-predict__purpose__full-with-note__div](../_includes/work_src/reusage/staged-predict__purpose__full-with-note__div.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
staged_predict(data,
    prediction_type=None,
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

### prediction_type

#### Description

The required prediction type.

Supported prediction types:
- Probability
- Class
- RawFormulaVal
- Exponent
- LogProbability

**Possible types**

string

**Default value**

None (Exponent for Poisson and Tweedie, RawFormulaVal for all other loss functions)

### ntree_start

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[ntree_start; ntree_end)` and the step of the trees to use to eval_period.

This parameter defines the index of the first tree to be used when applying the model or calculating the metrics (the inclusive left border of the range). Indices are zero-based.

**Possible types**

int

**Default value**

0

### ntree_end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[ntree_start; ntree_end)` and the step of the trees to use to eval_period.

This parameter defines the index of the first tree to be used when applying the model or calculating the metrics (the inclusive left border of the range). Indices are zero-based.

**Possible types**

int

**Default value**

0 (the index of the last tree to use equals to the number of trees in the model minus one)

### eval_period

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[ntree_start; ntree_end)` and the step of the trees to use to `eval_period`.

This parameter defines the step to iterate over the range `[ntree_start; ntree_end)`. For example, let's assume that the following parameter values are set:

- `ntree_start` is set 0
- `ntree_end` is set to N (the total tree count)
- `eval_period` is set to 2

In this case, the results are returned for the following tree ranges: `[0, 2), [0, 4), ... , [0, N)`.

**Possible types**

int
**Default value**

1 (the trees are applied sequentially: the first tree, then the first two trees, etc.)


### thread_count

#### Description

The number of threads to calculate predictions.

Optimizes the speed of execution. This parameter doesn't affect results.

**Possible types**

int

**Default value**

-1 (the number of threads is equal to the number of processor cores)

### verbose

#### Description

Output the measured evaluation metric to stderr.

**Possible types**

bool

**Default value**

None


## {{ dl__return-value }} {#output-format}

{% include [sections-with-methods-desc-python__staged_predict__output-type](../_includes/work_src/reusage/python__staged_predict__output-type.md) %}


