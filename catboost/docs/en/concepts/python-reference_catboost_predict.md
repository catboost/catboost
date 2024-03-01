# predict

{% include [sections-with-methods-desc-predict__purpose__full-with-note__div](../_includes/work_src/reusage/predict__purpose__full-with-note__div.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
predict(data,
    prediction_type=None,
    ntree_start={{ fit--ntree_start }},
    ntree_end=0,
    thread_count={{ fit__thread_count__wrappers }},
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

- {% include [libsvm-scipy-except-dia](../_includes/work_src/reusage-formats/scipy-except-dia.md) %}
- {{ python-type--pool }}
- {{ python-type__list_of_lists }}
- {{ python-type__np_ndarray }} of shape `(object_count, feature_count)`
- {{ python-type--pandasDataFrame }}
- {{ python_type__pandas-SparseDataFrame }}
- {{ python-type--pandasSeries }}
- [{{ python-type__FeaturesData }}](../concepts/python-features-data__desc.md)


For a single object:

- {{ python-type--list }} of feature values
- one-dimensional {{ python-type__np_ndarray }} with feature values

**Default value**

Required parameter

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

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[ntree_start; ntree_end)`.

This parameter defines the index of the first tree to be used when applying the model or calculating the metrics (the inclusive left border of the range). Indices are zero-based.

**Possible types**

int

**Default value**

0


### ntree_end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[ntree_start; ntree_end)`.

This parameter defines the index of the first tree to be used when applying the model or calculating the metrics (the inclusive left border of the range). Indices are zero-based.

**Possible types**

int

**Default value**

0 (the index of the last tree to use equals to the number of trees in the model minus one)

### thread_count

#### Description

The number of threads to calculate prediction.

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

{% include [sections-with-methods-desc-python__predict-returned-value](../_includes/work_src/reusage/python__predict-returned-value.md) %}
