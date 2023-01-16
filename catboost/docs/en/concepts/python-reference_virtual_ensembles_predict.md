# virtual_ensembles_predict

{% include [sections-with-methods-desc-predict__purpose__full-with-note__div](../_includes/work_src/reusage/predict__purpose__full-with-note__div.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
virtual_ensembles_predict(data,
    prediction_type='VirtEnsembles',
    ntree_end=0,
    virtual_ensembles_count=10,
    thread_count=-1 (the number of threads is equal to the number of processor cores),
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

Required prediction type. Supported prediction types: VirtEnsembles, TotalUncertainty

**Possible types**

string

**Default value**

VirtEnsembles


### ntree_end

#### Description

To reduce the number of trees to use when the model is applied or the metrics are calculated, set the range of the tree indices to `[ntree_start; ntree_end)` and the step of the trees to use to eval_period.

This parameter defines the index of the first tree not to be used when applying the model or calculating the metrics (the exclusive right border of the range). Indices are zero-based.

**Possible types**

int

**Default value**

0 (the index of the last tree to use equals to the number of trees in the model minus one)

### virtual_ensembles_count

#### Description

Number of tree ensembles to use. Each virtual ensemble can be considered as a truncated model.

**Possible types**

int

**Default value**

0 (the index of the last tree to use equals to the number of trees in the model minus one)


### thread_count

#### Description

The number of threads to use during the training.

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

**prediction_type VirtEnsembles**

Each virtual ensemble can be considered as truncated model. Returns virtual_ensembles_count predictions from each virtual ensemble. The return value type depends on the number of input objects and model type:

- Single object — Return numpy.ndarray one-dimensional or two-dimensional numpy.ndarray of shape (virtual_ensembles_count) or (virtual_ensembles_count, single document predict size) of virtual_ensemble.predict(document, prediction_type='RawFormulaVal') results. For model learned with RMSEWithUncertainty for virtual ensembles predictions used prediction_type='RMSEWithUncertainty' instead of prediction_type='RawFormulaVal'.
- Multiple objects — two-dimensional or three-dimensional numpy.ndarray of shape (number_of_objects, virtual_ensembles_count) or (number_of_objects, virtual_ensembles_count, single document predict size) similarly to Single object predict type.

**prediction_type TotalUncertainty**

- Single object
    - Regression (not RMSEWithUncertainty): one-dimensional numpy.ndarray [Mean Predictions, Knowledge Uncertainty]
    - RMSEWithUncertainty: one-dimensional numpy.ndarray [Mean Predictions, Knowledge Uncertainty, Data Uncertainty]
    - Classification: one-dimensional numpy.ndarray [Data Uncertainty, Total Uncertainty]

- Multiple objects

    Return two-dimensional numpy.ndarray of shape (number_of_objects, 2) or (number_of_objects, 3) similarly to single object return type.prediction_type VirtEnsembles:


Additional information is available in the [article](https://towardsdatascience.com/tutorial-uncertainty-estimation-with-catboost-255805ff217e).

