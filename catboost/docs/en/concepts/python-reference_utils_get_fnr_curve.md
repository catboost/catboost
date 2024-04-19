# get_fnr_curve

{% include [utils-get_fnr_curve__desc](../_includes/work_src/reusage-python/get_fnr_curve__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
get_fnr_curve(model=None,
              data=None,
              curve=None,
              thread_count=-1,
              plot=False)
```

## {{ dl--parameters }} {#parameters}

### model

#### Description

The trained model.

**Possible types**

catboost.CatBoost

**Default value**

None

### data

#### Description

A set of samples to build the FNR curve with.

Should not be used with the `curve` parameter.

**Possible types**

- catboost.Pool
- list of catboost.Pool

**Default value**

None

### curve

#### Description

ROC curve points.

Should not be used with the `data` parameter.

Required if the `data` and `model` parameters are set to None.

It is strictly recommended to use the output of the [get_roc_curve](python-reference_utils_get_roc_curve.md) function as the value of this parameter.

The input data must certain criteria:

- The threshold values should not increase.
- There should not be any repetitions of the fpr-tpr- threshold triplets.


**Possible types**

tuple of three arrays (fpr, tpr, thresholds)

**Default value**

None

### thread_count

#### Description

The number of threads to use.

Optimizes the speed of execution. This parameter doesn't affect results.

**Possible types**

int

**Default value**

-1 (the number of threads is equal to the number of processor cores)

### plot

#### Description

Plot a chart based on the found points.

**Possible types**

bool

**Default value**

False


## {{ dl--output-format }} {#output-format}

tuple of two arrays (thresholds, fnr)

## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve, get_fnr_curve

train_data = [[1,3],
              [0,4],
              [1,7],
              [3,0]]
train_labels = [1,0,1,1]
catboost_pool = Pool(train_data, train_labels)
model = CatBoostClassifier(learning_rate=0.03)
model.fit(train_data, train_labels, verbose=False)
roc_curve_values = get_roc_curve(model, catboost_pool)
(thresholds, fnr) = get_fnr_curve(curve=roc_curve_values, plot=True)
print(thresholds)
print(fnr)

```

Output:

```python
[1.         0.54411915 0.50344403 0.        ]
[1.         0.33333333 0.         0.        ]
```

