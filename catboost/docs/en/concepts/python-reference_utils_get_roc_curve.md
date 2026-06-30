# get_roc_curve

{% include [utils-get_roc_curve__desc](../_includes/work_src/reusage-python/get_roc_curve__desc.md) %}


This information is used to plot the ROC curve.

## {{ dl--invoke-format }} {#call-format}

```python
get_roc_curve(model,
              data,
              thread_count=-1,
              plot=False)
```

## {{ dl--parameters }} {#parameters}

### model

#### Description

The trained model.

**Possible types**

{{ python-type__catboostCatBoost }}

**Default value**

{{ python--required }}

### data

#### Description

A set of samples to build the ROC curve with.

**Possible types**

- {{ python-type--pool }}
- list of {{ python-type--pool }}

**Default value**

{{ python--required }}


### thread_count

#### Description

The number of threads to use.

{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible type**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

### plot

#### Description

Plot a chart based on the found points.

**Possible types**

{{ python-type--bool }}

**Default value**

False

## {{ dl--output-format }} {#output-format}

{{ python-type__for-roc-curve__tuple_of_three_arrays }}

## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve

train_data = [[1,3],
              [0,4],
              [1,7],
              [0,3]]
train_labels = [1,0,1,1]
catboost_pool = Pool(train_data, train_labels)
model = CatBoostClassifier(learning_rate=0.03)
model.fit(train_data, train_labels, verbose=False)
(fpr, tpr, thresholds) = get_roc_curve(model, catboost_pool, plot=True)
print(fpr)
print(tpr)
print(thresholds)
```

Output:
```bash
[0. 0. 0. 0. 1.]
[0.         0.33333333 0.66666667 1.         1.        ]
[1.         0.53533186 0.52910032 0.50608183 0.        ]
```

