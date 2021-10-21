# select_threshold

{% include [utils-select_decision_boundary__desc](../_includes/work_src/reusage-python/select_decision_boundary__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
select_threshold(model=None, 
                 data=None, 
                 curve=None, 
                 FPR=None,
                 FNR=None,
                 thread_count=-1)
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

### FPR

#### Description

Return the boundary at which the given FPR value is reached. Possible values of the parameter are in the range [0; 1].

Should not be used with the FNR parameter.

**Possible types**

float

**Default value**

None.

In this case the conditions for measuring the boundary depend on the value of the FNR parameter:

- None — The boundary should satisfy the FNR=FPR expression
- float in the [0; 1] range The boundary should satisfy the given FNR value

### FNR

#### Description

Return the boundary at which the given FNR value is reached. Possible values of the parameter are in the range [0; 1].

Should not be used with the FPR parameter.

**Possible types**

float

**Default value**

None.

In this case the conditions for measuring the boundary depend on the value of the FPR parameter:

- None — The boundary should satisfy the FNR=FPR expression
- float in the [0; 1] range — The boundary should satisfy the given FPR value

### thread_count

#### Description

The number of threads to use.

Optimizes the speed of execution. This parameter doesn't affect results.

**Possible types**

int

**Default value**

-1 (the number of threads is equal to the number of processor cores)


## {{ dl--output-format }} {#output-format}

{{ python-type--float }}

## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve, select_threshold

train_data = [[1,4],
              [2,5],
              [4,3],
              [0,4]]
train_labels = [1,1,0,1]
catboost_pool = Pool(train_data, train_labels)

model = CatBoostClassifier(learning_rate=0.03)
model.fit(train_data, train_labels, verbose=False)
roc_curve_values = get_roc_curve(model, catboost_pool)

boundary = select_threshold(model, 
                            curve=roc_curve_values,  
                            FPR=0.01)
print(boundary)
```

Output:
```
0.506369291052
```

