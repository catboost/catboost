# get_confusion_matrix

{% include [utils-get_confusion_matrix__desc](../_includes/work_src/reusage-python/get_confusion_matrix__desc.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```python
get_confusion_matrix(model, data, thread_count)
```

## {{ dl--parameters }} {#parameters-list}

### model

#### Description

The trained model.

**Possible types**

{{ python-type__catboostCatBoost }}

**Default value**

{{ python--required }}

### data

#### Description

A set of samples to build the confusion matrix with.

**Possible types**

{{ python-type--pool }}

**Default value**

{{ python--required }}

### thread_count

#### Description

The number of threads to use.

**Possible types**

{{ python-type--int }}

**Default value**

-1 (the number of threads is set to the number of CPU cores)


## {{ dl--output-format }} {#output-data-format}

confusion matrix : array, shape = [n_classes, n_classes]

## {{ dl--example }} {#examples}

#### Multiclassification

```python
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_confusion_matrix

train_data = [[1, 1924, 44],
              [1, 1932, 37],
              [0, 1980, 37],
              [1, 2012, 204]]

train_label = ["France", "USA", "USA", "UK"]

train_dataset = Pool(data=train_data,
                     label=train_label)

model = CatBoostClassifier(loss_function='MultiClass',
                           iterations=100,
                           verbose=False)

model.fit(train_dataset)

cm = get_confusion_matrix(model, Pool(train_data, train_label))
print(cm)

```

Output:

```bash
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 2.]]
```

#### Binary classification

```python
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_confusion_matrix

train_data = [[1, 1924, 44],
              [1, 1932, 37],
              [0, 1980, 37],
              [1, 2012, 204]]

train_label = [0, 1, 1, 0]

train_dataset = Pool(data=train_data,
                     label=train_label)

model = CatBoostClassifier(loss_function='Logloss',
                           iterations=100,
                           verbose=False)

model.fit(train_dataset)

cm = get_confusion_matrix(model, Pool(train_data, train_label))
print(cm)

```

Output:

```bash
[[2. 0.]
 [0. 2.]]
```

