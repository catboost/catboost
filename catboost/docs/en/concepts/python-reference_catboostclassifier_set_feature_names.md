# set_feature_names

{% include [non_pool__set_feature_names-non_pool__set_feature_names__div](../_includes/work_src/reusage-python/non_pool__set_feature_names__div.md) %}

## {{ dl--invoke-format }} {#method-call-format}

```python
set_feature_names(feature_names)
```

## {{ dl--parameters }} {#parameters}

### feature_names

#### Description

A one-dimensional array of feature names for each feature. The order and number of specified names must match the ones used in the dataset.

**Possible types**

- ndarray
- list

**Default value**

{{ python--required }}



## {{ output--example }} {#example}

{% include [non_pool__set_feature_names-example-intro](../_includes/work_src/reusage-python/example-intro.md) %}


```python
from catboost import CatBoostClassifier

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]
train_labels = [0, 0, 1, 1]

model = CatBoostClassifier(loss_function='Logloss')
model.fit(train_data, train_labels, verbose=False)

print("Original feature names:")
print(model.feature_names_)
model.set_feature_names(["feature_1", "feature_2"])
print("Changed feature names:")
print(model.feature_names_)

```

Output:

```no-highlight
Original feature names:
['0', '1']
Changed feature names:
['feature_1', 'feature_2']
```

