# FeaturesData

```python
class FeaturesData(num_feature_data=None,
                   cat_feature_data=None,
                   num_feature_names=None,
                   cat_feature_names=None)
```

## {{ dl--purpose }} {#purpose}

Allows to optimally store the feature data for further passing to the [Pool](python-reference_pool.md) constructor. The creation of pools from this representation is much faster than from generic {{ python-type__np_ndarray }}, {{ python-type--pandasDataFrame }} or {{ python-type--pandasSeries }} if the dataset contains both numerical and categorical features, most of which are numerical. Pass {{ python-type__np_ndarray }} with numpy.float32 dtype to get similar performance with datasets that contain only numerical features.

{% note warning %}

FeaturesData makes no checks at all to the input data. Use it only if there is confidence that everything is being done correctly, and it is preferable to avoid spending additional time on checks. Otherwise, pass the input dataset and target variables directly to the [Pool](python-reference_pool.md) class.

{% endnote %}


## {{ dl--parameters }} {#parameters}

### num_feature_data

#### Description

Numerical features for all objects from the dataset in the form of {{ python-type__np_ndarray }} of shape `(object_count x num_feature_count)` with dtype <q>numpy.float32</q>.

**Possible types**

{{ python-type__np_ndarray }}

**Default value**

None (the dataset does not contain numerical features)

### cat_feature_data

#### Description

Categorical features for all objects from the dataset in the form of {{ python-type__np_ndarray }} of shape `(object_count x cat_feature_count)` with dtype <q>object</q>.

The elements must be of {{ python-type__bytes }} type and should contain UTF-8 encoded strings.
{% note warning %}

Categorical features must be passed as strings, for example:

```
data=FeaturesData(cat_feature_data=np.array([['a','c'], ['b', 'c']], dtype=object))
```
Using other data types (for example, int32) raises an error.

{% endnote %}

**Possible types**

{{ python-type__np_ndarray }}

**Default value**

None (the dataset does not contain categorical features)

###  num_feature_names

#### Description

The names of numerical features in the form of a sequence of strings or bytes.

If the string is represented by the {{ python-type__bytes }} type, it must be UTF-8 encoded.

**Possible types**

- {{ python-type--list-of-strings }}

- {{ python-type__list-of-bytes }}

**Default value**

None (the `num_feature_names` data attribute is set to a list of empty strings)

###  cat_feature_names

#### Description

The names of categorical features in the form of a sequence of strings or bytes.

If the string is represented by the {{ python-type__bytes }} type, it must be UTF-8 encoded.

**Possible types**

- {{ python-type--list-of-strings }}

- {{ python-type__list-of-bytes }}

**Default value**

None (the `cat_feature_names` data attribute is set to a list of empty strings)

## {{ input_data__title__peculiarities }} {#specifics}

- The order of features in the created Pool is the following:
    ```
    [num_features (if any present)][cat_features (if any present)]
    ```

- The feature data must be passed in the same order when applying the trained model.

## {{ dl--methods }} {#methods}

Method | Description
----- | -----
[get_cat_feature_count](python-features-data_get-cat-feature-count.md) | Return the number of categorical features contained in the dataset.|
[get_feature_count](python-features-data_get-feature-count.md) | Return the total number of features (both numerical and categorical) contained in the dataset.
[get_feature_names](python-features-data_get-feature-names.md) | Return the names of features from the dataset.
[get_num_feature_count](python-features-data_get-num-feature-count.md) | Return the number of numerical features contained in the dataset.
[get_object_count](python-features-data_get-object-count.md) | Return the number of objects contained in the dataset.


## {{ dl__usage-examples }} {#usage-examples}

#### [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) with [FeaturesData](../concepts/python-features-data__desc.md)

```python
import numpy as np
from catboost import CatBoostClassifier, FeaturesData
# Initialize data
cat_features = [0,1,2]
train_data = FeaturesData(
    num_feature_data=np.array([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]], dtype=np.float32),
    cat_feature_data=np.array([["a", "b"], ["a", "b"], ["c", "d"]], dtype=object)
)
train_labels = [1,1,-1]
test_data = FeaturesData(
    num_feature_data=np.array([[2, 4, 6, 8], [1, 4, 50, 60]], dtype=np.float32),
    cat_feature_data=np.array([["a", "b"], ["a", "d"]], dtype=object)
)
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')
# Fit model
model.fit(train_data, train_labels)
# Get predicted classes
preds_class = model.predict(test_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(test_data, prediction_type='RawFormulaVal')
```

#### [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) with [Pool](../concepts/python-reference_pool.md) and [FeaturesData](../concepts/python-features-data__desc.md)

```python
import numpy as np
from catboost import CatBoostClassifier, FeaturesData, Pool
# Initialize data
train_data = Pool(
    data=FeaturesData(
        num_feature_data=np.array([[1, 4, 5, 6],
                                   [4, 5, 6, 7],
                                   [30, 40, 50, 60]],
                                   dtype=np.float32),
        cat_feature_data=np.array([["a", "b"],
                                   ["a", "b"],
                                   ["c", "d"]],
                                   dtype=object)
    ),
    label=[1, 1, -1]
)
test_data = Pool(
    data=FeaturesData(
        num_feature_data=np.array([[2, 4, 6, 8],
                                   [1, 4, 50, 60]],
                                   dtype=np.float32),
        cat_feature_data=np.array([["a", "b"],
                                   ["a", "d"]],
                                   dtype=object)
    )
)
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations = 2,
                           learning_rate = 1,
                           depth = 2,
                           loss_function = 'Logloss')
# Fit model
model.fit(train_data)
# Get predicted classes
preds_class = model.predict(test_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(test_data, prediction_type='RawFormulaVal')
```
