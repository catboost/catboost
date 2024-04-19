# Quick start

Use one of the following examples after [installing](python-installation.md) the {{ python-package }} to get started:
- [CatBoostClassifier](#classification)
- [CatBoostRegressor](#regression)
- [CatBoost](#classification-and-regression)


## CatBoostClassifier {#classification}

```python
import numpy as np

from catboost import CatBoostClassifier, Pool

# initialize data
train_data = np.random.randint(0,
                               100,
                               size=(100, 10))

train_labels = np.random.randint(0,
                                 2,
                                 size=(100))

test_data = catboost_pool = Pool(train_data,
                                 train_labels)

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
# train the model
model.fit(train_data, train_labels)
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)

```


## CatBoostRegressor {#regression}

```python
import numpy as np
from catboost import Pool, CatBoostRegressor
# initialize data
train_data = np.random.randint(0,
                               100,
                               size=(100, 10))
train_label = np.random.randint(0,
                                1000,
                                size=(100))
test_data = np.random.randint(0,
                              100,
                              size=(50, 10))
# initialize Pool
train_pool = Pool(train_data,
                  train_label,
                  cat_features=[0,2,5])
test_pool = Pool(test_data,
                 cat_features=[0,2,5])

# specify the training parameters
model = CatBoostRegressor(iterations=2,
                          depth=2,
                          learning_rate=1,
                          loss_function='RMSE')
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds = model.predict(test_pool)
print(preds)
```


## CatBoost {#classification-and-regression}

Datasets can be read from input files. For example, the [Pool](python-reference_pool.md) class offers this functionality.

```python
import numpy as np
from catboost import CatBoost, Pool

# read the dataset

train_data = np.random.randint(0,
                               100,
                               size=(100, 10))
train_labels = np.random.randint(0,
                                2,
                                size=(100))
test_data = np.random.randint(0,
                                100,
                                size=(50, 10))

train_pool = Pool(train_data,
                  train_labels)

test_pool = Pool(test_data)
# specify training parameters via map

param = {'iterations':5}
model = CatBoost(param)
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
preds_class = model.predict(test_pool, prediction_type='Class')
preds_proba = model.predict(test_pool, prediction_type='Probability')
preds_raw_vals = model.predict(test_pool, prediction_type='RawFormulaVal')
print("Class", preds_class)
print("Proba", preds_proba)
print("Raw", preds_raw_vals)
```

