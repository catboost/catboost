# sum_models

## {{ dl--purpose }} {#purpose}

{% include [sum_limits-python__sum-limits__desc](../_includes/work_src/reusage-python/python__sum-limits__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
sum_models(models,
           weights=None,
           ctr_merge_policy='IntersectingCountersAverage')
```

## {{ dl--parameters }} {#parameters}

{% include [sum_limits-python__sum-limits__parameters](../_includes/work_src/reusage-python/python__sum-limits__parameters.md) %}


{% note info %}

- The bias of the models sum is equal to the weighted sum of models biases.
- The scale of the models sum is equal to 1, leaf values are scaled before the summation.

{% endnote %}

## {{ dl--output-format }} {#usage-example}

{{ product }} model

## {{ input_data__title__example }} {#example}

```python
from catboost import CatBoostClassifier, Pool, sum_models
from catboost.datasets import amazon
import numpy as np
from sklearn.model_selection import train_test_split

train_df, _ = amazon()

y = train_df.ACTION
X = train_df.drop('ACTION', axis=1)

categorical_features_indices = np.where(X.dtypes != np.float)[0]

X_train, X_validation, y_train, y_validation = train_test_split(X,
                                                                y,
                                                                train_size=0.8,
                                                                random_state=42)

train_pool = Pool(X_train,
                  y_train,
                  cat_features=categorical_features_indices)
validate_pool = Pool(X_validation,
                     y_validation,
                     cat_features=categorical_features_indices)

models = []
for i in range(5):
    model = CatBoostClassifier(iterations=100,
                               random_seed=i)
    model.fit(train_pool,
              eval_set=validate_pool)
    models.append(model)

models_avrg = sum_models(models,
                         weights=[1.0/len(models)] * len(models))
```

