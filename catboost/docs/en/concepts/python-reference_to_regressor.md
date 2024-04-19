# to_regressor

## {{ dl--purpose }} {#purpose}

Convert a model of type [CatBoost](python-reference_catboost.md) to a model of type [CatBoostRegressor](python-reference_catboostregressor.md). The model can be converted if the loss function of the source model is compatible with the one of the resulting model.

## {{ dl--invoke-format }} {#call-format}

```python
to_regressor(model)
```

## {{ dl--parameters }} {#parameters}

{% include [to_classifier-regressor-to__classifier-regressor__table](../_includes/work_src/reusage-python/to__classifier-regressor__table.md) %}


## {{ dl--output-format }} {#usage-example}

{{ python-type__catboost-core-CatBoostRegressor }}

## {{ input_data__title__example }} {#example}

```python
from catboost import Pool, to_regressor, CatBoost

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]

train_labels = [0, 0, 1, 1]

model = CatBoost(params={'loss_function': 'Logloss'})

model.fit(train_data,
          train_labels,
          verbose=False)

print("Source model type: ", type(model))
converted_model = to_regressor(model)
print("Converted model type: ",  type(converted_model))

```

{% include [reusage-common-phrases-example-output](../_includes/work_src/reusage-common-phrases/example-output.md) %}


```bash
('Source model type: ', <class 'catboost.core.CatBoost'>)
('Converted model type: ', <class 'catboost.core.CatBoostRegressor'>)
```

