Export of CatBoost model as standalone Python code
==================================================

Catboost model could be saved as standalone Python code. This can ease an integration of a generated model into a Python application, or allow manual exploration and editing of the model parameters by advanced users.

The exported model code contains complete data for the current trained model and *apply_catboost_model()* function which applies the model to a given dataset. The only current dependency for the code is [CityHash library](https://github.com/Amper/cityhash/tree/4f02fe0ba78d4a6d1735950a9c25809b11786a56).


### Exporting from Catboost application via command line interface:

```bash
catboost fit --model-format Python <other_fit_parameters>
```

By default model is saved into *model.py* file, one could alter the output name using *-m* key. If there is more that one model-format specified, then the *.py* extention will be added to the name provided after *-m* key.


### Exporting from Catboost python library interface:

```python
model = CatBoost(<train_params>)
model.fit(train_pool)
model.save_model(OUTPUT_PYTHON_MODEL_PATH, format="python")
```


## Models trained with only Float features

If the model was trained using only numerical features (no cat features), then the application function in generated code will have the following interface:

```python
def apply_catboost_model(float_features):
```


### Parameters

| parameter      | type                       | description                                      |
|----------------|----------------------------|--------------------------------------------------|
| float_features | list of int or float values| features of a single document to make prediction |


### Return value

Prediction of the model for the document with given features, equivalent to CatBoost().predict(prediction_type='RawFormulaVal').


## Models trained with Categorical features

If the model was trained with categorical features present, then the application function in output code will be generated with the following interface:

```python
def apply_catboost_model(float_features, cat_features):
```


### Parameters

| parameter      | type                                 | description                               |
|----------------|--------------------------------------|-------------------------------------------|
| float_features | list of int or float features        | numerical features of a single document   |
| cat_features   | list of str or int or float features | categorical features of a single document |

NOTE: You need to pass float and categorical features separately in the same order they appeared in the train dataset. For example if you had features f1,f2,f3,f4, where f2 and f4 were considered categorical, you need to pass here float_features=[f1,f3], cat_features=[f2,f4].


### Return value

Prediction of the model for the document with given features, equivalent to CatBoost().predict(prediction_type='RawFormulaVal').


## Current limitations
- MultiClassification models are not supported.
- apply_catboost_model() function has reference implementation and may lack of performance comparing to native applicator of CatBoost, especially on large models and multiple of documents.


## Troubleshooting

Q: Generated model results differ from native model when categorical features present
A: Please check that the CityHash version 1 is used. Exact required revision of [Python CityHash library](https://github.com/Amper/cityhash/tree/4f02fe0ba78d4a6d1735950a9c25809b11786a56). There is also proper CityHash implementation in [Catboost repository](https://github.com/catboost/catboost/tree/master/library/python/cityhash). This is due other versions of CityHash may produce different hash code for the same string.
