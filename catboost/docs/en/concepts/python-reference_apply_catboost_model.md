# Python

## {{ dl--purpose }}
Apply the model in Python format. The method is available within the output Python file with the model description.

{% note alert %}

- {% include [reusage-common-phrases-apply_catboost_model__performance](../_includes/work_src/reusage-common-phrases/apply_catboost_model__performance.md) %}

{% endnote %}


## {{ dl__dependencies }}

{% include [reusage-common-phrases-city-hash](../_includes/work_src/reusage-common-phrases/city-hash.md) %}


## {{ dl--invoke-format }}

{% include [reusage-common-phrases-for-datasets-that-contain-only-numeric-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-only-numeric-features.md) %}


```python
{{ method_name__apply_python_model }}(float_features)
```

{% include [reusage-common-phrases-for-datasets-that-contain-both-numerical-and-categorical-features](../_includes/work_src/reusage-common-phrases/for-datasets-that-contain-both-numerical-and-categorical-features.md) %}


```python
{{ method_name__apply_python_model }}(float_features,
                     cat_features)
```

## {{ dl--parameters }}

### float_features


{% include [exported-models-float-features-desc](../_includes/work_src/reusage-common-phrases/float-features-desc.md) %}

Possible types:
- {{ python-type__list_of_int }}
- {{ python-type__list_of_float }}


### cat_features


{% include [exported-models-categorical-features-list](../_includes/work_src/reusage-common-phrases/categorical-features-list.md) %}

Possible types:
- {{ python-type__list_of_int }}
- {{ python-type__list_of_float }}
- {{ python-type--list-of-strings }}





{% note info %}

{% include [exported-models-numerical-and-categorical-features-start](../_includes/work_src/reusage-common-phrases/numerical-and-categorical-features-start.md) %}


```python
{{ method_name__apply_python_model }}(float_features=[f1,f3],
                     cat_features=[f2,f4])
```

{% endnote %}


## Related information

[`--model-format` key of the command-line train mode](../references/training-parameters/index.md#cli-reference_train-model)

### {{ dl--output-format }}

{{ python-type--numpy-ndarray }} (identical to the `[CatBoost()](python-reference_catboost.md).[predict](python-reference_catboost_predict.md)(prediction_type='RawFormulaVal')` method output)[Train a model](../references/training-parameters/index.md)
