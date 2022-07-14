# {{ function-name__get-feature-importance }}

{% include [sections-with-methods-desc-feature_importances--purpose](../_includes/work_src/reusage/feature_importances--purpose.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
get_feature_importance(data=None,
                       reference_data=None,
                       type=EFstrType.FeatureImportance,
                       prettified=False,
                       thread_count=-1,
                       verbose=False,
                       log_cout=sys.stdout,
                       log_cerr=sys.stderr)
```

## {{ dl--parameters }} {#parameters}

### data

#### Description

The dataset for feature importance calculation.

The required dataset depends on the selected feature importance calculation type (specified in the `type` parameter):

- {{ title__regular-feature-importance-PredictionValuesChange }} —  Either None or the same dataset that was used for training if the model does not contain information regarding the weight of leaves. All models trained with {{ product }} version 0.9 or higher contain leaf weight information by default.
- {{ title__regular-feature-importance-LossFunctionChange }}  —  Any dataset. Feature importances are calculated on a subset for large datasets.
- {{ title__predictiondiff }} —  A list of object pairs.

**Possible types**

{{ python-type--pool }}

**Default value**

{{ python--required }} for the {{ title__regular-feature-importance-LossFunctionChange }} and {{ title__ShapValues }} type of feature importances and in case the model does not contain information regarding the weight of leaves.

None otherwise.

### reference_data

#### Description
Reference data for Independent Tree SHAP values from [Explainable AI for Trees: From Local Explanations to Global Understanding](https://arxiv.org/abs/1905.04610v1). If `type` is [`ShapValues`](shap-values.md) and `reference_data` is not `None`, then Independent Tree SHAP values are calculated.

**Possible types**

{{ python-type--pool }}

**Default value**

None

### type

_Alias:_`fstr_type` (deprecated, use `type` instead)

#### Description

The type of feature importance to calculate.

Possible values:
- FeatureImportance: Equal to [{{ title__regular-feature-importance-PredictionValuesChange }}](../concepts/fstr.md#regular-feature-importance) for non-ranking metrics and [{{ title__regular-feature-importance-LossFunctionChange }}](../concepts/fstr.md#regular-feature-importances__lossfunctionchange) for ranking metrics (the value is determined automatically).

- [{{ title__ShapValues }}](../concepts/shap-values.md): A vector $v$ with contributions of each feature to the prediction for every input object and the expected value of the model prediction for the object (average prediction given no knowledge about the object).
- [{{ title__Interaction }}](../concepts/feature-interaction.md#feature-interaction-strength): The value of the feature interaction strength for each pair of features.

- {{ title__predictiondiff }}: A vector with contributions of each feature to the {{ prediction-type--RawFormulaVal }} difference for each pair of objects.

**Possible types**

- {{ python-type--string }}
- [{{ python-type__EFStrType }}](../concepts/python-efstr-type__desc.md)

{% note info %}

It is recommended to use {{ python-type__EFStrType }} for this parameter.

{% endnote %}

**Default value**

FeatureImportance

### prettified

#### Description

Return the feature importances as a list of the following pairs sorted by feature importance:
```
(feature_id, feature importance)
```

Should be used if one of the following values of the `type`parameter is selected:
- {{ title__regular-feature-importance-PredictionValuesChange }}
- {{ title__regular-feature-importance-PredictionValuesChange }}

**Possible types**

{{ python-type--bool }}

**Default value**

False

### thread_count

#### Description

{% include [reusage-thread-count-short-desc](../_includes/work_src/reusage/thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

### verbose

#### Description

The purpose of this parameter depends on the type of the given value:

- {{ python-type--bool }} —  Output progress to stdout.

    Works with the [{{ title__ShapValues }}](../concepts/shap-values.md) type of feature importance calculation.

- {{ python-type--int }} —  The logging period.


**Possible types**

- {{ python-type--bool }}
- {{ python-type--int }}

**Default value**

False

{% include [python__log-params](../_includes/work_src/reusage-python/python__log-params.md) %}

## {{ dl--output-format }} {#output-format}

{% include [sections-with-methods-desc-feature_importances--output-format](../_includes/work_src/reusage/feature_importances--output-format.md) %}


