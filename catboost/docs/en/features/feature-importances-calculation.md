# Feature importances

{% include [feature-importance-feature-importance__intro](../_includes/work_src/reusage-common-phrases/feature-importance__intro.md) %}

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}

{% if audience == "internal" %}

## {{ common-text__title__nirvana_cubes }}

{{ common-text__title__nirvana_cubes__dl }} | Options | Description
----- | ----- | -----
[CatBoost: Model Analysis](../yandex_specific/nirvana-operations/catboost__nirvana__model-analysis.md) | `analyse-mode` | Set to <q>fstr</q>.
`fstr-type` | Choose the required feature importance calculation type.

{% endif %}

## {{ python-package }}

{% include [feature-importance-use-one-of-the-following-methods-to-calculate-feature-importances](../_includes/work_src/reusage-common-phrases/use-one-of-the-following-methods-to-calculate-feature-importances.md) %}

- {% include [feature-importance-fit__attributes__feature-importance_](../_includes/work_src/reusage-common-phrases/fit__attributes__feature-importance_.md) %}

- {% include [feature-importance-methods__calc-fstr](../_includes/work_src/reusage-common-phrases/methods__calc-fstr.md) %}

|{{ features__table__title__python__class }}|{{ features__table__title__python__description }}|
|-------------------------------------------|-------------------------------------------------|
|[CatBoost](../concepts/python-reference_catboost.md)|[{{ function-name__get-feature-importance }}](../concepts/python-reference_catboost_get_feature_importance.md)|
|[CatBoostClassifier](../concepts/python-reference_catboostclassifier.md)| [{{ function-name__get-feature-importance }}](../concepts/python-reference_catboostclassifier_get_feature_importance.md)|
| [CatBoostRegressor](../concepts/python-reference_catboostregressor.md)| [{{ function-name__get-feature-importance }}](../concepts/python-reference_catboostregressor_get_feature_importance.md)|

These methods calculate and return the [feature importances](../concepts/fstr.md).

## {{ r-package }}

{% include [feature-importance-use-one-of-the-following-methods-to-calculate-feature-importances](../_includes/work_src/reusage-common-phrases/use-one-of-the-following-methods-to-calculate-feature-importances.md) %}


- Use the `feature_importances`[attribute](../concepts/r-reference.md#attributes) to get the feature importances.

- {% include [feature-importance-methods__calc-fstr](../_includes/work_src/reusage-common-phrases/methods__calc-fstr.md) %}

|{{ features__table__title__r__method }}|{{ features__table__title__python__description }}|
|-------------------------------------------|-------------------------------------------------|
|[catboost.get_feature_importance](../concepts/r-reference_catboost-get_feature_importance.md)|Calculate the [feature importances](../concepts/fstr.md) ([Feature importance](../concepts/output-data_feature-analysis_feature-importance.md) and [Feature interaction strength](../concepts/output-data_feature-analysis_feature-interaction-strength.md)).|



## {{ title__implementation__cli }}

Use the following command to calculate the feature importances during model training:

|{{ features__table__title__cli__command }}| {{ features__table__title__cli__keys }} | {{ features__table__title__cli__key-description }} |
|-------------------------------------------|-------------------------------------------------| -------------------------------------------------|
| [catboost fit](../references/training-parameters/index.md)| `--fstr-file` |  The name of the resulting file that contains [regular feature importance](../concepts/output-data_feature-analysis_feature-importance.md#per-feature-importance) data (see [Feature importance](../concepts/fstr.md)). <br> Set the required file name for further feature importance analysis.|
|                                                         |`--fstr-internal-file`|  The name of the resulting file that contains [internal feature importance](../concepts/output-data_feature-analysis_feature-importance.md#internal-feature-importance) data (see [Feature importance](../concepts/fstr.md)). <br> Set the required file name for further internal feature importance analysis.|

Use the following command to calculate the feature importances after model training:

|{{ features__table__title__cli__command }}| {{ features__table__title__cli__purpose }}|
|-------------------------------------------|-------------------------------------------------|
|[catboost fstr](../concepts/cli-reference_fstr-calc.md)| Calculate feature importances.|

## Related information

[Model analysis](../concepts/model-analysis.md)
