# Export a model to CoreML

{% include [export-to-core-ml-core-mle__intro](../_includes/work_src/reusage-python/core-mle__intro.md) %}


{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}


## {{ python-package }}

**{{ features__table__title__python__method }}**

- [save_model](../concepts/python-reference_catboost_save_model.md) ([CatBoost](../concepts/python-reference_catboost.md))
- [save_model](../concepts/python-reference_catboostclassifier_save_model.md) ([CatBoostClassifier](../concepts/python-reference_catboostclassifier.md))
- [save_model](../concepts/python-reference_catboostregressor_save_model.md) ([CatBoostRegressor](../concepts/python-reference_catboostregressor.md))


**{{ features__table__title__python__parameters }}**

`format`

**{{ features__table__title__python__description }}**


{% include [sections-with-methods-desc-python__save__export-format__short_desc](../_includes/work_src/reusage/python__save__export-format__short_desc.md) %}


Set this parameter to “coreml”.

**{{ features__table__title__python__method }}**

 `export_parameters`

**{{ features__table__title__python__parameters }}**

{% include [sections-with-methods-desc-python__export-parameters__intro](../_includes/work_src/reusage/python__export-parameters__intro.md) %}


The following values are supported for Apple CoreML:

- `prediction_type`. Possible values are <q>probability </q>and <q>raw</q>.

- `coreml_description`

- `coreml_model_version`

- `coreml_model_author`

- `coreml_model_license`


Refer to [the example](../concepts/python-usages-examples.md) for more details.

## {{ title__implementation__cli }}

The following command keys can be specified for the corresponding commands and are used when the model is trained:

**{{ features__table__title__cli__command }}**

[catboost fit](../references/training-parameters/index.md)

**{{ features__table__title__cli__keys }}**

`--model-format`

**{{ features__table__title__cli__key-description }}**

{% include [reusage-cli__mode-format__short-desc](../_includes/work_src/reusage/cli__mode-format__short-desc.md) %}


Set the value of this key to “json”.

