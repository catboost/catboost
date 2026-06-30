# Export a model to Python or C++

The {{ product }} model can be saved as standalone Python or C++ code. This solution simplifies the integration of resulting models to Python and C++ applications, allows to port models to architectures that are not directly supported by {{ product }} (such as IBM z/Architecture) and allows advanced users to manually explore or edit the model parameters.

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}

- [python](#python)
- [cli](#cli)

Depending on the output format the resulting file contains one of the following methods for applying the model:
- Python — [{{ method_name__apply_python_model }}](../concepts/python-reference_apply_catboost_model.md)
- C++ — [C++](../concepts/c-plus-plus-api_applycatboostmodel.md)

{% note alert %}

- {% include [reusage-common-phrases-apply_catboost_model__performance](../_includes/work_src/reusage-common-phrases/apply_catboost_model__performance.md) %}

{% endnote %}

## {{ python-package }}

The following parameters can be set for the corresponding methods and are used when the model is saved:

**{{ features__table__title__python__method }}**

- [save_model](../concepts/python-reference_catboost_save_model.md) ([CatBoost](../concepts/python-reference_catboost.md))
- [save_model](../concepts/python-reference_catboostclassifier_save_model.md) ([CatBoostClassifier](../concepts/python-reference_catboostclassifier.md))
- [save_model](../concepts/python-reference_catboostregressor_save_model.md) ([CatBoostRegressor](../concepts/python-reference_catboostregressor.md))


**{{ features__table__title__python__parameters }}**

`format`

**{{ features__table__title__python__description }}**

{% include [sections-with-methods-desc-python__save__export-format__short_desc](../_includes/work_src/reusage/python__save__export-format__short_desc.md) %}

Set the value of this parameter to “python” or “cpp”.

## {{ title__implementation__cli }}

The following command keys can be specified for the corresponding commands and are used when the model is trained:

**{{ features__table__title__cli__command }}** [catboost fit](../references/training-parameters/index.md)

**{{ features__table__title__cli__keys }}**

`--model-format`

**{{ features__table__title__cli__key-description }}**

{% include [reusage-cli__mode-format__short-desc](../_includes/work_src/reusage/cli__mode-format__short-desc.md) %}

Set the value of this key to “Python” or “CPP”.
