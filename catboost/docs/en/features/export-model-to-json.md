# Export a model to JSON

The trained {{ product }} model can be saved as a JSON file. This file can be accessed later to apply the model. Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.

{% include [feature-importance-choose-the-required-implementation-for-more-details](../_includes/work_src/reusage-common-phrases/choose-the-required-implementation-for-more-details.md) %}


## {{ python-package }}

|methods| {{ features__table__title__python__parameters }}| description|
|-------|-----------|-----------|
|- [save_model](../concepts/python-reference_catboost_save_model.md) ([CatBoost](../concepts/python-reference_catboost.md)) <br> - [save_model](../concepts/python-reference_catboostclassifier_save_model.md) ([CatBoostClassifier](../concepts/python-reference_catboostclassifier.md)) <br> - [save_model](../concepts/python-reference_catboostregressor_save_model.md) ([CatBoostRegressor](../concepts/python-reference_catboostregressor.md))| `format`| The output format of the model. <br> Set this parameter to “json”.|


## {{ title__implementation__cli }}

The following command keys can be specified for the corresponding commands and are used when the model is trained:

|command|{{ features__table__title__cli__keys }}| description|
|-------|-----------|-----------|
|[catboost fit](../references/training-parameters/index.md)| `--model-format`| A comma-separated list of output model formats. <br> Set the value of this key to “json”.|
