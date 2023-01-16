# Continue learning from baseline

{{ product }} supports training with a baseline model. Training parameters of the baseline model and the one being trained may vary. The final model is the sum of these two models.

## FML

### [Train Formula on FML](../yandex_specific/nirvana-operations/catboost__nirvana__train-formula-on-fml.md)

**Input:** `continueFormula`

#### Description

A FML formula to continue the training for.

## {{ product-nirvana }}

### [Train {{ product }} with {{ product__matrixnet }} interface](../yandex_specific/nirvana-operations/catboost__nirvana__train-catboost-with-matrixnet-interface.md)

**Input**

`externalProgress`

#### Description

The native {{ product }} model file ({{ yandex-specific__model_ops__EConvertModelType__CBM }}) with information regarding the training progress. This file allows to recover the training after an interruption from where it left off.
{% note info %}

- GPU snapshots can not be input if training is performed on CPU and vice versa.
- The number of iterations set in the options defines the final number of iterations of the model (not the number of iterations that should take place after recovering the training).

{% endnote %}

### [CatBoost: Train](../yandex_specific/nirvana-operations/catboost__nirvana__train-catboost.md)

**Input**

`baseline_model`

#### Description

The input model that is used to calculate the baseline for objects of the input dataset.

If set, the training consists of the following steps:
1. The model from this input is applied to input datasets (training and validation).
1. The resulting predictions are used as a baseline to train the new model.

The resulting model is the sum of `baseline_model` and the model obtained on step 2.

{% note info %}

The number of iterations set in the options defines the number of iterations for the model that is obtained on step 2.

{% endnote %}

