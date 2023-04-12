# Recovering training after an interruption

During the training, {{ product }} makes snapshots — backup copies of intermediate results. If an unexpected interruption occurs (for instance, the computer accidentally turns off), training can be continued from the saved state. In this case, the completed iterations of building trees don't need to be repeated.

Saving snapshots can be enabled when setting training parameters. Refer to the descriptions below for details. If enabled, the default period for making snapshots is {{ fit__snapshot-interval__default }} seconds. This period can be changed with training parameters described below.

To restore an interrupted training from a previously saved snapshot, launch the training in the same folder with the same parameters. In this case, {{ product }} finds the snapshot and resumes the training from the iteration where it has stopped.

{% if audience == "internal" %}

#### {{ common-text__title__nirvana_cubes }}

**{{ common-text__title__nirvana_cubes__dl }}** [CatBoost: Train](../yandex_specific/nirvana-operations/catboost__nirvana__train-catboost.md)

**Input:**
`snapshot_file`

#### Description


The native {{ product }} model file ({{ yandex-specific__model_ops__EConvertModelType__CBM }}) with information regarding the training progress. This file allows to recover the training after an interruption from where it left off.

{% note info %}

- GPU snapshots can not be input if training is performed on CPU and vice versa.
- The number of iterations set in the options defines the final number of iterations of the model (not the number of iterations that should take place after recovering the training).

{% endnote %}

{% endif %}

{% list tabs %}

- {{ python-package }}

  **{{ features__table__title__python__method }}**

  - [fit](../concepts/python-reference_catboost_fit.md) ([CatBoost](../concepts/python-reference_catboost.md))
  - [fit](../concepts/python-reference_catboostclassifier_fit.md) ([CatBoostClassifier](../concepts/python-reference_catboostclassifier.md))
  - [fit](../concepts/python-reference_catboostregressor_fit.md) ([CatBoostRegressor](../concepts/python-reference_catboostregressor.md))


  **{{ features__table__title__python__parameters }}**

  {% cut "save_snapshot" %}

    `save_snapshot`

   {% include [python-save-snapshot__python-desc__short-desc](../_includes/work_src/reusage/save-snapshot__python-desc__short-desc.md) %}

   Set this parameter to "True".

  {% endcut %}

  {% cut "save_snapshot" %}

   `save_snapshot`

   The name of the file to save the training progress information in. This file is used for [recovering training after an interruption](../features/snapshots.md).

   {% include [reusage-snapshot-filename-desc](../_includes/work_src/reusage/snapshot-filename-desc.md) %}

   {% include [reusage-common-phrases-snapshot-not-working-for-cv](../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

  {% endcut %}

  {% cut "snapshot_interval" %}

   `snapshot_interval`

   {% include [python-python__snapshot_interval__desc__div](../_includes/work_src/reusage/python__snapshot_interval__desc__div.md) %}

  {% endcut %}

- {{ r-package }}

  **{{ features__table__title__r__method }}** [catboost.train](../concepts/r-reference_catboost-train.md)

  **{{ features__table__title__r__parameters }}**

  {% cut "save_snapshot" %}

     `save_snapshot`

     {% include [python-save-snapshot__python-desc__short-desc](../_includes/work_src/reusage/save-snapshot__python-desc__short-desc.md) %}

     Set this parameter to "True".

  {% endcut %}

  {% cut "snapshot_file" %}

    `snapshot_file`

    The name of the file to save the training progress information in. This file is used for [recovering training after an interruption](../features/snapshots.md).

    {% include [reusage-snapshot-filename-desc](../_includes/work_src/reusage/snapshot-filename-desc.md) %}

     {% include [reusage-common-phrases-snapshot-not-working-for-cv](../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

  {% endcut %}

  {% cut "snapshot_interval" %}

   `snapshot_interval`

    {% include [python-python__snapshot_interval__desc__div](../_includes/work_src/reusage/python__snapshot_interval__desc__div.md) %}

  {% endcut %}

- {{ title__implementation__cli }}

  **{{ features__table__title__cli__command }}** [catboost fit](../references/training-parameters/index.md)

  **{{ features__table__title__cli__keys }}**

   {% cut "--snapshot-file" %}

   `--snapshot-file`

    {% include [reusage-cli__snapshot-file-desc__div](../_includes/work_src/reusage/cli__snapshot-file-desc__div.md) %}

     Use this parameter to enable snapshotting.

    {% endcut %}

   {% cut "--snapshot-interval" %}

    `--snapshot-interval`

    {% include [python-python__snapshot_interval__desc__div](../_includes/work_src/reusage/python__snapshot_interval__desc__div.md) %}

   {% endcut %}

{% endlist %}