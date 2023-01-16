# Output settings

## logging_level {#logging_level}

Command line: `--logging-level`

#### Description

The logging level to output to stdout.

Possible values:
- Silent — Do not output any logging information to stdout.

- Verbose — Output the following data to stdout:

    - optimized metric
    - elapsed time of training
    - remaining time of training

- Info — Output additional information and the number of trees.

- Debug — Output debugging information.

**Type**

{{ python-type--string }}

**Default value**

{% cut "Python package" %}

None (corresponds to the {{ fit--verbose }} logging level)

{% endcut %}

{% cut "R package, Command-line" %}

Verbose

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}

## metric_period {#metric_period}

Command line: `--metric-period`

#### Description

The frequency of iterations to calculate the values of [objectives and metrics](../../concepts/loss-functions.md). The value should be a positive integer.

The usage of this parameter speeds up the training.

{% cut "Python package, Command-line" %}

{% note info %}

It is recommended to increase the value of this parameter to maintain training speed if a GPU processing unit type is used.

{% endnote %}

{% endcut %}

**Type**

 {{ python-type--int }}

**Default value**

{{ fit__metric-period }}

**Supported processing units**

{{ cpu-gpu }}

## verbose {#verbose}

Command line: `--verbose`

_Alias:_`verbose_eval`

#### Description

{% include [sections-with-methods-desc-python__feature-importances__verbose__short-description__list-intro](../../_includes/work_src/reusage/python__feature-importances__verbose__short-description__list-intro.md) %}

- {{ python-type--bool }} — Defines the logging level:
    - <q>True</q>  corresponds to the Verbose logging level
    - <q>False</q> corresponds to the Silent logging level

- {{ python-type--int }} — Use the Verbose logging level and set the logging period to the value of this parameter.

{% cut "Python package, R package" %}

{% note alert %}

Do not use this parameter with the `logging_level` parameter.

{% endnote %}

{% endcut %}

{% cut "Command-line" %}

{% note alert %}

Do not use this parameter with the `--logging-level` parameter.

{% endnote %}

{% endcut %}

**Type**
- {{ python-type--bool }}
- {{ python-type--int }}

**Default value**

{{ train_verbose_fr-of-iterations-to-output__default }}

**Supported processing units**

{{ cpu-gpu }}

## train_dir {#train_dir}

Command line: `--train-dir`

#### Description

The directory for storing the files generated during training.

**Type**

{{ python-type--string }}

**Default value**

{% cut "Python package, R package" %}

{{ fit--train_dir }}

{% endcut %}

{% cut "Command-line" %}

Current directory

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}

## model_size_reg {#model_size_reg}

Command line: `--model-size-reg`

#### Description

The model size regularization coefficient. The larger the value, the smaller the model size. Refer to the [Model size regularization coefficient](../../references/model-size-reg.md) section for details.

Possible values are in the range $[0; \inf)$.

This regularization is needed only for models with categorical features (other models are small). Models with categorical features might weight tens of gigabytes or more if categorical features have a lot of values. If the value of the regularizer differs from zero, then the usage of categorical features or feature combinations with a lot of values has a penalty, so less of them are used in the resulting model.

Note that the resulting quality of the model can be affected. Set the value to 0 to turn off the model size optimization option.

**Type**

 {{ python-type--float }}

**Default value**

{% cut "Python package" %}

None ({{ fit__model_size_reg }})

{% endcut %}

{% cut "R package, Command-line" %}

{{ fit__model_size_reg }}

{% endcut %}

**Supported processing units** {{ cpu-gpu }}

## allow_writing_files {#allow_writing_files}

#### Description

Allow to write analytical and snapshot files during training.

If set to <q>False</q>, the [snapshot](../../features/snapshots.md) and [data visualization](../../features/visualization.md) tools are unavailable.

**Type**

{{ python-type--bool }}

**Default value**

{{ fit--allow-writing-files }}

**Supported processing units**

{{ cpu-gpu }}

## save_snapshot {#save_snapshot}

#### Description

Enable snapshotting for [restoring the training progress after an interruption](../../features/snapshots.md). If enabled, the default period for making snapshots is {{ fit__snapshot-interval__default }} seconds. Use the `snapshot_interval` parameter to change this period.

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

**Type**

{{ python-type--bool }}

**Default value**

{{ fit--save_snapshot }}

**Supported processing units**

{{ cpu-gpu }}

## snapshot_file {#snapshot_file}

#### Description

The name of the file to save the training progress information in. This file is used for [recovering training after an interruption](../../features/snapshots.md).

{% include [reusage-snapshot-filename-desc](../../_includes/work_src/reusage/snapshot-filename-desc.md) %}

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

**Type**

{{ python-type--string }}

**Default value**

{% cut "experiment..." %}

{{ fit--snapshot-file-python }}

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}

## snapshot_interval {#snapshot_interval}

#### Description

The interval between saving snapshots in seconds.

The first snapshot is taken after the specified number of seconds since the start of training. Every subsequent snapshot is taken after the specified number of seconds since the previous one. The last snapshot is taken at the end of the training.

{% include [reusage-common-phrases-snapshot-not-working-for-cv](../../_includes/work_src/reusage-common-phrases/snapshot-not-working-for-cv.md) %}

**Type**

{{ python-type--int }}

**Default value**

{{ fit__snapshot-interval__default }}

**Supported processing units**

{{ cpu-gpu }}

## roc_file {#roc_file}

#### Description

The name of the [output file](../../concepts/output-data_roc-curve-points.md) to save the ROC curve points to. This parameter can only be set in [cross-validation](../../concepts/python-reference_cv.md) mode if the {{ error-function--Logit }} loss function is selected. The ROC curve points are calculated for the test fold.

The output file is saved to the `catboost_info` directory.

**Type**

{{ python-type--string }}

**Default value**

None (the file is not saved)

**Supported processing units**


{{ cpu-gpu }}
