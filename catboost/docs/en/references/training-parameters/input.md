# Input file settings

These parameters are only for Command-line.

## -f, --learn-set {#-f}

#### Description

The path to the input file{% if audience == "internal" %} or table{% endif %} that contains the dataset description.

{% include [files-internal-files-internal__desc__full](../../_includes/work_src/reusage-formats/files-internal__desc__full.md) %}

**{{ cli__params-table__title__default }}**

{{ fit__learn-set }}

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## -t, --test-set {#-t}

#### Description

A comma-separated list of input files that contain the validation dataset description (the format must be the same as used in the training dataset).

{% if audience == "internal" %}

{% include [files-internal-format-for-multimple-files](../../yandex_specific/_includes/reusage-formats/format-for-multimple-files.md) %}

{% include [files-internal-files-internal__desc__possbile-values](../../yandex_specific/_includes/reusage-formats/files-internal__desc__possbile-values.md) %}

{% endif %}

**{{ cli__params-table__title__default }}**

Omitted. If this parameter is omitted, the validation dataset isn't used.

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

{% note alert %}

Only a single validation dataset can be input if the training is performed on GPU (`--task-type` is set to GPU)

{% endnote %}

## --cd, --column-description {#--cd}

#### Description

The path to the input file {% if audience == "internal" %}or table{% endif %} that contains the [columns description](../../concepts/input-data_column-descfile.md).

{% if audience == "internal" %}

{% include [internal__cd-internal-cd-desc](../../yandex_specific/_includes/reusage-formats/internal-cd-desc.md) %}

{% endif %}

**{{ cli__params-table__title__default }}:**

If omitted, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --learn-pairs {#--learn-pairs}

#### Description

The path to the input file that contains the [pairs description](../../concepts/input-data_pairs-description.md) for the training dataset.

This information is used for calculation and optimization of [](../../concepts/loss-functions-ranking.md).

**{{ cli__params-table__title__default }}**

Omitted.

{% include [loss-functions-pairwisemetrics_require_pairs_data](../../_includes/work_src/reusage-common-phrases/pairwisemetrics_require_pairs_data.md) %}

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --test-pairs {#--test-pairs}

#### Description

The path to the input file that contains the [pairs description](../../concepts/input-data_pairs-description.md) for the validation dataset.

{% include [reusage-learn_pairs__where_is_used](../../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}

**{{ cli__params-table__title__default }}**

Omitted.

{% include [loss-functions-pairwisemetrics_require_pairs_data](../../_includes/work_src/reusage-common-phrases/pairwisemetrics_require_pairs_data.md) %}

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --learn-group-weights {#--learn-group-weights}

#### Description

The path to the input file that contains the weights of groups. Refer to the [Group weights](../../concepts/input-data_group-weights.md) section for format details.

{% include [reusage-input-data-group_weights__input-dataset-requirement](../../_includes/work_src/reusage-input-data/group_weights__input-dataset-requirement.md) %}

{% include [reusage-input-data-group_weights__precedence-over-datasetdesc](../../_includes/work_src/reusage-input-data/group_weights__precedence-over-datasetdesc.md) %}

**{{ cli__params-table__title__default }}:**

Omitted (group weights are either read from the dataset description or set to 1 for all groups if absent in the input dataset)

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --test-group-weights {#--test-group-weights}

#### Description

The path to the input file that contains the weights of groups for the validation dataset. Refer to the [Group weights](../../concepts/input-data_group-weights.md) section for format details.

{% include [reusage-input-data-group_weights__input-dataset-requirement](../../_includes/work_src/reusage-input-data/group_weights__input-dataset-requirement.md) %}

{% include [reusage-input-data-group_weights__precedence-over-datasetdesc](../../_includes/work_src/reusage-input-data/group_weights__precedence-over-datasetdesc.md) %}

**{{ cli__params-table__title__default }}:**

Omitted (group weights are either read from the dataset description or set to 1 for all groups if absent in the input dataset)

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --force-unit-auto-pair-weights {#--force-unit-auto-pair-weights}

#### Description

For each auto-generated pair in pairwise losses, set the pair weight equal to one.

**{{ cli__params-table__title__default }}:**

Omitted (for each auto-generated pair, the weight is set equal to the weight of the group containing the elements of the pair)

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --learn-baseline {#--learn-baseline}

#### Description

The path to the input file that contains baseline values for the training dataset. Refer to the [Baseline ](../../concepts/input-data_baseline.md) section for format details.

**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --test-baseline {#--test-baseline}

#### Description

The path to the input file that contains baseline values for the validation dataset. Refer to the [Baseline ](../../concepts/input-data_baseline.md) section for format details.

**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --delimiter {#--delimiter}

#### Description

The delimiter character used to separate the data in the dataset description input file.

Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% include [libsvm-note-restriction-delimiter-separated-format](../../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}

**{{ cli__params-table__title__default }}**

{{ fit__delimiter }}

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}

## --has-header {#--has-header}

#### Description

Read the column names from the first line of the dataset description file if this parameter is set.

{% include [libsvm-note-restriction-delimiter-separated-format](../../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}

**{{ cli__params-table__title__default }}:**

False (the first line is supposed to have the same data as the rest of them)

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}


## --params-file {#--params-files}

#### Description

The path to the input JSON file that contains the training parameters, for example:

```
{
"learning_rate": 0.1,
"bootstrap_type": "No"
}
```

Names of training parameters are the same as for the [{{ python-package }}](../../references/training-parameters/index.md) or the [{{ r-package }}](../../concepts/r-reference_catboost-train.md#parameters-list).

If a parameter is specified in both the JSON file and the corresponding command-line parameter, the command-line value is used.

**{{ cli__params-table__title__default }}**

Omitted

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}


## --nan-mode {#--nan-mode}

#### Description

The method for [processing missing values](../../concepts/algorithm-missing-values-processing.md) in the input dataset.

Possible values:

{% include [reusage-missing-values-mv-processing-methods](../../_includes/work_src/reusage-missing-values/mv-processing-methods.md) %}

Using the  {{ fit__nan_mode__min }} or {{ fit__nan_mode__max }} value of this parameter guarantees that a split between missing values and other values is considered when selecting a new split in the tree.

{% note info %}

The method for processing missing values can be set individually for each feature in the [Custom quantization borders and missing value modes](../../concepts/input-data_custom-borders.md) input file. Such values override the ones specified in this parameter.

{% endnote %}

**{{ cli__params-table__title__default }}**

{{ fit--nan_mode }}

**{{ cli__params-table__title__processing-units-type }}**

{{ cpu-gpu }}


