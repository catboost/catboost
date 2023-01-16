### --fstr-type

#### Description

The [feature importance](../../../concepts/fstr.md) output format.

Possible values:
- [{{ title__regular-feature-importance-PredictionValuesChange }}](../../../concepts/fstr.md#regular-feature-importance)
- [{{ title__regular-feature-importance-LossFunctionChange }}](../../../concepts/fstr.md#regular-feature-importances__lossfunctionchange)
- [{{ title__internal-feature-importance }}](../../../concepts/fstr.md#internal-feature-importance)
- [{{ title__Interaction }}](../../../concepts/feature-interaction.md#feature-interaction-strength)
- [{{ title__InternalInteraction }}](../../../concepts/feature-interaction.md#internal-feature-interaction-strength)
- [{{ title__ShapValues }}](../../../concepts/shap-values.md)

**{{ cli__params-table__title__default }}**

Required parameter


### -m, --model-file, --model-path

#### Description

{% include [concept_pcd_bsy_xz-cli__m-desc__short](cli__m-desc__short.md) %}


**{{ cli__params-table__title__default }}**

{{ calc--model-path }}

### --model-format

#### Description

The format of the input model.

Possible values:
- {{ fit__model-format_CatboostBinary }}.
- {{ fit__model-format_applecoreml }} (only datasets without categorical features are currently supported).
- {{ fit__model-format_json }} (multiclassification models are not currently supported). Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.


**{{ cli__params-table__title__default }}**

{{ fit__model-format }}

### --input-path

#### Description

{% include [concept_pcd_bsy_xz-cli__input-path-desc__short](cli__input-path-desc__short.md) %}


{% include [reusage-leaf-weights-not-in-the-model](leaf-weights-not-in-the-model.md) %}


**{{ cli__params-table__title__default }}**

{{ fstr--input-path }}


### --column-description, --cd

#### Description

{% include [reusage-cd-short-desct](cd-short-desct.md) %}


{% include [reusage-leaf-weights-not-in-the-model](leaf-weights-not-in-the-model.md) %}


****{{ cli__params-table__title__default }}****

If omitted, it is assumed that the first column in the file with the dataset description defines the label value, and the other columns are the values of numerical features.


### -o, --output-path

#### Description

The path to the output file with data for [feature analysis](../../../concepts/output-data_feature-analysis.md).

**{{ cli__params-table__title__default }}**

{{ fstr--output-path }}


### -T, --thread-count

#### Description

{% include [reusage-thread-count-short-desc](thread-count-short-desc.md) %}


{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}


**{{ cli__params-table__title__default }}**

{{ fit--thread_count }}
