
Additional format-dependent parameters for:

{% cut "Apple CoreML" %}

Possible values (all are strings):
 - `prediction_type`. Possible values are <q>probability </q>and <q>raw</q>.
 - `coreml_description`
 - `coreml_model_version`
 - `coreml_model_author`
 - `coreml_model_license`

{% endcut %}

{% cut "ONNX-ML" %}

{% include [reusage-export-formats-additional-parameters__onnx-model__possible-values__list-only](../reusage-export-formats/additional-parameters__onnx-model__possible-values__list-only.md) %}

See the [ONNX-ML parameters](../../../references/onnx-parameters.md) reference for details.

{% endcut %}

{% cut "PMML" %}

Possible values (all are strings):

{% include [reusage-export-formats-additional-parameters__pmml__possible-values__list-only](../reusage-export-formats/additional-parameters__pmml__possible-values__list-only.md) %}

See the [PMML parameters](../../../references/pmml-parameters.md) reference for details.

{% endcut %}
