### fname

#### Description

The path to the output model.

**Possible types**

{{ python-type--string }}

**Default value**

{{ python--required }}

### format

#### Description

The output format of the model.

{% include [reusage-python__exporting-models__format](python__exporting-models__format.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{{ fitpython__model-format_default }}

### export_parameters

#### Description

Additional format-dependent parameters for:

- Apple CoreML

    Possible values (all are strings):

    - `prediction_type`. Possible values are <q>probability </q>and <q>raw</q>.

    - `coreml_description`

    - `coreml_model_version`

    - `coreml_model_author`

    - `coreml_model_license`


- ONNX-ML

    {% include [reusage-export-formats-additional-parameters__onnx-model__possible-values__list-only](../reusage-export-formats/additional-parameters__onnx-model__possible-values__list-only.md) %}

    See the [ONNX-ML parameters](../../../references/onnx-parameters.md) reference for details.


- PMML

    Possible values (all are strings):

    {% include [reusage-export-formats-additional-parameters__pmml__possible-values__list-only](../reusage-export-formats/additional-parameters__pmml__possible-values__list-only.md) %}

    See the [PMML parameters](../../../references/pmml-parameters.md) reference for details.



**Possible types**

{{ python-type--dict }}

**Default value**

None

### pool

#### Description

The dataset previously used for training.

This parameter is required if the model contains categorical features and the output format is {{ fitpython__model-format_cpp }}, {{ fitpython__model-format_python }}, or {{ cli__metadata__dump_format__json }}.

{% note info %}

The model can be saved to the {{ cli__metadata__dump_format__json }} format without a pool. In this case it is available for review but it is not applicable.

{% endnote %}

**Possible types**

- {{ python-type--pool }}
- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}
- {{ python-type__FeaturesData }}

**Default value**
None
