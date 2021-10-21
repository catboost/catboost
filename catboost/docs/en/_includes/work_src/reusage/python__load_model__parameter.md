### fname

#### Description

The path to the input model.

**Possible types**

{{ python-type--string }}

**Default value**

{{ python--required }}

### format

#### Description

The format of the input model.

Possible values:
- {{ fitpython__model-format_cbm }} — {{ product }} binary format.
- {{ fit__model-format_applecoreml }}(only datasets without categorical features are currently supported).
- {{ fitpython__model-format_json }} — JSON format. Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.
- {{ fitpython__model-format_onnx }} — ONNX-ML format (only datasets without categorical features are currently supported). Refer to [https://onnx.ai/](https://onnx.ai/) for details. See the [ONNX](../../../concepts/apply-onnx-ml.md) section for details on applying the resulting model.

**Possible types**

{{ python-type--string }}

**Default value**

{{ fitpython__model-format_cbm }}
