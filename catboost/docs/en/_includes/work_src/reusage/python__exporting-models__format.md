
Possible values:
- {{ fitpython__model-format_cbm }} — {{ product }} binary format.
- {{ fitpython__model-format_coreml }} — Apple CoreML format (only datasets without categorical features are currently supported).
- {{ fitpython__model-format_json }} — JSON format. Refer to the [CatBoost JSON model tutorial]({{ catboost-repo__json-tutorial }}) for format details.
- {{ fitpython__model-format_python }} — Standalone Python code (multiclassification models are not currently supported). See the [Python](../../../concepts/python-reference_apply_catboost_model.md) section for details on applying the resulting model.
- {{ fitpython__model-format_cpp }} — Standalone C++ code (multiclassification models are not currently supported). See the [C++](../../../concepts/c-plus-plus-api_applycatboostmodel.md) section for details on applying the resulting model.
- {{ fitpython__model-format_onnx }} — ONNX-ML format (only datasets without categorical features are currently supported). Refer to [https://onnx.ai/](https://onnx.ai/) for details. See the [ONNX](../../../concepts/apply-onnx-ml.md) section for details on applying the resulting model.
- {% include [reusage-cli__export-model-formats__pmml__p](cli__export-model-formats__pmml__p.md) %}

    {% note info %}

    Multiclassification models are not currently supported.

    {% endnote %}
