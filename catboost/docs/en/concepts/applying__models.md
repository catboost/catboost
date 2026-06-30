# Applying models

{% if audience == "internal" %}

{% include [applying_models](../yandex_specific/_includes/applying_models.md) %}

{% endif %}

The following non-standard methods for applying the models are supported:
- {% if audience == "internal" %}

  {% include [yandex-specific-methods](../yandex_specific/_includes/yandex-specific-methods.md) %}

  {% endif %}
- C/C++:
    - [Evaluation library](c-plus-plus-api_dynamic-c-pluplus-wrapper.md)
    - [Standalone evaluator](c-plus-plus-api_header-only-evaluator.md)

- [Java](java-package.md)
- [CoreML](export-coreml.md)
- [Node.js](apply-node-js.md)
- [ONNX](apply-onnx-ml.md)
- [Rust](apply-rust.md)
- [.NET](apply-dotnet.md)
- [{{ other-products__clickhouse }}](../features/catboost-with-clickhouse.md)
- [PMML](apply-pmml.md)
- Models exported as code:
    - [C++](c-plus-plus-api_applycatboostmodel.md)
    - [Python](python-reference_apply_catboost_model.md)

{% if audience == "internal" %}

It is also possible to [convert numerical {{ product }} models to `mn_sse`](../yandex_specific/applying-models/converting_float_only_catboost_models.md).

{% endif %}
