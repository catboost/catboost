
The default processing mode is {{ fit--nan_mode }}. The following methods for changing the default mode are provided:

- Globally for all features in the
    {% cut "`nan_mode`" %}

    The method for  [processing missing values](../../../concepts/algorithm-missing-values-processing.md) in the input dataset.

    {% include [reusage-cmd__nan-mode__div](../reusage/cmd__nan-mode__div.md) %}

    {% endcut %}

    [ training parameter](../../../references/training-parameters/index.md).
- Individually for each feature in the [Custom quantization borders and missing value modes](../../../concepts/input-data_custom-borders.md) input file. Such values override the global default setting.
