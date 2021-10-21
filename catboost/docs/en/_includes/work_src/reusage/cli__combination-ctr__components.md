
Components:
- `CtrType` — The method for transforming categorical features to numerical features.

    Supported methods for training on CPU:

    - Borders
    - Buckets
    - BinarizedTargetMeanValue
    - Counter

    Supported methods for training on GPU:

    - Borders
    - Buckets
    - FeatureFreq
    - FloatTargetMeanValue

- `{{ ctr-types__TargetBorderCount }}` — The number of borders for label value [quantization](../../../concepts/quantization.md). Only used for regression problems. Allowed values are integers from 1 to 255 inclusively. The default value is {{ fit--target_border_count }}.

    This option is available for training on CPU only.

- `TargetBorderType` — The [quantization](../../../concepts/quantization.md) type for the label value. Only used for regression problems.

    Possible values:

    - Median
    - Uniform
    - UniformAndQuantiles
    - MaxLogSum
    - MinEntropy
    - GreedyLogSum

    By default, {{ fit--target_border_type }}.

    This option is available for training on CPU only.

- `CtrBorderCount` — The number of splits for categorical features. Allowed values are integers from 1 to 255 inclusively.
- {% include [ctr-params-ctr__desc__ctrbordertype_intro](ctr__desc__ctrbordertype_intro.md) %}

    {% include [ctr-params-ctr__desc__ctrbordertype__supported-cpu](ctr__desc__ctrbordertype__supported-cpu.md) %}

    Supported values for training on GPU:
    - Uniform
    - Median

- `Prior` — Use the specified priors during training (several values can be specified).

    Possible formats:
    - One number — Adds the value to the numerator.
    - Two slash-delimited numbers (for GPU only) — Use this format to set a fraction. The number is added to the numerator and the second is added to the denominator.
