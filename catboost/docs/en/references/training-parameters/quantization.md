# Quantization settings

## target_border {#target_border}

Command-line: `--target-border`

#### Description

If set, defines the border for converting target values to 0 and 1.

Depending on the specified value:

- $target\_value \le border\_value$ the target is converted to 0
- $target\_value > border\_value$ the target is converted to 1

**Type**

{{ python-type--float }}

**Default value**

{% cut "Python package, R package" %}

None

{% endcut %}

{% cut "Command line" %}

The value is not set

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}

## border_count {#border_count}

Command-line: `-x`, `--border-count`

_Alias:_ `max_bin`

#### Description

The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.

**Type**

{{ python-type--int }}

**Default value**

{% include [reusage-default-values-border_count](../../_includes/work_src/reusage-default-values/border_count.md) %}

**Supported processing units**

{{ cpu-gpu }}

## feature_border_type {#feature_border_type}

Command-line: `--feature-border-type`

#### Description

The [quantization mode](../../concepts/quantization.md) for numerical features.

Possible values:
- Median
- Uniform
- UniformAndQuantiles
- MaxLogSum
- MinEntropy
- GreedyLogSum

**Type**

{{ python-type--string }}

**Default value**

{{ fit--feature_border_type }}

**Supported processing units**

{{ cpu-gpu }}

## per_float_feature_quantization {#per_float_feature_quantization}

Command-line: `--per-float-feature-quantization`

#### Description

The quantization description for the specified feature or list of features.

Description format for a single feature:
```
FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]
```

Examples:

- ```
    per_float_feature_quantization='0:border_count=1024'
    ```

  In this example, the feature indexed 0 has 1024 borders.

- ```python
    per_float_feature_quantization=['0:border_count=1024', '1:border_count=1024']
    ```

  In this example, features indexed 0 and 1 have 1024 borders.

**Type**

- {{ python-type--string }}
- {{ python-type--list-of-strings }}

**Default value**

{% cut "Python package, R package" %}

None

{% endcut %}

{% cut "Command-line" %}

Ommited

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}
