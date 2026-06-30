# quantize

{% include [quantized-python_quantized](../_includes/work_src/reusage-python/python_quantized.md) %}

## {{ dl--invoke-format }} {#call-format}

```python
quantize(ignored_features=None,
         per_float_feature_quantization=None,
         border_count=None,
         max_bin=None,
         feature_border_type=None,
         dev_efb_max_buckets=None,
         nan_mode=None,
         input_borders=None,
         simple_ctr=None,
         combinations_ctr=None,
         per_feature_ctr=None,
         ctr_target_border_count=None,
         task_type=None,
         used_ram_limit=None)
```

## {{ dl--parameters }} {#parameters}

### ignored_features

#### Description

{% include [reusage-ignored-feature__common-div](../_includes/work_src/reusage/ignored-feature__common-div.md) %}

For example, use the following construction if features indexed 1, 2, 7, 42, 43, 44, 45, should be ignored:
```
[1,2,7,42,43,44,45]
```

**Possible types**

{{ python-type--list }}

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}

### per_float_feature_quantization

#### Description

The quantization  for the given list of features (one or more).

{% include [python-python__per_floa_feature_quantization__-format__intro-and-format](../_includes/work_src/reusage/python__per_floa_feature_quantization__-format__intro-and-format.md) %}


Examples:
- ```
    per_float_feature_quantization=['0:1024']
    ```

    {% include [python-feature_0_has_1024_borders](../_includes/work_src/reusage/feature_0_has_1024_borders.md) %}

    The following example is equivalent to the one given above:

    ```
    per_float_feature_quantization=['0:border_count=1024']
    ```

- ```
    per_float_feature_quantization=['0:border_count=1024','1:border_count=1024']
    ```

    {% include [python-feature_0-1_have_1024_borders](../_includes/work_src/reusage/feature_0-1_have_1024_borders.md) %}


**Possible types**

{{ python-type--list-of-strings }}

**Default value**

None

**Supported processing units**

{{ cpu-gpu }}


### border_count

_Alias:_`max_bin`

#### Description

The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.

**Possible types**

{{ python-type--int }}

**Default value**

{% include [reusage-default-values-border_count](../_includes/work_src/reusage-default-values/border_count.md) %}

**Supported processing units**

{{ cpu-gpu }}


### feature_border_type

#### Description

The [quantization mode](../concepts/quantization.md) for numerical features.

Possible values:
- Median
- Uniform
- UniformAndQuantiles
- MaxLogSum
- MinEntropy
- GreedyLogSum

**Possible types**

{{ python-type--string }}

**Default value**

{{ fit--feature_border_type }}

**Supported processing units**

{{ cpu-gpu }}

### dev_efb_max_buckets

#### Description

Maximum bucket count in exclusive features bundle.

Possible values are in the range [0, 65536]

**Possible types**

{{ python-type--int }}

**Default value**

None

**Supported processing units**

{{ calcer_type__cpu }}

### nan_mode

#### Description

{% include [python-python__nan-mode_preprocessing__intro](../_includes/work_src/reusage/python__nan-mode_preprocessing__intro.md) %}


{% include [reusage-cmd__nan-mode__list-only__div](../_includes/work_src/reusage/cmd__nan-mode__list-only__div.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{{ fit--nan_mode }}

**Supported processing units**

{{ cpu-gpu }}

### input_borders

#### Description

{% include [reusage-cli__input-borders-file__desc__div](../_includes/work_src/reusage/cli__input-borders-file__desc__div.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{{ cli__input-borders-file__default }}

**Supported processing units**

{{ cpu-gpu }}

### simple_ctr

#### Description

{% include [reusage-cli__simple-ctr__intro](../_includes/work_src/reusage/cli__simple-ctr__intro.md) %}


Format:

```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__simple-ctr__components](../_includes/work_src/reusage/cli__simple-ctr__components.md) %}


{% include [reusage-cli__simple-ctr__examples__p](../_includes/work_src/reusage/cli__simple-ctr__examples__p.md) %}

**Possible types**

{{ python-type--string }}


**Supported processing units**

{{ cpu-gpu }}


### combinations_ctr

#### Description

{% include [reusage-cli__combination-ctr__intro](../_includes/work_src/reusage/cli__combination-ctr__intro.md) %}


```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__combination-ctr__components](../_includes/work_src/reusage/cli__combination-ctr__components.md) %}

**Possible types**

{{ python-type--string }}

**Supported processing units**

{{ cpu-gpu }}


### per_feature_ctr

#### Description


{% include [reusage-cli__per-feature-ctr__intro](../_includes/work_src/reusage/cli__per-feature-ctr__intro.md) %}


```
['FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__per-feature-ctr__components](../_includes/work_src/reusage/cli__per-feature-ctr__components.md) %}

**Possible types**

{{ python-type--string }}

**Supported processing units**

{{ cpu-gpu }}


### ctr_target_border_count

#### Description

{% include [reusage-cli__ctr-target-border-count__short-desc](../_includes/work_src/reusage/cli__ctr-target-border-count__short-desc.md) %}


The value of the `{{ ctr-types__TargetBorderCount }}` component overrides this parameter if it is specified for one of the following parameters:

- `simple_ctr`
- `combinations_ctr`
- `per_feature_ctr`

**Possible types**

{{ python-type--int }}

**Default value**

{{ parameters__ctr-target-border-count__default }}

**Supported processing units**

{{ cpu-gpu }}


### task_type

#### Description

The processing unit type to use for training.

Possible values:
- CPU
- GPU

**Possible types**

{{ python-type--string }}

**Default value**

{{ fit__python-r__calcer_type }}

**Supported processing units**

{{ cpu-gpu }}

### used_ram_limit

#### Description

{% note alert %}

- This option affects only the CTR calculation memory usage.
- In some cases it is impossible to limit the amount of CPU RAM used in accordance with the specified value.

{% endnote %}

Attempt to limit the amount of used CPU RAM.

Format:

```
<size><measure of information>
```
Supported measures of information (non case-sensitive):
- MB
- KB
- GB

For example:

```
2gb

```

**Possible types**

int

**Default value**

None (memory usage is no limited)

**Supported processing units**

{{ calcer_type__cpu }}




