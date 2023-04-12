# quantize

{% include [quantized-python_quantized](../reusage-python/python_quantized.md) %}

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


**Possible types**

{{ python-type--list }}

#### Description


{% include [reusage-ignored-feature__common-div](ignored-feature__common-div.md) %}


For example, use the following construction if features indexed 1, 2, 7, 42, 43, 44, 45, should be ignored:
```
[1,2,7,42,43,44,45]
```


**Default value**

None

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### per_float_feature_quantization


**Possible types**

{{ python-type--list-of-strings }}

#### Description


The quantization description for the given list of features (one or more).

{% include [python-python__per_floa_feature_quantization__description-format__intro-and-format](../../_includes/work_src/reusage/python__per_floa_feature_quantization__description-format__intro-and-format.md) %}


Examples:
- ```
    per_float_feature_quantization=['0:1024']
    ```

    {% include [python-feature_0_has_1024_borders](feature_0_has_1024_borders.md) %}

    The following example is equivalent to the one given above:

    ```
    per_float_feature_quantization=['0:border_count=1024']
    ```

- ```
    per_float_feature_quantization=['0:border_count=1024','1:border_count=1024']
    ```

    {% include [python-feature_0-1_have_1024_borders](feature_0-1_have_1024_borders.md) %}


**Default value**

None

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### border_count


_Alias:_`max_bin`

**Possible types**

{{ python-type--int }}

#### Description


The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.


**Default value**


{% include [reusage-default-values-border_count](../reusage-default-values/border_count.md) %}



**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### feature_border_type


**Possible types**

{{ python-type--string }}

#### Description


The [quantization mode](../concepts/quantization.md) for numerical features.

Possible values:
- Median
- Uniform
- UniformAndQuantiles
- MaxLogSum
- MinEntropy
- GreedyLogSum


**Default value**

{{ fit--feature_border_type }}

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### dev_efb_max_buckets


**Possible types**

{{ python-type--int }}

#### Description


Maximum bucket count in exclusive features bundle.

Possible values are in the range [0, 65536]

exclusive features bundle -- это такой набор факторов, что в каждой строчке датасета только один из них не равен значению по умолчанию.во время загрузки датасета Катбуст автоматически находит такие наборы факторов и заменяет каждый набор на один новый фактор.dev_efb_max_buckets задает максимальное число бинов для этих новых факторов.\u0000


**Default value**

None

**Supported processing units**

{{ calcer_type__cpu }}

### nan_mode


**Possible types**

{{ python-type--string }}

#### Description


{% include [python-python__nan-mode_preprocessing__intro](python__nan-mode_preprocessing__intro.md) %}


{% include [reusage-cmd__nan-mode__list-only__div](cmd__nan-mode__list-only__div.md) %}



**Default value**

{{ fit--nan_mode }}

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### input_borders


**Possible types**

{{ python-type--string }}

#### Description


{% include [reusage-cli__input-borders-file__desc__div](cli__input-borders-file__desc__div.md) %}



**Default value**

{{ cli__input-borders-file__default }}

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### simple_ctr


**Possible types**

{{ python-type--string }}

#### Description


{% include [reusage-cli__simple-ctr__intro](cli__simple-ctr__intro.md) %}


Format:

```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__simple-ctr__components](cli__simple-ctr__components.md) %}


{% include [reusage-cli__simple-ctr__examples__p](cli__simple-ctr__examples__p.md) %}



**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### combinations_ctr


**Possible types**

{{ python-type--string }}

#### Description


{% include [reusage-cli__combination-ctr__intro](cli__combination-ctr__intro.md) %}


```
['CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'CtrType[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__combination-ctr__components](cli__combination-ctr__components.md) %}



**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### per_feature_ctr


**Possible types**

{{ python-type--string }}

#### Description


{% include [reusage-cli__per-feature-ctr__intro](cli__per-feature-ctr__intro.md) %}


```
['FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
 'FeatureId:CtrType:[:{{ ctr-types__TargetBorderCount }}=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num_1/denum_1]..[:Prior=num_N/denum_N]',
  ...]
```

{% include [reusage-cli__per-feature-ctr__components](cli__per-feature-ctr__components.md) %}



**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### ctr_target_border_count


**Possible types**

{{ python-type--int }}

#### Description


{% include [reusage-cli__ctr-target-border-count__short-desc](cli__ctr-target-border-count__short-desc.md) %}


The value of the `{{ ctr-types__TargetBorderCount }}` component overrides this parameter if it is specified for one of the following parameters:

- `simple_ctr`
- `combinations_ctr`
- `per_feature_ctr`


**Default value**

{{ parameters__ctr-target-border-count__default }}

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### task_type


**Possible types**

{{ python-type--string }}

#### Description


The processing unit type to use for training.

Possible values:
- CPU
- GPU


**Default value**

{{ fit__python-r__calcer_type }}

**Supported processing units**


{% include [reusage-python-cpu-and-gpu](../reusage-python/cpu-and-gpu.md) %}



### used_ram_limit


**Possible types**

{{ python-type--int }}

#### Description


Attempt to limit the amount of used CPU RAM.

{% note alert %}

- This option affects only the CTR calculation memory usage.
- In some cases it is impossible to limit the amount of CPU RAM used in accordance with the specified value.

{% endnote %}


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


**Default value**

{{ fit__used-ram-limit }}

**Supported processing units**


{% include [reusage-python-cpu](../reusage-python/cpu.md) %}



## {{ input_data__title__example }} {#example}

Quantize the given dataset and [save](python-reference_pool_save.md) it to a file:

```python
import numpy as np
from catboost import Pool, CatBoostRegressor


train_data = np.random.randint(1, 100, size=(10000, 10))
train_labels = np.random.randint(2, size=(10000))
quantized_dataset_path = 'quantized_dataset.bin'

# save quantized dataset
train_dataset = Pool(train_data, train_labels)
train_dataset.quantize()
train_dataset.save(quantized_dataset_path)

```
