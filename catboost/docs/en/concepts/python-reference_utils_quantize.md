# quantize

{% include [utils-pool__quantize__div](../_includes/work_src/reusage-python/pool__quantize__div.md) %}

## {{ dl--invoke-format }} {#call-format}

```python
quantize(data_path,
         column_description=None,
         pairs=None,
         delimiter='\t',
         has_header=False,
         feature_names=None,
         thread_count=-1,
         ignored_features=None,
         per_float_feature_quantization=None,
         border_count=None,
         max_bin=None,
         feature_border_type=None,
         nan_mode=None,
         input_borders=None,
         task_type=None,
         used_ram_limit=None,
         random_seed=None)
```

## {{ dl--parameters }} {#parameters}

### data_path

#### Description

The path to the input file{% if audience == "internal" %} or table{% endif %} that contains the dataset description.

{% include [files-internal-files-internal__desc__full](../_includes/work_src/reusage-formats/files-internal__desc__full.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{{ loss-functions__params__q__default }}

### column_description

#### Description

The path to the input file {% if audience == "internal" %}or table{% endif %} that contains the [columns description](../concepts/input-data_column-descfile.md).

{% if audience == "internal" %}

{% include [internal__cd-internal-cd-desc](../yandex_specific/_includes/reusage-formats/internal-cd-desc.md) %}

{% endif %}

**Possible types**

{{ python-type--string }}

**Default value**

None

### pairs

#### Description

The path to the input file that contains the [pairs description](../concepts/input-data_pairs-description.md).

{% include [reusage-learn_pairs__where_is_used](../_includes/work_src/reusage/learn_pairs__where_is_used.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{% include [methods-param-desc-python__pairs__default-short](../_includes/work_src/reusage/python__pairs__default-short.md) %}

### delimiter

#### Description

The delimiter character used to separate the data in the dataset description input file.

Only single char delimiters are supported. If the specified value contains more than one character, only the first one is used.

{% include [libsvm-note-restriction-delimiter-separated-format](../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}

**Possible types**

{{ fit__delimiter }}

**Default values**

{{ cpu-gpu }}

### has_header

#### Description

Read the column names from the first line of the dataset description file if this parameter is set.

{% include [libsvm-note-restriction-delimiter-separated-format](../_includes/work_src/reusage-formats/note-restriction-delimiter-separated-format.md) %}

**Possible types**

{{ python-type--bool }}

**Default value**

False

### feature_names

#### Description

A list of names for each feature in the dataset.

**Possible types**

{{ python-type--list }}

**Default value**

None

### thread_count

#### Description

The number of threads to use.

{% include [reusage-thread_count__cpu_cores__optimizes-the-speed-of-execution](../_includes/work_src/reusage/thread_count__cpu_cores__optimizes-the-speed-of-execution.md) %}

**Possible types**

{{ python-type--int }}

**Default values**

{{ fit__thread_count__wrappers }}

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

### per_float_feature_quantization

#### Description

{% include [python-python__per-float-feature-quantization__desc-without-examples](../_includes/work_src/reusage/python__per-float-feature-quantization__desc-without-examples.md) %}


Example:

{% include [python-python__per_float_feature_quantization__string_example](../_includes/work_src/reusage/python__per_float_feature_quantization__string_example.md) %}


**Possible types**

{{ python-type--list-of-strings }}

**Default value**

{{ fit--feature_border_type }}

### border_count

_Alias:_`max_bin`

#### Description

The number of splits for numerical features. Allowed values are integers from 1 to 65535 inclusively.

**Possible types**

{{ python-type--int }}

**Default value**

The default value depends on the processing unit type:
- {{ calcer_type__cpu }}: 254
- {{ calcer_type__gpu }}: 128


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

### nan_mode

#### Description

The method for  [processing missing values](../concepts/algorithm-missing-values-processing.md) in the input dataset.

{% include [reusage-cmd__nan-mode__div](../_includes/work_src/reusage/cmd__nan-mode__div.md) %}

**Possible types**

{{ python-type--string }}

**Default value**

{{ fit--nan_mode }}

### input_borders

#### Description

Load [Custom quantization borders and missing value modes](../concepts/input-data_custom-borders.md) from a file (do not generate them).

Borders are automatically generated before training if this parameter is not set.

**Possible types**

{{ python-type--string }}

**Default value**

None

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

### used_ram_limit

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
**Possible types**

{{ python-type--int }}

**Default value**

{{ fit__used-ram-limit }}

### random_seed

#### Description

The random seed used for training.

**Possible types**

{{ python-type--int }}

**Default value**

None ({{ fit--random_seed }})


## {{ dl--output-format }} {#output-format}

{{ python-type--pool }} (a quantized pool)

## {{ dl__usage-examples }} {#usage-examples}

The following is the input file with the dataset description:

```
4	52	64	73
3	87	32	54
9	34	35	45
8	9	83	32
```

The pool is created as follows:

```python
from catboost.utils import quantize

quantized_pool=quantize(data_path="pool__utils__quantize_data")
print(type(quantized_pool))
```

The output of this example:
```
<class 'catboost.core.Pool'>
```
