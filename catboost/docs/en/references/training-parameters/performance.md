# Performance settings

## thread_count {#thread_count}

Command-line: `-T`, `--thread-count`

#### Description

The number of threads to use during the training.

{% cut "Python package, Command-line" %}

- **For CPU**

  Optimizes the speed of execution. This parameter doesn't affect results.

- **For GPU**
  The given value is used for reading the data from the hard drive and does not affect the training.

  During the training one main thread and one thread for each GPU are used.

{% endcut %}

{% cut "R package" %}

Optimizes the speed of execution. This parameter doesn't affect results.

{% endcut %}

**Type**

{{ python-type--int }}

**Default value**

{{ fit__thread_count__wrappers }}

**Supported processing units**

{{ cpu-gpu }}

## used_ram_limit {#used_ram_limit}

Command-line: `--used-ram-limit`

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
**Type**

{{ python-type--int }}

**Default value**

{{ fit__used-ram-limit }}

**Supported processing units**

{{ calcer_type__cpu }}

## gpu_ram_part {#gpu_ram_part}

Command-line: `--gpu-ram-part`

#### Description

How much of the GPU RAM to use for training.

**Type**

{{ python-type--float }}

**Default value**

{{ fit__gpu__gpu-ram-part }}

**Supported processing units**

{{ calcer_type__cpu }}

## pinned_memory_size {#pinned_memory_size}

Command-line: `--pinned-memory-size`

#### Description

How much pinned (page-locked) CPU RAM to use per GPU.

The value should be a positive integer or `inf`. Measure of information can be defined for integer values.

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
**Type**

{{ python-type--int }}

**Default value**

{{ fit__gpu__pinned-memory-size }}

**Supported processing units**

{{ calcer_type__cpu }}

## gpu_cat_features_storage {#gpu_cat_features_storage}

Command-line: `--gpu-cat-features-storage`

#### Description

The method for storing the categorical features' values.

Possible values:
- {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }}
- {{ fit__gpu__gpu_cat_features_storage__value__GpuRam }}

{% note info %}

Use the {{ fit__gpu__gpu_cat_features_storage__value__CpuPinnedMemory }} value if feature combinations are used and the available GPU RAM is not sufficient.

{% endnote %}

**Type**

{{ python-type--string }}

**Default value**

{% cut "Python package" %}

None (set to {{ fit__gpu__use-cpu-ram-for-catfeatures }})

{% endcut %}

{% cut "Command-line" %}

{{ fit__gpu__use-cpu-ram-for-catfeatures }}

{% endcut %}

**Supported processing units**

{{ calcer_type__cpu }}

## data_partition {#data_partition}

Command-line: `--data-partition`

#### Description

The method for splitting the input dataset between multiple workers.

Possible values:
- {{ fit__gpu__data-partition__mode__FeatureParallel }} — Split the input dataset by features and calculate the value of each of these features on a certain GPU.

  For example:

    - GPU0 is used to calculate the values of features indexed 0, 1, 2
    - GPU1 is used to calculate the values of features indexed 3, 4, 5, etc.

- {{ fit__gpu__data-partition__mode__DocParallel }} — Split the input dataset by objects and calculate all features for each of these objects on a certain GPU. It is recommended to use powers of two as the value for optimal performance.

  For example:
    - GPU0 is used to calculate all features for objects indexed `object_1`, `object_2`
    - GPU1 is used to calculate all features for objects indexed `object_3`, `object_4`, etc.

**Type**

{{ python-type--string }}

**Default value**

{{ fit__gpu__data-partition }}

**Supported processing units**

{{ calcer_type__cpu }}
