# Multiclassification settings

## classes_count {#classes_count}

Command-line: `--classes-count`

#### Description

{% include [reusage-classes-count__main-desc](../../_includes/work_src/reusage/classes-count__main-desc.md) %}

{% include [reusage-classes-count__possible-values](../../_includes/work_src/reusage/classes-count__possible-values.md) %}


If this parameter is specified the labels for all classes in the input dataset should be smaller than the given value

**Type**

 {{ python-type--int }}

**Default value**

{% cut "Python package" %}

None.

{{ fit--classes-count }}

{% endcut %}

{% cut "R package" %}

maximum class label + 1

{% endcut %}

{% cut "Command line" %}

- `maximum class label + 1` if the `--class-names` parameter is not specified
- the quantity of classes names if the `--class-names` parameter is specified

{% endcut %}

**Supported processing units**

{{ cpu-gpu }}


## --class-names {#--class-names}

This parameter is only for Command-line.

#### Description

Classes names. Allows to redefine the default values when using the {{ error-function--MultiClass }} and {{ error-function--Logit }} metrics.

If the upper limit for the numeric class label is specified, the number of classes names should match this value.

{% note warning %}

The quantity of classes names must match the quantity of classes weights specified in the `--class-weights` parameter and the number of classes specified in the `--classes-count` parameter.

{% endnote %}

Format:

```
<name for class 1>,..,<name for class N>
```

For example:

```
smartphone,touchphone,tablet
```

**Default value**

{{ fit--class-names }}

**Supported processing units**

{{ cpu-gpu }}

