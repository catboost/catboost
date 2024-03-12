# score

Calculate the NDCG@top [metric](../concepts/loss-functions.md) for the objects in the given dataset.


## {{ dl--invoke-format }} {#call-format}

```
score(X,
      y=None,
      group_id=None,
      top=None,
      type=None,
      denominator=None,
      group_weight=None,
      thread_count=-1)
```

## {{ dl--parameters }} {#parameters}

### X

#### Description

The description is different for each group of possible types.

**Possible types**

{% cut "{{ python-type--pool }}" %}

The input training dataset.

{% note info %}

If a nontrivial value of the `cat_features` parameter is specified in the constructor of this class, {{ product }} checks the equivalence of categorical features indices specification from the constructor parameters and in this Pool class.

{% endnote %}

{% endcut %}

{% cut "{{ python-type--list }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}, {{ python-type--pandasSeries }}" %}

The input training dataset in the form of a two-dimensional feature matrix.

{% endcut %}

{% cut "{{ python_type__pandas-SparseDataFrame }}, {{ python_type__scipy-sparse-spmatrix }} (all subclasses except dia_matrix)" %}

{% include [libsvm-libsvm__desc](../_includes/work_src/reusage-formats/libsvm__desc.md) %}

{% endcut %}

**Default value**

{{ python--required }}


### y

#### Description

{% include [methods-param-desc-label--short-desc-evaluation](../_includes/work_src/reusage/label--short-desc-evaluation.md) %}

{% include [methods-param-desc-label--detailed-desc-ranking](../_includes/work_src/reusage/label--detailed-desc-ranking.md) %}

{% note info %}

Do not use this parameter if the input training dataset (specified in the `X` parameter) type is {{ python-type--pool }}.

{% endnote %}

{% include [methods-param-desc-label--possible-types-1d-default-supported-processing-units](../_includes/work_src/reusage/label--possible-types-1d-default-supported-processing-units.md) %}

### group_id

#### Description

A ranking group.

{% note info %}

Do not use this parameter if the input training dataset (specified in the `X` parameter) type is {{ python-type--pool }}.

{% endnote %}

**Possible types**

- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None

### top

#### Description

 NDCG, Number of top-ranked objects to calculate NDCG

**Possible types**

- unsigned integer, up to `pow(2, 32) / 2 - 1`

**Default value**

None

### type

#### Description

Metric type: Base or Exp.

**Possible types**

- str

**Default value**

None

### denominator

#### Description

Denominator type.

**Possible types**

- str

**Default value**

None

### group_weight

#### Description

The weights of all objects within the defined groups from the input data in the form of one-dimensional array-like data.
Used for calculating the final values of trees. By default, it is set to one for all objects in all groups.
Only a `weight` or `group_weight` parameter can be used at the time.

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None

### thread_count

#### Description

The number of threads to use.

Optimizes the speed of execution. This parameter doesn't affect results.

**Possible types**

- int

**Default value**

-1 (the number of threads is equal to the number of processor cores)

## {{ dl--output-format }} {#output-format}
{{ python-type--float }}
