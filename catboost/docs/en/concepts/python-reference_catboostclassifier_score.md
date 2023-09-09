# score

{% include [catboost-classifier-score-score__purpose](../_includes/work_src/reusage-python/score__purpose.md) %}


## {{ dl--invoke-format }} {#call-format}

```
score(X, y)
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

{% include [methods-param-desc-label--short-desc1](../_includes/work_src/reusage/label--short-desc1.md) %}


{% include [methods-param-desc-label--short-desc2](../_includes/work_src/reusage/label--short-desc2.md) %}


{% note info %}

Do not use this parameter if the input training dataset (specified in the `X` parameter) type is {{ python-type--pool }}.

{% endnote %}

**Possible types**

- {{ python-type--list }}
- {{ python-type--numpy-ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

None


## {{ dl--output-format }} {#output-format}

{{ python-type--float }}
