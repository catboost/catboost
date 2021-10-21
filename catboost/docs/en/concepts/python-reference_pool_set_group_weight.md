# set_group_weight

{% include [set_group_weight-set_group_weight__desc](../_includes/work_src/reusage-python/set_group_weight__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_group_weight(group_weight)
```

## {{ dl--parameters }} {#parameters}

### group_weight

#### Description

{% include [methods-param-desc-python__group_weight__first-sentence](../_includes/work_src/reusage/python__group_weight__first-sentence.md) %}

The length of this array must be equal to the number of objects in the dataset.

All weights within one group must be equal.

{% include [pool-only-non-negative-values-are-supported](../_includes/work_src/reusage-python/only-non-negative-values-are-supported.md) %}

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

{{ python--required }}
