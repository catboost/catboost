# set_timestamp

{% include [set_timestamp-set_timestamp__desc](../_includes/work_src/reusage-python/set_timestamp__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_timestamp(timestamp)
```

## {{ dl--parameters }} {#parameters}

### timestamp

#### Description

Timestamps for all input objects.
Should contain non-negative integer values.
Useful for sorting a learning dataset by this field during training.

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value** 

{{ python--required }}
