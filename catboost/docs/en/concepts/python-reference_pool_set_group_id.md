# set_group_id

{% include [set_group_id-set_group_id__desc](../_includes/work_src/reusage-python/set_group_id__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```
set_group_id(group_id)
```

## {{ dl--parameters }} {#parameters}

### group_id

#### Description

Group identifiers for all input objects. Supported identifier types are:
- {{ python-type--int }}
- string types ({{ python-type--string }} or {{ python-type__unicode }} for Python 2 and {{ python-type__bytes }} or {{ python-type--string }} for Python 3).

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value** 

{{ python--required }}
