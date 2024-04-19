# set_subgroup_id

{% include [set_subgroup_id-set_subgroup_identifiers__desc](../_includes/work_src/reusage-python/set_subgroup_identifiers__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_subgroup_id(subgroup_id)
```

## {{ dl--parameters }} {#parameters}

### subgroup_id

#### Description

Subgroup identifiers for all input objects. Supported identifier types are:
- {{ python-type--int }}
- string types ({{ python-type--string }} or {{ python-type__unicode }} for Python 2 and {{ python-type__bytes }} or {{ python-type--string }} for Python 3).

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value**

{{ python--required }}
