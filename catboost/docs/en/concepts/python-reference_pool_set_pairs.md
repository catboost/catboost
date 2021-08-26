# set_pairs

{% include [set_pairs-set_pairs__desc](../_includes/work_src/reusage-python/set_pairs__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_pairs(pairs)
```

## {{ dl--parameters }} {#parameters}

### pairs

#### Description

{% include [methods-param-desc-python__pairs__intro](../_includes/work_src/reusage/python__pairs__intro.md) %}

- `N` is the number of pairs.
- Each pair is represented as a row in the matrix:
    - The first element in the row (`pair[0]`) contains the zero-based index of the winner object.
    - The second element in the row (`pair[1]`) contains the zero-based index of the loser object.

This information is used for optimization and calculation of [Pairwise metrics](loss-functions-ranking.md).

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasDataFrame }}

**Default value** 

{{ python--required }}
