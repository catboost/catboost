# set_label

{% include [set_label-set_label__desc](../_includes/work_src/reusage-python/set_label__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_label(label)
```

## {{ dl--parameters }} {#parameters}

### label

#### Description

A one-dimensional array-like of numeric label values. The length must match the number of objects in the dataset (`Pool.num_row()`). Only numeric dtypes are accepted -- reconstruct the Pool for string or categorical targets.

**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasSeries }}
- Single-column {{ python-type--pandasDataFrame }}
- [polars.Series](https://docs.pola.rs/api/python/stable/reference/series/index.html)
- Single-column [polars.DataFrame](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html)

**Default value**

{{ python--required }}

## {{ dl--output-format }} {#return-value}

{{ python-type--pool }} -- returns `self` for chaining.

## {{ input_data__title__example }} {#example}

```python
import numpy as np
from catboost import Pool

train_data = [[76, 'blvd', 41, 50, 7],
              [75, 'today', 57, 0, 48],
              [70, 'letters', 33, 17, 7],
              [72, 'now', 43, 29, 12],
              [60, 'back', 2, 0, 1]]

label_values = [1, 0, 0, 1, 4]

input_pool = Pool(data=train_data,
                  label=label_values,
                  cat_features=[1])

new_labels = [0, 1, 1, 0, 2]
input_pool.set_label(new_labels)

input_pool.get_label()
```

## See also

- [Pool.set_weight](python-reference_pool_set_weight.md)
- [fit()](python-reference_catboost_fit.md)
