# set_baseline

{% include [set_baseline-set_baseline__desc](../_includes/work_src/reusage-python/set_baseline__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_baseline(baseline)
```

## {{ dl--parameters }} {#parameters}

### baseline

#### Description

{% include [reusage-input-data-baseline__shortdesc](../_includes/work_src/reusage-input-data/baseline__shortdesc.md) %}


**Possible types**

- {{ python-type--list }}
- {{ python-type__np_ndarray }}
- {{ python-type--pandasDataFrame }}
- {{ python-type--pandasSeries }}

**Default value** 

{{ python--required }}

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

input_pool = Pool(data = train_data,
                  label = label_values,
                  cat_features = [1])

input_pool.set_baseline([1, 3, 2, 1, 2])

input_pool.get_baseline()
```
