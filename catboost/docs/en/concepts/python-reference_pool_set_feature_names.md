# set_feature_names

{% include [set_feature_names-set_feature_names__desc](../_includes/work_src/reusage-python/set_feature_names__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
set_feature_names(feature_names)
```

## {{ dl--parameters }} {#parameters}

### feature_names

#### Description

A list of names for each feature in the dataset.

**Possible types**

{{ python-type--list-of-strings }}

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

input_pool.set_feature_names(['year', 'name', 'BLBRD', 'CAC', 'OAC'])
```
