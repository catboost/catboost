# get_group_id

{% include [get_group_id-get_group_id__desc](../_includes/work_src/reusage-python/get_group_id__desc.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
get_group_id()
```

## {{ dl--output-format }} {#output-format}

{{ python-type--list }}

## {{ input_data__title__example }} {#example}

```python
from catboost import Pool

train_data = [[75, 'today', 5, 4, 3],
              [76, 'blvd', 4, 5, 7],
              [70, 'letters', 0, 3, 4],
              [60, 'back', 9, 0, 1]]

label_values = [0, 1, 0, 4]

input_pool = Pool(data=train_data,
                  label=label_values,
                  cat_features=[1],
                  group_id=["CA", "TN", "TN", "TN"])


print(input_pool.get_group_id())

```

Output:
```no-highlight
[13555145920995255203L, 14627267897973961738L, 14627267897973961738L, 14627267897973961738L]
```

