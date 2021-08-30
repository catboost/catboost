# get_scale_and_bias

{% include [get_scale_and_bias-get_scale_and_bias__desc](../_includes/work_src/reusage-python/get_scale_and_bias__desc.md) %}

## {{ dl--invoke-format }}

```python
get_scale_and_bias()
```

## {{ dl--output-format }}

{% include [get_scale_and_bias-get_scale_and_bias__return_value_type](../_includes/work_src/reusage-python/get_scale_and_bias__return_value_type.md) %}


## {{ dl--example }}

```python
from catboost import CatBoostClassifier, Pool
import numpy as np

train_data = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[1, 1, -1],
                  weight=[0.1, 0.2, 0.3])

model = CatBoostClassifier()

print(model.get_scale_and_bias())

```

The output of this example:

```no-highlight
(1.0, 0.0)
```

