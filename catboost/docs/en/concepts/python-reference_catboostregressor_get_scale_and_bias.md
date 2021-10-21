# get_scale_and_bias

{% include [get_scale_and_bias-get_scale_and_bias__desc](../_includes/work_src/reusage-python/get_scale_and_bias__desc.md) %}

## {{ dl--invoke-format }} {#method-call-format}

```python
get_scale_and_bias()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [get_scale_and_bias-get_scale_and_bias__return_value_type](../_includes/work_src/reusage-python/get_scale_and_bias__return_value_type.md) %}

## {{ dl--example }} {#examples}

```python
from catboost import CatBoostRegressor, Pool
import numpy as np

train_data = [[1, 4, 5, 6],
              [4, 5, 6, 7],
              [30, 40, 50, 60]]

eval_data = [[2, 4, 6, 8],
             [1, 4, 50, 60]]

train_labels = [10, 20, 30]

model = CatBoostRegressor()

print(model.get_scale_and_bias())

```

The output of this example:

```no-highlight
(1.0, 0.0)
```

