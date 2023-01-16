# get_scale_and_bias

{% include [get_scale_and_bias-get_scale_and_bias__desc](../_includes/work_src/reusage-python/get_scale_and_bias__desc.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```python
get_scale_and_bias()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [get_scale_and_bias-get_scale_and_bias__return_value_type](../_includes/work_src/reusage-python/get_scale_and_bias__return_value_type.md) %}


```python
from catboost import CatBoost
import numpy as np

train_data = np.random.randint(1, 100, size=(100, 10))
train_labels = np.random.randint(2, size=(100))

model = CatBoost()

print(model.get_scale_and_bias())

```

The output of this example:

```no-highlight
(1.0, 0.0)
```

