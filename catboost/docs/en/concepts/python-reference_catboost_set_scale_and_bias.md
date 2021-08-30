# set_scale_and_bias

{% include [set_scale_and_bias-set_scale_and_bias__desc](../_includes/work_src/reusage-python/set_scale_and_bias__desc.md) %}

## {{ dl--invoke-format }} {#method-call-format}

```
set_scale_and_bias(scale, bias)
```

## {{ dl--parameters }} {#parameters}

{% include [set_scale_and_bias-set_scale_and_bias__params-table](../_includes/work_src/reusage-python/set_scale_and_bias__params-table.md) %}


## {{ dl--example }} {#examples}

```python
from catboost import CatBoost
import numpy as np

train_data = np.random.randint(1, 100, size=(100, 10))
train_labels = np.random.randint(2, size=(100))

model = CatBoost()

print("Default scale and bias: " + str(model.get_scale_and_bias()))
model.set_scale_and_bias(0.5, 0.5)
print("Modified scale and bias: " + str(model.get_scale_and_bias()))

```

The output of this example:

```
Default scale and bias: (1.0, 0.0)
Modified scale and bias: (0.5, 0.5)
```

