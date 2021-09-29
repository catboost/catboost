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
from catboost import CatBoostClassifier, Pool
import numpy as np

train_data = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[1, 1, -1],
                  weight=[0.1, 0.2, 0.3])

model = CatBoostClassifier()

print("Default scale and bias: " + str(model.get_scale_and_bias()))
model.set_scale_and_bias(0.5, 0.5)
print("Modified scale and bias: " + str(model.get_scale_and_bias()))

```

The output of this example:

```no-highlight
Default scale and bias: (1.0, 0.0)
Modified scale and bias: (0.5, 0.5)
```

