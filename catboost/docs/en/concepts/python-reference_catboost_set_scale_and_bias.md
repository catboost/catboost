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

train_data = [[0], [1], [2], [3]]
train_labels = [1.0, 2.0, 3.0, 4.0]

model = CatBoost({
    "iterations": 1,
    "loss_function": "RMSE",
    "boost_from_average": False,
    "verbose": False,
})
model.fit(train_data, train_labels)

print("Default scale and bias: " + str(model.get_scale_and_bias()))
model.set_scale_and_bias(0.5, 0.5)
print("Modified scale and bias: " + str(model.get_scale_and_bias()))

```

The output of this example:

```
Default scale and bias: (1.0, 0.0)
Modified scale and bias: (0.5, 0.5)
```

