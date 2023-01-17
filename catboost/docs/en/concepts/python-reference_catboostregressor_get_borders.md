# get_borders

{% include [get_borders-get_borders__desc__div](../_includes/work_src/reusage-python/get_borders__desc__div.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```python
get_borders()
```

## {{ output--example }} {#usage-example}

```python
from catboost import CatBoostRegressor

train_data = [[1, 4, 5, 6],
              [4, 5, 6, 7],
              [30, 40, 50, 60]]

eval_data = [[2, 4, 6, 8],
             [1, 4, 50, 60]]

train_labels = [10, 20, 30]

model = CatBoostRegressor()

model.fit(train_data,
          train_labels,
          verbose=False)

print(model.get_borders())

```

Output:
```no-highlight
{0: [2.5, 17.0], 1: [4.5, 22.5], 2: [5.5, 28.0], 3: [6.5, 33.5]}
```

