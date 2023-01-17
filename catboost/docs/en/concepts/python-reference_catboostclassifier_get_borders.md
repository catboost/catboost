# get_borders

{% include [get_borders-get_borders__desc__div](../_includes/work_src/reusage-python/get_borders__desc__div.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```python
get_borders()
```

## {{ output--example }} {#usage-example}

```python
from catboost import CatBoostClassifier

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]
train_labels = [0, 0, 1, 1]

model = CatBoostClassifier(loss_function='Logloss')
model.fit(train_data, train_labels, verbose=False)

print model.get_borders()

```

```python
{0: [2.0, 6.0, 8.5], 1: [2.0]}
```

