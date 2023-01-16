
```python
from catboost import Pool

data = [[1, 3],
        [0, 4],
        [1, 7],
        [6, 4],
        [5, 3]]

dataset = Pool(data)
print(dataset.num_row())

dataset_part = dataset.slice([0, 1, 2])
print(dataset_part.num_row())

```
Get a slice of five objects from the input dataset:
