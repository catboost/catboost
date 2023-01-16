
```python
from catboost import CatBoostClassifier, Pool

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]

train_labels = [0, 0, 1, 1]

eval_data = [[2, 1],
             [3, 1],
             [9, 0],
             [5, 3]]

eval_labels = [0, 1, 1, 0]

eval_dataset = Pool(eval_data,
                    eval_labels)

model = CatBoostClassifier(learning_rate=0.03,
                           eval_metric='AUC')

model.fit(train_data,
          train_labels,
          eval_set=eval_dataset,
          verbose=False)

print(model.get_best_iteration())

```
