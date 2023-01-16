
### Load the model from a file {#load-from-file}

The following example illustrates how to save a trained model to a file and then load it.

```python
from catboost import CatBoostClassifier, Pool

train_data = [[1, 3],
              [0, 4],
              [1, 7]]
train_labels = [1, 0, 1]

# catboost_pool = Pool(train_data, train_labels)

model = CatBoostClassifier(learning_rate=0.03)
model.fit(train_data,
          train_labels,
          verbose=False)

model.save_model("model")

from_file = CatBoostClassifier()

from_file.load_model("model")

```

