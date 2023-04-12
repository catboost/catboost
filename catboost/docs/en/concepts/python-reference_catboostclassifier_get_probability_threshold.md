# get_probability_threshold

Return threshold for class separation in binary classification task for a trained model.

## {{ dl--invoke-format }} {#method-call-format}

```python
get_probability_threshold()
```


## {{ output--example }} {#example}

The following example trains a simple binary classification model and then shows, how setting probability threshold affects predicted labels.


```python
from catboost import CatBoostClassifier

train_data = [[0, 3],
              [4, 1],
              [8, 1],
              [9, 1]]
train_labels = [0, 0, 1, 1]

model = CatBoostClassifier(loss_function='Logloss')
model.fit(train_data, train_labels, verbose=False)

print("Original classification threshold:")
print(model.get_probability_threshold())
print("Predicted 1-st class probability:")
print(model.predict(train_data, prediction_type='Probability')[:, 1])
print("Predictions with 0.5 threshold:")
print(model.predict(train_data))
model.set_probability_threshold(0.275)
print("Predictions after setting 0.275 threshold:")
print(model.predict(train_data))

```

Output:

```
Original classification threshold:
0.5
Predicted 1-st class probability:
[0.27410171 0.29538924 0.73013974 0.7409906 ]
Predictions with 0.5 threshold:
[0 0 1 1]
Predictions after setting 0.275 threshold:
[0 1 1 1]
```

