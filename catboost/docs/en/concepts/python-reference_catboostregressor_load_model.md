# load_model

{% include [sections-with-methods-desc-load_model--purpose](../_includes/work_src/reusage/load_model--purpose.md) %}


## {{ dl--invoke-format }} {#call-format}

```
load_model(fname, format='cbm')
```

## {{ dl--parameters }} {#parameters}

{% include [sections-with-methods-desc-python__load_model__parameter](../_includes/work_src/reusage/python__load_model__parameter.md) %}


## {{ dl__usage-examples }} {#usage-examples}

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

