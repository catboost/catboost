# epsilon

{% include [datasets-datasets__epsilon](../_includes/work_src/reusage-python/datasets__epsilon.md) %}


This dataset is best suited for binary classification.

The training dataset contains 400000 objects. Each object is described by 2001 columns. The first column contains the label value, all other columns contain numerical features.

The validation dataset contains 100000 objects. The structure is identical to the training dataset.

## {{ dl--invoke-format }} {#method-call}

```python
epsilon()
```

## {{ dl--output-format }} {#type-of-return-value}

{% include [datasets-datasets__output](../_includes/work_src/reusage-python/datasets__output.md) %}


## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.datasets import epsilon
epsilon_train, epsilon_test = epsilon()
```

