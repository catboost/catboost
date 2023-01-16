# get_metadata

{% include [sections-with-methods-desc-python__method__metadata__desc__div](../_includes/work_src/reusage/python__method__metadata__desc__div.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
get_metadata()
```

## {{ dl--output-format }} {#output-format}

{% include [sections-with-methods-desc-python__method__metadata__desc__type-of-return-value__p](../_includes/work_src/reusage/python__method__metadata__desc__type-of-return-value__p.md) %}


## {{ dl--example }} {#examples}

```python
import numpy as np
from catboost import Pool, CatBoostRegressor
# initialize data
train_data = np.random.randint(0, 100, size=(100, 10))
train_label = np.random.randint(0, 1000, size=(100))
# initialize Pool
train_pool = Pool(train_data, train_label)

# specify the training parameters 
model = CatBoostRegressor()
#train the model
model.fit(train_pool)

# get proxy reference for convenience
metadata = model.get_metadata()
# set some metadata key
metadata['example_key'] = 'example_value'

# check if key is in metadata
print('needed_key' in metadata)

# iterate metadata keys
for i in metadata:  # for i in meta.keys() would also work
    print(i)

# iterate metadata key-values
for i in metadata.items():
    print(i)

# delete some key
del metadata['example_key']

# copy metadata to python dictionary
metadata_copy = dict(metadata)
```

