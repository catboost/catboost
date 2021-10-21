# save

{% include [pool_save-python__pool__save__desc__div](../_includes/work_src/reusage-python/python__pool__save__desc__div.md) %}


## {{ dl--invoke-format }} {#call-format}

```python
save(fname)
```

## {{ dl--parameters }} {#parameters}
### fname

#### Description

The name of the output file to save the pool to.

**Possible types** 

{{ python-type--string }}

**Default value**  

Required parameter

## {{ input_data__title__example }} {#example}

[Quantize](python-reference_pool_quantized.md) the given dataset and save it to a file:

```python
import numpy as np
from catboost import Pool, CatBoostRegressor


train_data = np.random.randint(1, 100, size=(10000, 10))
train_labels = np.random.randint(2, size=(10000))
quantized_dataset_path = 'quantized_dataset.bin'

# save quantized dataset
train_dataset = Pool(train_data, train_labels)
train_dataset.quantize()
train_dataset.save(quantized_dataset_path)

```
