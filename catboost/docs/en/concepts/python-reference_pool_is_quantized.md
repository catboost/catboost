# is_quantized

{% include [pool__is_quantized-python__pool__is_quantized__desc__div](../_includes/work_src/reusage-python/python__pool__is_quantized__desc__div.md) %}

## {{ dl--invoke-format }} {#call-format}

```python
is_quantized()
```

## {{ dl__usage-examples }} {#examples}

Create a pool, check whether it is quantized, [quantize](python-reference_pool_quantized.md) it and check whether it is quantized once again.

```python
import numpy as np
from catboost import Pool


train_data = np.random.randint(1, 100, size=(10000, 10))
train_labels = np.random.randint(2, size=(10000))
quantized_dataset_path = 'quantized_dataset.bin'

train_dataset = Pool(train_data, train_labels)
print(train_dataset.is_quantized())

train_dataset.quantize()
print(train_dataset.is_quantized())

```

