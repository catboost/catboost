# get_gpu_device_count

{% include [utils-utils__get-gpu-device-count__desc](../_includes/work_src/reusage-python/utils__get-gpu-device-count__desc.md) %}


{% note info %}

- The returned value is <q>0</q> if the installed or compiled package does not support training on GPU.
- Use the `CUDA_VISIBLE_DEVICES` environment variable to limit the list of available devices.

{% endnote %}


## {{ dl--invoke-format }} {#call-format}

```python
get_gpu_device_count()
```

## {{ dl__usage-examples }} {#usage-examples}

```python
from catboost.utils import get_gpu_device_count
print('I see %i GPU devices' % get_gpu_device_count())
```

