
Create a pool from a file and quantize it while loading the data. This compresses the size of the initial dataset and provides an opportunity to load huge datasets that can not be loaded to RAM otherwise.
{% note info %}

The input data should contain only numerical features (other types are not currently supported).

{% endnote %}

This method gives an identical result to implementing the following code but is less RAM consuming:
```python
pool = Pool(filename, **some_pool_load_params)
pool.quantize(**some_quantization_params)
return pool
```
