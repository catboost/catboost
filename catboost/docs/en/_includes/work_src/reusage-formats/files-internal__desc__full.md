
Format:
```
[scheme://]<path>
```

- `scheme` (optional) defines the type of the input dataset. Possible values:

    - `{{ input_data__scheme__quantized }}` — {{ python-type--pool }} [quantized](../../../concepts/python-reference_pool_quantized.md) pool.
    - `{{ input_data__scheme__libsvm }}` — dataset in the [extended libsvm format](../../../concepts/input-data_libsvm.md).

    If omitted, a dataset in the [Native {{ product }} Delimiter-separated values format](../../../concepts/input-data_values-file.md) is expected.

- `path` defines the path to the dataset description.
