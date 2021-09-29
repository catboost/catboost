# get_text_feature_indices

{% include [get_text_feature_indices-get_text_features_indices__desc](../_includes/work_src/reusage-python/get_text_features_indices__desc.md) %}


## {{ dl--invoke-format }} {#method-call-format}

```
get_text_feature_indices()
```

## {{ dl--output-format }} {#type-of-return-value}

{{ dl--output-format }}

## {{ dl--example }} {#examples}

```python
from catboost import Pool

data = [[1, 3, "Unthrifty loveliness"],
        [0, 4, "why dost thou spend"],
        [1, 7, "Upon thy self"],
        [6, 4, "thy beauty's legacy"]]

dataset = Pool(data, text_features=[2])

print(dataset.get_text_feature_indices())

```

The output of this example:
```no-highlight
[2]
```

