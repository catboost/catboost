# size

{% include [descriptions-size](../_includes/work_src/reusage-tokenizer/size.md) %}

## {{ dl--invoke-format }} {#call-format}

```
size()
```

## {{ dl--output-format }} {#return-value}

{{ python-type--int }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

print(dictionary.size)
```

Output:
```bash
4
```

