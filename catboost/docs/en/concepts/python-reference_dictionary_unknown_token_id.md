# unknown_token_id

{% include [descriptions-unknown_token_id](../_includes/work_src/reusage-tokenizer/unknown_token_id.md) %}

## {{ dl--invoke-format }} {#call-format}

```
unknown_token_id()
```

## {{ dl--output-format }} {#return-value}

{{ python-type--int }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

print(dictionary.unknown_token_id)
print(dictionary.get_token(dictionary.unknown_token_id))
```

Output:
```bash
4
```

