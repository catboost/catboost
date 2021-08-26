# min_unused_token_id

{% include [descriptions-min_unused_token_id](../_includes/work_src/reusage-tokenizer/min_unused_token_id.md) %}

Identifiers are assigned consistently to all input tokens. Some additional identifiers are reserved for internal needs. This method returns the first unused identifier.

All further identifiers are assumed to be unassigned to any token.

## {{ dl--invoke-format }} {#call-format}

```
min_unused_token_id()
```

## {{ dl--output-format }} {#return-value}

{{ python-type--int }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

print(dictionary.min_unused_token_id)

```

Output:
```bash
6
```

