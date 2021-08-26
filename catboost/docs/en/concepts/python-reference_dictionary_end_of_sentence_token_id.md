# end_of_sentence_token_id

{% include [descriptions-end_of_sentence_token_id](../_includes/work_src/reusage-tokenizer/end_of_sentence_token_id.md) %}

{{ dl--invoke-format }}

```
end_of_sentence_token_id()
```

## {{ dl--output-format }} {#return-value}

{{ python-type--int }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

print(dictionary.end_of_sentence_token_id)

```

Output:
```bash
5
```

