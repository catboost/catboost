# get_tokens

{% include [descriptions-get_tokens](../_includes/work_src/reusage-tokenizer/get_tokens.md) %}


## {{ dl--invoke-format }} {#call-format}

```
get_tokens(token_id)
```

## {{ dl--parameters }} {#parameters}

### token_id

#### Description

A list of token identifiers that should be returned.

**Data types**

{{ python-type--list }}

**Default value**

{{ loss-functions__params__q__default }}

## {{ dl--output-format }} {#return-value}

{{ python-type--list }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

print(dictionary.get_tokens([1,3]))

```

Output:
```bash
['his', 'whatever']
```

