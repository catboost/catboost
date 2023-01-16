# get_token

{% include [descriptions-get_token](../_includes/work_src/reusage-tokenizer/get_token.md) %}


## {{ dl--invoke-format }} {#call-format}

```
get_token(token_id)
```

## {{ dl--parameters }} {#parameters}

### token_id

#### Description

The identifier of the token that should be returned.

**Data types**

{{ python-type--int }}

**Default value** 

{{ loss-functions__params__q__default }}

## {{ dl--output-format }} {#return-value}

{{ python-type--string }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

print(dictionary.get_token(3))
```

Output:
```bash
whatever
```

{% note info %}

This method returns the token value in accordance with the numeration in the built dictionary. In this example, the numeration for the input list for building the dictionary and the one of the dictionary differ.

{% endnote %}


