# get_top_tokens

{% include [descriptions-get_top_tokens](../_includes/work_src/reusage-tokenizer/get_top_tokens.md) %}

{% include [reusage-tokenizer-fbd_only](../_includes/work_src/reusage-tokenizer/fbd_only.md) %}


## {{ dl--invoke-format }} {#call-format}

```
get_top_tokens(top_size=None)
```

## {{ dl--parameters }} {#parameters}

### top_size

#### Description

The top size to output.

**Data types**

{{ python-type--int }}

**Default value**

10

## {{ dl--output-format }} {#return-value}

{{ python-type--list }}

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(['A', 'C', 'C', 'A', 'B', 'A', 'D'])

print(dictionary.get_top_tokens(2))

```

Output:
```bash
['A', 'C']
```

