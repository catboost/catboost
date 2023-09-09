# apply

{% include [descriptions-apply](../_includes/work_src/reusage-tokenizer/apply.md) %}


## {{ dl--invoke-format }} {#call-format}

```
apply(data,
      tokenizer=None,
      unknown_token_policy=None)
```

## {{ dl--parameters }} {#parameters}

### data

#### Description

The input text to apply the dictionary to.

A zero-, one- or two-dimensional array-like data.

**Data types**

{{ python-type--string }}, {{ python-type--numpy-ndarray }}, {{ python-type--pandasDataFrame }}

**Default value**

{{ loss-functions__params__q__default }}

### tokenizer

#### Description

The tokenizer for text processing.

If this parameter is specified and a one-dimensional data is input, each element in this list is considered a sentence and is tokenized.

**Data types**

[Tokenizer](../concepts/python-reference_tokenizer.md)

**Default value**

None (the input data is considered tokenized)

### unknown_token_policy

#### Description

The policy for processing unknown tokens.

Possible values:
- {{ dictionary__policy__skip }} — All unknown tokens are skipped from the resulting token ids list (empty values are put in compliance)
- {{ dictionary__policy__insert }} — A coinciding ID is put in compliance with all unknown tokens. This ID matches the number of the tokens in the dictionary.

**Data types**

{{ python-type--string }}

**Default value**

{{ dictionary__policy__skip }}

## {{ dl--output-format }} {#return-value}

A one- or two-dimensional array with token IDs.

## {{ output--example }} {#example}

```python
from catboost.text_processing import Dictionary

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his", "tender", "heir", "whatever"])

applied_model = dictionary.apply(["might", "bear", "his", "memory"])

print(applied_model)
```

Output:

```bash
[[], [], [1], []]
```

## An example with input string tokenization {#tokenizing-example}

```python
from catboost.text_processing import Dictionary, Tokenizer

tokenized = Tokenizer()

dictionary = Dictionary(occurence_lower_bound=0)\
    .fit(["his tender heir whatever"], tokenizer=tokenized)

applied_model = dictionary.apply(["might", "bear", "his", "memory"])

print(applied_model)
```

Output:

```bash
[[], [], [1], []]
```
