# tokenize

{% include [tokenize-tokenize__purpose](../_includes/work_src/reusage-tokenizer/tokenize__purpose.md) %}


## {{ dl--invoke-format }} {#call-format}

```
tokenize(s)
```

## {{ dl--parameters }} {#parameters}

### s

#### Description

The input string that has to be tokenized.

**Data types**

{{ data-type__String }}

**Default value**

{{ loss-functions__params__q__default }}


## {{ dl--output-format }} {#return-value}

A {{ python-type--list }} of tokens.

## {{ output--example }} {#example}

```python
from catboost.text_processing import Tokenizer


text="Still, I would love to see you at 12, if you don't mind"

tokenized = Tokenizer(lowercasing=True,
                      separator_type='BySense',
                      token_types=['Word', 'Number']).tokenize(text)

print tokenized
```

Output:
```bash
['still', 'i', 'would', 'love', 'to', 'see', 'you', 'at', '12', 'if', 'you', "don't", 'mind']
```

