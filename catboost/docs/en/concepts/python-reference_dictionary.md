# Dictionary

```python
class Dictionary(token_level_type=None,
                 gram_order=None,
                 skip_step=None,
                 start_token_id=None,
                 end_of_word_policy=None,
                 end_of_sentence_policy=None,
                 occurence_lower_bound=None,
                 max_dictionary_size=None,
                 num_bpe_units=None, 
                 skip_unknown=None, 
                 dictionary_type='FrequencyBased')
```

## {{ dl--purpose }} {#purpose}

Process dictionaries. The text must be [tokenized](python-reference_tokenizer.md) before working with dictionaries.

## {{ dl--parameters }} {#parameters}

{% include [dictionary-options-dictionary__options](../_includes/work_src/reusage-tokenizer/dictionary__options.md) %}


## {{ dl--methods }} {#methods}

### [fit](python-reference_dictionary_fit.md)

{% include [descriptions-fit](../_includes/work_src/reusage-tokenizer/fit.md) %}

### [apply](python-reference_dictionary_apply.md)

{% include [descriptions-apply](../_includes/work_src/reusage-tokenizer/apply.md) %}

### [size](python-reference_dictionary_size.md)

{% include [descriptions-size](../_includes/work_src/reusage-tokenizer/size.md) %}

### [get_token](python-reference_dictionary_get_token.md)

{% include [descriptions-get_token](../_includes/work_src/reusage-tokenizer/get_token.md) %}

### [get_tokens](python-reference_dictionary_get_tokens.md)

{% include [descriptions-get_tokens](../_includes/work_src/reusage-tokenizer/get_tokens.md) %}

### [get_top_tokens](python-reference_dictionary_get_top_tokens.md)

{% include [descriptions-get_top_tokens](../_includes/work_src/reusage-tokenizer/get_top_tokens.md) %}

### [unknown_token_id](python-reference_dictionary_unknown_token_id.md)

{% include [descriptions-unknown_token_id](../_includes/work_src/reusage-tokenizer/unknown_token_id.md) %}

### [end_of_sentence_token_id](python-reference_dictionary_end_of_sentence_token_id.md)

{% include [descriptions-end_of_sentence_token_id](../_includes/work_src/reusage-tokenizer/end_of_sentence_token_id.md) %}

### [min_unused_token_id](python-reference_dictionary_min_unused_token_id.md)

{% include [descriptions-min_unused_token_id](../_includes/work_src/reusage-tokenizer/min_unused_token_id.md) %}

### [load](python-reference_dictionary_load.md)
 
{% include [descriptions-load](../_includes/work_src/reusage-tokenizer/load.md) %}

### [save](python-reference_dictionary_save.md)

{% include [descriptions-save](../_includes/work_src/reusage-tokenizer/save.md) %}

