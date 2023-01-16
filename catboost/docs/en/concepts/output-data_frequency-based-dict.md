# Frequency Based Dictionary

#### {{ output--contains }}
The trained Frequency Based Dictionary.
#### {{ output__header-format }}

The first row in the output file contains information regarding the training parameters.

Format:

```
{"key_1":"value_1","key_2":"value_2",.., "key_N":"value_N"}
```

#### {{ output--format }}

The second row contains the number of tokens in the dictionary.

Each row starting from the second contains information regarding a single token.

Format:
```
<token_ID><\t><number_of_occurrences><\t><token>
```

- `token ID` — A zero-based token identifier. Tokens are sorted case sensitive ordering.
    
- {% include [dictionaries-number-of-occurrences-desc](../_includes/work_src/reusage-tokenizer/number-of-occurrences-desc.md) %}
    
- `token` — The value of the token.
    

#### {{ output--example }}

```
{"end_of_word_token_policy":"Insert","skip_step":"0","start_token_id":"0","token_level_type":"Word","dictionary_format":"id_count_token","end_of_sentence_token_policy":"Skip","gram_order":"1"}
11
0       1	How
1       1	It's
2       1	Today
3       1	and
4       1	forever
5       1	high
6       1	moon
7       1	snowing
8       1	the
9       1	today
10      1	tomorrow
```

