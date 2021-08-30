# BPE Dictionary

#### {{ output--contains }}
The trainedBPE dictionary.
#### {{ output--format }}

Each line contains information regarding a single new token.

Format:

```
<token_id1><\t><token_id2><\t><number_of_occurrences><\t><token>
```

- `token_id1` — The token ID of the first part of the new token.
    
- `token_id2` — The token ID of the second part of the new token.
    
- {% include [dictionaries-number-of-occurrences-desc](../_includes/work_src/reusage-tokenizer/number-of-occurrences-desc.md) %}
    
- `token` — The value of the token.

{% include [dictionaries-tokens-origin](../_includes/work_src/reusage-tokenizer/tokens-origin.md) %}


#### {{ output--example }}

The following is the frequency based dictionary:

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

The following is the BPE dictionary:

```
0      5     1	How high
1      7     1	It's snowing
2      10    1	Today tomorrow
3      4     1	and forever
8      6     1	the moon
14     9     1	It's snowing today
13     17    1	How high the moon
15     16    1	Today tomorrow and forever
```

Identifiers in the range [0;10] point to tokens from the Frequency Based dictionary.

Identifiers 11 and 12 are reserved for the unknown and end of sentence tokens respectively.

Identifiers starting from 13 point to the tokens from the BPE dictionary.

