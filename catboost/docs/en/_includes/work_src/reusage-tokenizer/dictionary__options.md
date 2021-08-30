### token_level_type

#### Description

The token level type. This parameter defines what should be considered a separate token.
Possible values:
- {{ dictionary__token-level-type__word }}
- {{ dictionary__token-level-type__letter }}

**Data types**

{{ python-type--string }}

**Default value**

{{ dictionary__token-level-type__word }}

### gram_order

#### Description

The number of words or letters in each token.

{% include [dictionary-maybe-some-other-time](maybe-some-other-time.md) %}

If the token level type is set to {{ dictionary__token-level-type__word }} and this parameter is set to 2, the following tokens are formed:
- <q>maybe some</q>
- <q>some other</q>
- <q>other time</q>

**Data types**

{{ python-type--int }}

**Default value**

1

### skip_step

#### Description

The number of words or letters to skip when joining them to tokens. This parameter takes effect if the value of the `gram_order` parameter is strictly greater than 1.

{% include [dictionary-maybe-some-other-time](maybe-some-other-time.md) %}


If the token level type is set to {{ dictionary__token-level-type__word }}, `gram_order` is set to 2 and this parameter is set to 1, the following tokens are formed:
- <q>maybe other</q>
- <q>some time</q>


**Data types**

{{ python-type--int }}

**Default value**

0

### start_token_id

#### Description

The initial shift for the token identifier.

{% include [dictionary-maybe-some-other-time](maybe-some-other-time.md) %}


If this parameter is set to 42, the following identifiers are assigned to tokens:
- 42 — <q>maybe</q>
- 43 — <q>some</q>
- 44 — <q>other</q>
- 45 — <q>time</q>

**Data types**

{{ python-type--int }}

**Default value**

0

### end_of_word_policy

#### Description

The policy for processing implicit tokens that point to the end of the word.

Possible values:

- {{ dictionary__policy__skip }}
- {{ dictionary__policy__insert }}

**Data types**

{{ python-type--string }}

**Default value**

{{ dictionary__policy__insert }}

### end_of_sentence_policy

#### Description

The policy for processing implicit tokens that point to the end of the sentence.

Possible values:

- {{ dictionary__policy__skip }}
- {{ dictionary__policy__insert }}

**Data types**

{{ python-type--string }}

**Default value**

{{ dictionary__policy__skip }}

### occurence_lower_bound

#### Description

The lower limit of token occurrences in the text to include it in the dictionary.

**Data types**

{{ python-type--int }}

**Default value**

50

### max_dictionary_size

#### Description

The maximum number of tokens in the dictionary.

**Data types**

{{ python-type--int }}

**Default value**

-1 (the size of the dictionary is not limited)

### num_bpe_units

#### Description

The number of token pairs that should be combined to a single token. The most popular tokens are combined into one and added to the dictionary as a new token.

This parameter takes effect if the value of the `dictionary_type` parameter is set to {{ dictionary__dictionary-type__Bpe }}.

**Data types**

{{ python-type--int }}

**Default value**

0 (token pairs are not combined)

### skip_unknown

#### Description

Skip unknown tokens when building the dictionary.

This parameter takes effect if the value of the `dictionary_type` parameter is set to {{ dictionary__dictionary-type__Bpe }}.

**Data types**

{{ python-type--bool }}

**Default value**

False (a special common token is assigned for all unknown tokens)

### dictionary_type

#### Description

The dictionary type.

Possible values:
- {{ dictionary__dictionary-type__FrequencyBased }}. Takes into account only the most frequent tokens. The size of the dictionary and the lower limit of token occurrences in the text to include it in the dictionary are set in `occurence_lower_bound` and `max_dictionary_size` parameters respectively.
- {{ dictionary__dictionary-type__Bpe }}. Takes into account the most frequent tokens and then makes new tokens from combinations of the most frequent token pairs. Refer to the [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) paper for algorithm details. If selected, both the Frequency Based and Bpe dictionaries are created.

**Data types**

{{ python-type--string }}

**Default value**

{{ dictionary__dictionary-type__FrequencyBased }}
