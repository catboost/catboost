# Dictionaries options

The following is a list of options for the 

{% cut "`dictionaries`" %}

`--dictionaries`for the Command-line version

{% endcut %}

 parameter (these options are set in `option_name`):**{{ common-text__table-title__param }}:** `token_level_type`

**{{ common-text__table-title__data-types }}:** {{ python-type--string }}

**{{ common-text__table-title__desc }}:**
The token level type. This parameter defines what should be considered a separate token.

Possible values:
- {{ dictionary__token-level-type__word }}
- {{ dictionary__token-level-type__letter }}


**{{ common-text__table-title__default_value }}:** {{ dictionary__token-level-type__word }}


**{{ common-text__table-title__param }}:** `gram_order`

**{{ common-text__table-title__data-types }}:** {{ python-type--int }}

**{{ common-text__table-title__desc }}:**
The number of words or letters in each token.

{% include [dictionary-maybe-some-other-time](../_includes/work_src/reusage-tokenizer/maybe-some-other-time.md) %}

If the token level type is set to {{ dictionary__token-level-type__word }} and this parameter is set to 2, the following tokens are formed:
- <q>maybe some</q>
- <q>some other</q>
- <q>other time</q>


**{{ common-text__table-title__default_value }}:** 1


**{{ common-text__table-title__param }}:** `skip_step`

**{{ common-text__table-title__data-types }}:** {{ python-type--int }}

**{{ common-text__table-title__desc }}:**
The number of words or letters to skip when joining them to tokens. This parameter takes effect if the value of the `gram_order` parameter is strictly greater than 1.

{% include [dictionary-maybe-some-other-time](../_includes/work_src/reusage-tokenizer/maybe-some-other-time.md) %}

If the token level type is set to {{ dictionary__token-level-type__word }}, `gram_order` is set to 2 and this parameter is set to 1, the following tokens are formed:
- <q>maybe other</q>
- <q>some time</q>


**{{ common-text__table-title__default_value }}:** 0


**{{ common-text__table-title__param }}:** `start_token_id`

**{{ common-text__table-title__data-types }}:** {{ python-type--int }}

**{{ common-text__table-title__desc }}:**
The initial shift for the token identifier.

{% include [dictionary-maybe-some-other-time](../_includes/work_src/reusage-tokenizer/maybe-some-other-time.md) %}

If this parameter is set to 42, the following identifiers are assigned to tokens:
- 42 — <q>maybe</q>
- 43 — <q>some</q>
- 44 — <q>other</q>
- 45 — <q>time</q>


**{{ common-text__table-title__default_value }}:** 0


**{{ common-text__table-title__param }}:** `end_of_word_token_policy`

**{{ common-text__table-title__data-types }}:** {{ python-type--string }}

**{{ common-text__table-title__desc }}:**
The policy for processing implicit tokens that point to the end of the word.

Possible values:
- {{ dictionary__policy__skip }}
- {{ dictionary__policy__insert }}


**{{ common-text__table-title__default_value }}:** {{ dictionary__policy__insert }}


**{{ common-text__table-title__param }}:** `end_of_sentence_token_policy`

**{{ common-text__table-title__data-types }}:** {{ python-type--string }}

**{{ common-text__table-title__desc }}:**
The policy for processing implicit tokens that point to the end of the sentence.

Possible values:
- {{ dictionary__policy__skip }}
- {{ dictionary__policy__insert }}


**{{ common-text__table-title__default_value }}:** {{ dictionary__policy__skip }}


**{{ common-text__table-title__param }}:** `occurrence_lower_bound`

**{{ common-text__table-title__data-types }}:** {{ python-type--int }}

**{{ common-text__table-title__desc }}:**
The lower limit of token occurrences in the text to include it in the dictionary.


**{{ common-text__table-title__default_value }}:** 50


**{{ common-text__table-title__param }}:** `max_dictionary_size`

**{{ common-text__table-title__data-types }}:** {{ python-type--int }}

**{{ common-text__table-title__desc }}:**
The maximum number of tokens in the dictionary.


**{{ common-text__table-title__default_value }}:** -1 (the size of the dictionary is not limited)

