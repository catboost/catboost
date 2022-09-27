# Text processing parameters

These parameters are only for Python package and Command-line.

## tokenizers {#tokenizers}

Command-line: `--tokenizers`

#### Description

{% include [reusage-cli__tokenizers__desc__div](../../_includes/work_src/reusage/cli__tokenizers__desc__div.md) %}

```json
[{
'TokenizerId1': <value>,
'option_name_1': <value>,
..
'option_name_N': <value>,}]
```

- `TokenizerId` — The unique name of the tokenizer.
- `option_name` — One of the [supported tokenizer options](../../references/tokenizer_options.md).

{% note info %}

This parameter works with `dictionaries` and `feature_calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% endnote %}

{% cut "Usage example" %}

```python
tokenizers = [{
	'tokenizerId': 'Space',
	'delimiter': ' ',
	'separator_type': 'ByDelimiter',
},{
	'tokenizerId': 'Sense',
	'separator_type': 'BySense',
}]
```

{% endcut %}

**Type**

{{ python-type__list-of-json }}

**Default value**

–

**Supported processing units**

{{ calcer_type__cpu }}


## dictionaries {#dictionaries}

Command-line: `--dictionaries`

#### Description

{% include [reusage-cli__dictionaries__desc__div](../../_includes/work_src/reusage/cli__dictionaries__desc__div.md) %}

```
[{
'dictionaryId1': <value>,
'option_name_1': <value>,
..
'option_name_N': <value>,}]
```

- `DictionaryId` — The unique name of dictionary.
- `option_name` — One of the [supported dictionary options](../../references/dictionaries_options.md).

{% note info %}

This parameter works with `tokenizers` and `feature_calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% endnote %}

{% cut "Usage example" %}

```python
dictionaries = [{
	'dictionaryId': 'Unigram',
	'max_dictionary_size': '50000',
	'gram_count': '1',
},{
	'dictionaryId': 'Bigram',
	'max_dictionary_size': '50000',
	'gram_count': '2',
}]
```

{% endcut %}

**Type**

{{ python-type__list-of-json }}

**Default value**

–

**Supported processing units**

{{ calcer_type__cpu }}


## feature_calcers {#feature_calcers}

Command-line: `--feature-calcers`

#### Description

{% include [reusage-cli__feature-calcers__desc__div](../../_includes/work_src/reusage/cli__feature-calcers__desc__div.md) %}


```json
['FeatureCalcerName[:option_name=option_value],
]
```

- `FeatureCalcerName` — The required [feature calcer](../../references/text-processing__feature_calcers.md).

- `option_name` — Additional options for feature calcers. Refer to the [list of supported calcers](../../references/text-processing__feature_calcers.md) for details on options available for each of them.


{% note info %}

This parameter works with `tokenizers` and `dictionaries` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```python
feature_calcers = [
	'BoW:top_tokens_count=1000',
	'NaiveBayes',
]
```

{% endcut %}

{% endnote %}

Type
 {{ python-type--list-of-strings }}

**Default value**

–

**Supported processing units**

{{ calcer_type__cpu }}

## text_processing {#text_processing}

Command-line: `--text-processing`

#### Description

{% include [reusage-cli__text-processing__div](../../_includes/work_src/reusage/cli__text-processing__div.md) %}

- `tokenizers`
- `dictionaries`
- `feature_calcers`

{% note alert %}

Do not use this parameter with the following ones:

- `tokenizers`
- `dictionaries`
- `feature_calcers`

{% endnote %}

**Type**

{{ python-type__json }}

**Default value**

[Default value](../../references/text-processing__test-processing__default-value.md)

**Supported processing units**

{{ calcer_type__cpu }}

