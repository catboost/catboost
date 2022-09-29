# Text features

{{ product }} supports numerical, categorical, text, and embeddings features.

Text features are used to build new numeric features. See the [Transforming text features to numerical features](../concepts/algorithm-main-stages_text-to-numeric.md) section for details.

Choose the implementation for details on the methods and/or parameters used that are required to start using text features.

## {{ python-package }}

### Class / method
- [CatBoost](../concepts/python-reference_catboost.md) ([fit](../concepts/python-reference_catboost_fit.md))
- [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) ([fit](../concepts/python-reference_catboostclassifier_fit.md))
- [Pool](../concepts/python-reference_pool.md)

#### Parameters

##### text_features

A one-dimensional array of text columns indices (specified as integers) or names (specified as strings).

{% include [reusage-python__cat_features__description__non-catfeatures-text](../_includes/work_src/reusage/python__cat_features__description__non-catfeatures-text.md) %}

### Text processing parameters

Supported [training parameters](../references/training-parameters/index.md):

#### tokenizers

**Description**

{% include [reusage-cli__tokenizers__desc__div](../_includes/work_src/reusage/cli__tokenizers__desc__div.md) %}

```json
[{
'TokenizerId1': <value>,
'option_name_1': <value>,
..
'option_name_N': <value>,}]
```

- `TokenizerId` — The unique name of the tokenizer.
- `option_name` — One of the [supported tokenizer options](../references/tokenizer_options.md).

{% note info %}

This parameter works with `dictionaries` and `feature_calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

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

**Possible types**

{{ python-type__list-of-json }}

**Default value**

–

**Supported processing units**

{% include [reusage-python-cpu-and-gpu](../_includes/work_src/reusage-python/cpu-and-gpu.md) %}

#### dictionaries

**Description**

{% include [reusage-cli__dictionaries__desc__div](../_includes/work_src/reusage/cli__dictionaries__desc__div.md) %}

```
[{
'dictionaryId1': <value>,
'option_name_1': <value>,
..
'option_name_N': <value>,}]
```

- `DictionaryId` — The unique name of dictionary.
- `option_name` — One of the [supported dictionary options](../references/dictionaries_options.md).

{% note info %}

This parameter works with `tokenizers` and `feature_calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

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


**Possible types**

{{ python-type__list-of-json }}

**undefined:**

–

**Supported processing units**

{% include [reusage-python-cpu-and-gpu](../_includes/work_src/reusage-python/cpu-and-gpu.md) %}

#### feature_calcers

**Description**

{% include [reusage-cli__feature-calcers__desc__div](../_includes/work_src/reusage/cli__feature-calcers__desc__div.md) %}


```json
['FeatureCalcerName[:option_name=option_value],
]
```

- `FeatureCalcerName` — The required [feature calcer](../references/text-processing__feature_calcers.md).

- `option_name` — Additional options for feature calcers. Refer to the [list of supported calcers](../references/text-processing__feature_calcers.md) for details on options available for each of them.


{% note info %}

This parameter works with `tokenizers` and `dictionaries` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```python
feature_calcers = [
	'BoW:top_tokens_count=1000',
	'NaiveBayes',
]
```

{% endcut %}

{% endnote %}

**Possible types**

{{ python-type--list-of-strings }}

**Default value**

–

**Supported processing units**

{% include [reusage-python-cpu-and-gpu](../_includes/work_src/reusage-python/cpu-and-gpu.md) %}


#### text_processing

**Description**

{% include [reusage-cli__text-processing__div](../_includes/work_src/reusage/cli__text-processing__div.md) %}

- `tokenizers`
- `dictionaries`
- `feature_calcers`

{% note alert %}

Do not use this parameter with the following ones:

- `tokenizers`
- `dictionaries`
- `feature_calcers`

{% endnote %}

**Possible types**

{{ python-type__json }}

**Default value**

 [Default value](../references/text-processing__test-processing__default-value.md)

**Supported processing units**

{% include [reusage-python-cpu-and-gpu](../_includes/work_src/reusage-python/cpu-and-gpu.md) %}

### Additional classes

Additional classes are provided for text processing:

#### [Tokenizer](../concepts/python-reference_tokenizer.md)

**Class purpose:**

{% include [text-features-python__tokenize_class__description](../_includes/work_src/reusage-python/python__tokenize_class__description.md) %}

#### [Dictionary](../concepts/python-reference_dictionary.md)

**Class purpose:**

{% include [text-features-python__dictionary_class__description](../_includes/work_src/reusage-python/python__dictionary_class__description.md) %}

## {{ title__implementation__cli }}

For the [Train a model](../references/training-parameters/index.md) command:

### --tokenizers

**Key description:**

Tokenizers used to preprocess {{ data-type__text }} type feature columns before creating the dictionary.

Format:

```
TokenizerId[:option_name=option_value]
```

- `TokenizerId` — The unique name of the tokenizer.
- `option_name` — One of the [supported tokenizer options](../references/tokenizer_options.md).

{% note info %}

This parameter works with `--dictionaries` and `--feature-calcers` parameters.

For example, if a single tokenizer, three dictionaries and two feature calcers are given, a total of 6 new groups of features are created for each original text feature ($1 \cdot 3 \cdot 2 = 6$).

{% cut "Usage example" %}

```
--tokenizers "Space:delimiter= :separator_type=ByDelimiter,Sense:separator_type=BySense"
```

{% endcut %}

{% endnote %}



### --dictionaries

**Command keys:**
Dictionaries used to preprocess {{ data-type__text }} type feature columns.

Format:

```
DictionaryId[:option_name=option_value]
```

- `DictionaryId` — The unique name of dictionary.
- `option_name` — One of the [supported dictionary options](../references/dictionaries_options.md).

{% note info %}

This parameter works with `--tokenizers` and `--feature-calcers` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```
--dictionaries "Unigram:gram_count=1:max_dictionary_size=50000,Bigram:gram_count=2:max_dictionary_size=50000"
```

{% endcut %}

{% endnote %}


### --feature-calcers

**Command keys:**
Feature calcers used to calculate new features based on preprocessed {{ data-type__text }} type feature columns.

Format:

```
FeatureCalcerName[:option_name=option_value]
```

- `FeatureCalcerName` — The required [feature calcer](../references/text-processing__feature_calcers.md).

- `option_name` — Additional options for feature calcers. Refer to the [list of supported calcers](../references/text-processing__feature_calcers.md) for details on options available for each of them.


{% note info %}

This parameter works with `--tokenizers` and `--dictionaries` parameters.

{% include [reusage-tokenizer-dictionaries-feature-calcers__note_div](../_includes/work_src/reusage/tokenizer-dictionaries-feature-calcers__note_div.md) %}

{% cut "Usage example" %}

```
--feature-calcers BoW:top_tokens_count=1000,NaiveBayes
```

{% endcut %}

{% endnote %}


### --text-processing

**Command keys:**
A JSON specification of tokenizers, dictionaries and feature calcers, which determine how text features are converted into a list of float features.

[Example](../references/text-processing__specification-example.md)

Refer to the description of the following parameters for details on supported values:

- `--tokenizers`
- `--dictionaries`
- `--feature-calcers`

{% note alert %}

Do not use this parameter with the following ones:
- `--tokenizers`
- `--dictionaries`
- `--feature-calcers`

{% endnote %}

