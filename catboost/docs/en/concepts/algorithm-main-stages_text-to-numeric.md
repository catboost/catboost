# Transforming text features to numerical features

{% include [types-of-supported-features-supported-feature-types](../_includes/work_src/reusage-common-phrases/supported-feature-types.md) %}


Text features are transformed to numerical. The transformation method generally includes the following stages:
1. **Loading and storing text features**

    The text feature is loaded as a column. Every element in this column is a string.

    To load text features to {{ product }}:
    - Specify the {{ cd-file__col-type__Text }} column type in the [column descriptions](input-data_column-descfile.md) file if the dataset is loaded from a file.
    - Use the `text_features` parameter in the Python package.

1. **Text preprocessing**

    1. **Tokenization**

    Each value of a text feature is converted to a string sequence by splitting the original string by space.

    An element of such sequence is called _word_.

    1. **Dictionary creation**

    A dictionary is a data structure that collects all values of a text feature, defines the minimum unit of text sequence representation (which is called _token_), and assigns a number to each token.

    The type of dictionary defines the token type:

    - {{ dictionary__token-level-type__letter }} — A symbol from the string. For example, the <q>abra cadabra</q> text forms the following dictionary: `{a, b, c, d, r}`.

    - {{ dictionary__token-level-type__word }} — A word (an element from the sequence of strings obtained on step [2.a](#text-processing__step__tokenization)). For example, the <q>abra cadabra</q> text forms the following dictionary: `{'abra', 'cadabra'}`.

    The type of dictionary can be specified in the `token_level_type` dictionary parameter.

    Token sequences can be combined to one unique token, which is called _N-gramm_. N stands for the length (the number of combined sequences) of this new sequence. The length can be specified in the `gram_order` dictionary parameter.

    Combining sequences can be useful if it is required to perceive the text more continuously. For example, let's examine the following texts: <q>cat defeat mouse</q> and <q>mouse defeat cat</q>. These texts have the same tokens in terms of {{ dictionary__token-level-type__word }} dictionaries (`{'cat', 'defeat', 'mouse'}`) and different tokens in terms of bi-gram word dictionary (`{'cat defeat', 'defeat mouse'}` and `{'mouse defeat', 'defeat cat'}`).

    It is also possible to filter rare words using the `occurence_lower_bound` dictionary parameter or to limit the maximum dictionary size to the desired number of using the `max_dictionary_size` dictionary parameter.

    1. **Converting strings to numbers**

    Each string from the text feature is converted to a token identifier from the dictionary.

    {% cut "Example" %}

    Source text:

    ObjectId | Text feature
    ----- | -----
    0 | "Cats are so cute :)"
    1 | "Mouse scares me"
    2 | "The cat defeated the mouse"
    3 | "Cute: Mice gather an army!"
    4 | "Army of mice defeated the cat :("
    5 | "Cat offers peace"
    6 | "Cat is scared :("
    7 | "Cat and mouse live in peace :)"

    Splitting text into words:

    ObjectId | Text feature
    ----- | -----
    0 | ['Cats', 'are', 'so', 'cute', ':)']
    1 | ['Mouse', 'scares', 'me']
    2 | ['The', 'cat', 'defeated', 'the', 'mouse']
    3 | ['Cute:', 'Mice', 'gather', 'an', 'army!']
    4 | ['Army', 'of', 'mice', 'defeated', 'the', 'cat', ':(']
    5 | ['Cat', 'offers', 'peace']
    6 | ['Cat', 'is', 'scared', ':(']
    7 | ['Cat', 'and', 'mouse', 'live', 'in', 'peace', ':)']

    Creating dictionary:

    Word | TokenId
    ----- | -----
    "Cats" | 0
    "are" | 1
    "so" | 2
    "cute" | 3
    ...
    "and" | 26
    "live" | 27
    "in" | 28

    Converting strings into numbers:

    ObjectId | TokenizedFeature
    ----- | -----
    0 | [0, 1, 2, 3, 4]
    1 | [5, 6, 7]
    2 | [8, 9, 10, 11, 12]
    3 | [13, 14, 15, 16, 17]
    4 | [18, 19, 20, 10, 11, 9, 21]
    5 | [22, 23, 24]
    6 | [22, 25, 26, 21]
    7 | [22, 27, 12, 28, 29, 24, 4]

    {% endcut %}

1. **Estimating numerical features**

    Numerical features are calculated based on the source tokenized.

    Supported methods for calculating numerical features:

    - {{ dictionary__feature-calcers__BoW }} (Bag of words) — Boolean (0/1) features reflecting whether the object contains the token_id. The number of features is equal to the dictionary size.

    Supported options:
    - top_tokens_count — The maximum number of features to create. If set, the specified number top tokens is taken into account and the corresponding number of new features is created.

    - {{ dictionary__feature-calcers__NaiveBayes }} — Multinomial naive bayes model, the number of created features is equal to the number of classes. To avoid target leakage, this model is computed online on several dataset permutations (similarly to the estimation of CTRs).

    - {{ dictionary__feature-calcers__BM25 }} — A function that is used for ranking purposes by search engines to estimate the relevance of documents. To avoid target leakage, this model is computed online on several dataset permutations (similarly to the estimation of CTRs).

1. **Training**

    Computed numerical features are passed to the regular {{ product }} training algorithm.
