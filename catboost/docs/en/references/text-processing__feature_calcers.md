# Feature calcers and corresponding options

The following is a list of options for the --feature-calcers`for the Command-line version parameter (these options are set in `option_name`):

- {{ dictionary__feature-calcers__BoW }} (Bag of words) — Boolean (0/1) features reflecting whether the object contains the token_id. The number of features is equal to the dictionary size.

    Supported options:
    - top_tokens_count — The maximum number of features to create. If set, the specified number top tokens is taken into account and the corresponding number of new features is created.

- {{ dictionary__feature-calcers__NaiveBayes }} — Multinomial naive bayes model, the number of created features is equal to the number of classes. To avoid target leakage, this model is computed online on several dataset permutations (similarly to the estimation of CTRs).

- {{ dictionary__feature-calcers__BM25 }} — A function that is used for ranking purposes by search engines to estimate the relevance of documents. To avoid target leakage, this model is computed online on several dataset permutations (similarly to the estimation of CTRs).

