# Transforming embedding features to numerical features

{% include [types-of-supported-features-supported-feature-types](../_includes/work_src/reusage-common-phrases/supported-feature-types.md) %}


Embedding features are transformed to numerical. The transformation method generally includes the following stages:
1. **Loading and storing embedding features**

    The embedding feature is loaded as a column. Every element in this column is an array of fixed size of numerical values.

    To load embedding features to {{ product }}:
    - Specify the {{ cd-file__col-type__NumVector }} column type in the [column descriptions](input-data_column-descfile.md) file if the dataset is loaded from a file.
    - Use the `embedding_features` parameter in the Python package.

1. **Estimating numerical features**

    Each embedding is transformed to the one or multiple numeric features.

    Supported methods for calculating numerical features:

    - [Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
        - For classification the features will be calculated as Gaussian likelihood values for each class.

    - [K Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
        - For classification the features will be counts of target classes among the found neighbors from the training set.
        - For regression the single feature will be the average target value among the found neighbors from the training set.

1. **Training**

    Computed numerical features are passed to the regular {{ product }} training algorithm.
