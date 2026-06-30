# Embeddings features

{{ product }} supports numerical, categorical, text, and embeddings features.

Embedding features are used to build some new numeric features.
At the moment, we support two types of such derived numerical features. The first one uses Linear Discriminant Analysis to make a projection to lower dimension space. The second one uses the nearest neighbor search to calculate the number of close-by embeddings in every class.

We do not use coordinates of embedding features in our models. If you think that they could improve the quality of a model, you can add them as numerical features together with embedding ones.

Even though every vector feature can be used in a model, we optimized performance for:

- Vectors with dimensions in the order of several hundreds.
- Datasets with normally distributed classes.

Choose the implementation for details on the methods and/or parameters used that are required to start using embeddings features.

## {{ python-package }}

### Class / method
- [CatBoost](../concepts/python-reference_catboost.md) ([fit](../concepts/python-reference_catboost_fit.md))
- [CatBoostClassifier](../concepts/python-reference_catboostclassifier.md) ([fit](../concepts/python-reference_catboostclassifier_fit.md))
- [Pool](../concepts/python-reference_pool.md)

#### Parameters

##### embedding_features

A one-dimensional array of embeddings columns indices (specified as integers) or names (specified as strings).

{% include [reusage-python__cat_features__description__non-catfeatures-text](../_includes/work_src/reusage/python__cat_features__description__non-catfeatures-text.md) %}

## Command-line version binary

Specify `NumVector` for embedding features' columns in [the column description file](../concepts/input-data_column-descfile#numvector) when they are present in the input datasets.
