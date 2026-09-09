# predictProba

The {{ product }} Java package provides several methods for applying a classification model to different types of objects and input features and getting class probabilities.

{% note info %}

The model prediction results will be correct only if the numeric and categorical features parameters contain all the features used in the model in the same order.

{% endnote %}

{% note info %}

These methods are only applicable to models that have been trained with one of the classification loss functions. A `CatBoostError` is thrown otherwise.

The prediction dimension of the returned object is always equal to the number of classes. For binary classification models the probabilities of both classes are returned (the probability of the class `0` is at index 0 and the probability of the class `1` is at index 1), even though the raw model prediction is one-dimensional.

{% endnote %}

## Object, array of numerical features, array of categorical features {#object-numfeaturesarray-catfeaturesarray}

```java
public CatBoostPredictions predictProba(float[] numericFeatures,
                                        String[] catFeatures)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

Apply the model to the given object and calculate class probabilities.

#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatures**

An array of input categorical features.


#### {{ java__dl__returns }}

`CatBoostPredictions` with class probabilities for the specified object.


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Object, array of numerical features, array of hashes of categorical features {#object-numfeaturesarray-catfeaturehashesarray}

```java
public CatBoostPredictions predictProba(float[] numericFeatures,
                                        int[] catFeatureHashes)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

Apply the model to the given object and calculate class probabilities.

#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatureHashes**

An array of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.


#### {{ java__dl__returns }}

`CatBoostPredictions` with class probabilities for the specified object.


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Object, arrays of numerical, categorical, text and embedding features {#object-all-feature-types}

```java
public CatBoostPredictions predictProba(float[] numericFeatures,
                                        String[] catFeatures,
                                        String[] textFeatures,
                                        float[][] embeddingFeatures)
```

```java
public CatBoostPredictions predictProba(float[] numericFeatures,
                                        int[] catFeatureHashes,
                                        String[] textFeatures,
                                        float[][] embeddingFeatures)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

Apply the model to the given object and calculate class probabilities.

#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatures**

An array of input categorical features.

**catFeatureHashes**

An array of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.

**textFeatures**

An array of input text features.

**embeddingFeatures**

An array of input embedding features.


#### {{ java__dl__returns }}

`CatBoostPredictions` with class probabilities for the specified object.


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Batch of objects, matrix of numerical features, matrix of categorical features {#batch-of-objects-matrixnumfeatures-matrixcatfeatures}

```java
public CatBoostPredictions predictProba(float[][] numericFeatures,
                                        String[][] catFeatures)
```

```java
public CatBoostPredictions predictProba(float[][] numericFeatures,
                                        int[][] catFeatureHashes)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

Apply the model to a batch of objects and calculate class probabilities.

#### {{ dl--parameters }}

**numericFeatures**

A matrix of input numerical features.

**catFeatures**

A matrix of input categorical features.

**catFeatureHashes**

A matrix of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.


#### {{ java__dl__returns }}

`CatBoostPredictions` with class probabilities for a batch of objects.


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Batch of objects, matrices of numerical, categorical, text and embedding features {#batch-of-objects-all-feature-types}

```java
public CatBoostPredictions predictProba(float[][] numericFeatures,
                                        String[][] catFeatures,
                                        String[][] textFeatures,
                                        float[][][] embeddingFeatures)
```

```java
public CatBoostPredictions predictProba(float[][] numericFeatures,
                                        int[][] catFeatureHashes,
                                        String[][] textFeatures,
                                        float[][][] embeddingFeatures)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

Apply the model to a batch of objects and calculate class probabilities.

#### {{ dl--parameters }}

**numericFeatures**

A matrix of input numerical features.

**catFeatures**

A matrix of input categorical features.

**catFeatureHashes**

A matrix of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.

**textFeatures**

A matrix of input text features.

**embeddingFeatures**

A matrix of input embedding features.


#### {{ java__dl__returns }}

`CatBoostPredictions` with class probabilities for a batch of objects.


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}
