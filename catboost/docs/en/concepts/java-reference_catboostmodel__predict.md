# predict

The {{ product }} Java package provides several methods for applying a model to different types of objects and input features.

{% note info %}

The model prediction results will be correct only if the numeric and categorical features parameters contain all the features used in the model in the same order.

{% endnote %}

## Batch of objects, matrix of numerical features, matrix of hashes of categorical features, new object with model predictions {#batch-of-object-matrix-of-numerical-features-matrix-of-hashes-of-catfeatures-new-object}

```java
public CatBoostPredictions predict(float[][] numericFeatures,
                                   int[][] catFeatureHashes)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturescatfeaturehashes](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturescatfeaturehashes.md) %}


#### {{ dl--parameters }}

**numericFeatures**

A matrix of input numerical features.

**catFeatureHashes**

A matrix of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__batch-of-objects](../_includes/work_src/reusage-java/returns__catboost-predictions__batch-of-objects.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Batch of objects, matrix of numerical features, matrix of hashes of categorical features, specified object with model predictions {#batch-of-object-matrix-of-numerical-features-matrix-of-hashes-of-catfeatures-object-from-the-constructor}

```java
public void predict(float[][] numericFeatures,
                    int[][] catFeatureHashes,
                    CatBoostPredictions prediction)
```

#### {{ java__ref-table-header__modifier-and-type }}

{{ java__modifier-and-type__void }}

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturescatfeaturehashessprediction](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturescatfeaturehashessprediction.md) %}


#### {{ dl--parameters }}

**numericFeatures**

A matrix of input numerical features.

**catFeatureHashes**

A matrix of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.

**prediction**

The model's predictions.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__batch-of-objects](../_includes/work_src/reusage-java/returns__catboost-predictions__batch-of-objects.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Batch of objects, matrix of numerical features, matrix of categorical features, new object with model predictions {#batch-of-objects-matrixnumfeatures-matrixcatfeatures-newobject}

```java
public CatBoostPredictions predict(float[][] numericFeatures,
                                   String[][] catFeatures)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturescatfeatures](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturescatfeatures.md) %}


#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatures**

A matrix of input categorical features.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__batch-of-objects](../_includes/work_src/reusage-java/returns__catboost-predictions__batch-of-objects.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Batch of objects, matrix of numerical features, matrix of categorical features, specified object with model predictions {#batch-of-objects-matrixnumfeatures-matrixcatfeatures-objectfromconstructor}

```java
public void predict(float[][] numericFeatures,
                    String[][] catFeatures,
                    CatBoostPredictions prediction)
```

#### {{ java__ref-table-header__modifier-and-type }}

{{ java__modifier-and-type__void }}

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturescatfeaturesprediction__desc](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturescatfeaturesprediction__desc.md) %}


#### {{ dl--parameters }}

**numericFeatures**

A matrix of input numerical features.

**catFeatures**

A matrix of input categorical features.

**prediction**

The model's predictions.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__batch-of-objects](../_includes/work_src/reusage-java/returns__catboost-predictions__batch-of-objects.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Object, array of numerical features, array of hashes of categorical features, new object with model predictions {#object-numfeaturesarray-catfeatureshashesarray-newobject}

```java
public CatBoostPredictions predict(float[] numericFeatures,
                                   int[] catFeatureHashes)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturescatfeatureshashes.dita](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturescatfeatureshashes.dita.md) %}


#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatureHashes**

An array of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__one-object](../_includes/work_src/reusage-java/returns__catboost-predictions__one-object.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Object, array of numerical features, array of hashes of categorical features, specified object with model predictions {#object-numfeaturesarray-catfeatureshashesarray-objectfromconstructor}

```java
public void predict(float[] numericFeatures,
                    int[] catFeatureHashes,
                    CatBoostPredictions prediction)
```

#### {{ java__ref-table-header__modifier-and-type }}

{{ java__modifier-and-type__void }}

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturescatfeatureshashesprediction__desc](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturescatfeatureshashesprediction__desc.md) %}


#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatureHashes**

An array of hashes of input categorical features. These hashes must be computed by the `hashCategoricalFeature(String)` function.

**prediction**

The model's predictions.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__one-object](../_includes/work_src/reusage-java/returns__catboost-predictions__one-object.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Object, array of numerical features, array of categorical features, new object with model predictions {#object-numfeaturesarray-catfeaturesarray-newobject}

```java
predict(float[] numericFeatures,
        String[] catFeatures)
```

#### {{ java__ref-table-header__modifier-and-type }}

[CatBoostPredictions](java-reference_catboostpredictions.md)

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturesjlcatfeatures](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturesjlcatfeatures.md) %}


#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatures**

An array of input categorical features.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__one-object](../_includes/work_src/reusage-java/returns__catboost-predictions__one-object.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}


## Object, array of numerical features, array of categorical features, specified object with model predictions {#object-numfeaturesarray-catfeaturesarray-objectfromconstructor}

```java
predict(float[] numericFeatures,
        String[] catFeatures,
        CatBoostPredictions prediction)
```

#### {{ java__ref-table-header__modifier-and-type }}
{{ java__modifier-and-type__void }}

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostmodel__predictnumericfeaturesjlcatfeaturesprediction__desc](../_includes/work_src/reusage-java/java-reference_catboostmodel__predictnumericfeaturesjlcatfeaturesprediction__desc.md) %}


#### {{ dl--parameters }}

**numericFeatures**

An array of input numerical features.

**catFeatures**

An array of input categorical features.

**prediction**

The model's predictions.


#### {{ java__dl__returns }}

{% include [reusage-java-returns__catboost-predictions__one-object](../_includes/work_src/reusage-java/returns__catboost-predictions__one-object.md) %}


#### {{ java__dl__throws }}

{% include [reusage-java-throws__catboosterror__in-case-of-native-lib-errors](../_includes/work_src/reusage-java/throws__catboosterror__in-case-of-native-lib-errors.md) %}

