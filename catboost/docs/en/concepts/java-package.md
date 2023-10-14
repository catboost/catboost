# Java

#### Quick start

To apply a previously trained {{ product }} model in Java:
1. Install the package using a package manager.

    Add the following block to the dependencies section of the pom.xml file for Maven:

    ```xml
    <!-- https://mvnrepository.com/artifact/ai.catboost/catboost-prediction -->
    <dependency>
    <groupId>ai.catboost</groupId>
    <artifactId>catboost-prediction</artifactId>
    <version>source_version</version>
    </dependency>
    ```

    `source_version` should be set to one of the [main {{ product }} releases]({{ releases-page }}). Available versions can also be checked on the [Maven repository site](https://h.yandex-team.ru/?https%3A%2F%2Fmvnrepository.com%2Fartifact%2Fai.catboost%2Fcatboost-prediction).

1. Load the trained model:

    ```java
    import ai.catboost.CatBoostModel;
    import ai.catboost.CatBoostPredictions;

    CatBoostModel model = CatBoostModel.loadModel("model.cbm");
    ```

1. Apply the model:
    ```java
    CatBoostPredictions prediction = model.predict(new float[]{0.1f, 0.3f, 0.2f}, new String[]{"foo", "bar", "baz"});
    // assuming that this is a regression task
    System.out.print("model value is " + String.valueOf(prediction.get(0, 0));
    ```

#### Provided classes

Class | Description |
:--- | :---
[CatBoostModel](java-reference_catboostmodel.md) | Basic model application methods. |
[CatBoostPredictions](java-reference_catboostpredictions.md) | A wrapper that provides methods for making convenient predictions for certain classes.|

#### [Build from source](../installation/java-installation-build-from-source-maven.md)
