# Java

#### Quick start

To apply a previously trained {{ product }} model in Java:
1. Install the package using a package manager.

    Note that the package contains native code shared libraries inside. The supported platforms are:

    |Operating system|CPU architectures|GPU support using [CUDA](https://developer.nvidia.com/cuda-zone)|
    |--------|-----------------|------------|
    | macOS (versions currently supported by Apple) | x86_64 and arm64 |no|
    | Linux (compatible with [manylinux2014 platform tag](https://peps.python.org/pep-0599/) ) | x86_64 and aarch64 |yes|
    | Windows 10 and 11 | x86_64 |yes|

    {% note info %}

    Release binaries for x86_64 CPU architectures are built with SIMD extensions SSE2, SSE3, SSSE3, SSE4 enabled. If you need to run {{ product }} on older CPUs that do not support these instruction sets build [{{ product }} artifacts yourself](../installation/java-installation-build-from-source-maven.md)

    {% endnote %}

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
