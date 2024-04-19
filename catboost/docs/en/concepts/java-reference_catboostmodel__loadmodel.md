# loadModel

The {{ product }} Java package provides different methods for loading {{ product }} models.

## Load the model from a stream {#loadfromastream}

```java
public static CatBoostModel loadModel(java.io.InputStream in)
```

#### {{ dl--purpose }}

{% include [reusage-java-java__loadcatboost-from-stream__desc](../_includes/work_src/reusage-java/java__loadcatboost-from-stream__desc.md) %}


#### {{ dl--parameters }}

**in**

The input stream with the {{ product }} model.

#### {{ java__dl__returns }}

{% include [reusage-java-java__load-model__returns](../_includes/work_src/reusage-java/java__load-model__returns.md) %}


#### {{ java__dl__throws }}

- {% include [reusage-java-throws__catboosterror__when-model-not-loaded](../_includes/work_src/reusage-java/throws__catboosterror__when-model-not-loaded.md) %}

- {% include [reusage-java-throws__javaioioexception](../_includes/work_src/reusage-java/throws__javaioioexception.md) %}


## Load the model from a file {#loadfromafile}

```java
public static CatBoostModel loadModel(String modelPath)
```

#### {{ dl--purpose }}

{% include [reusage-java-java__loadcatboost-from-file__desc](../_includes/work_src/reusage-java/java__loadcatboost-from-file__desc.md) %}


#### {{ dl--parameters }}

**modelPath**

The path to the input {{ product }} model.

#### {{ java__dl__returns }}

{% include [reusage-java-java__load-model__returns](../_includes/work_src/reusage-java/java__load-model__returns.md) %}


#### {{ java__dl__throws }}

- {% include [reusage-java-throws__catboosterror__when-model-not-loaded](../_includes/work_src/reusage-java/throws__catboosterror__when-model-not-loaded.md) %}

- {% include [reusage-java-throws__javaioioexception](../_includes/work_src/reusage-java/throws__javaioioexception.md) %}

