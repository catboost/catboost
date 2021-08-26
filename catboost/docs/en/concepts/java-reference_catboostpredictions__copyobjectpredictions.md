# copyObjectPredictions

The {{ product }} Java package provides several methods for copying predictions to different types of arrays.

## Separate array {#separate-array}

```java
public double[] copyObjectPredictions(int objectIndex)
```

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostpredictions__copyobjectpredictionsobjectindex__desc](../_includes/work_src/reusage-java/java-reference_catboostpredictions__copyobjectpredictionsobjectindex__desc.md) %}


#### {{ dl--parameters }}

**objectIndex**

The index of the object.


## Specified array {#specified-array}

```java
public void copyObjectPredictions(int objectIndex,
                                  double[] predictions)
```

#### {{ dl--purpose }}

{% include [reusage-java-java-reference_catboostpredictions__copyobjectpredictionsobjectindexpredictions__desc](../_includes/work_src/reusage-java/java-reference_catboostpredictions__copyobjectpredictionsobjectindexpredictions__desc.md) %}


#### {{ dl--parameters }}

**objectIndex**

The index of the object.

**predictions**

The array to copy predictions to.
