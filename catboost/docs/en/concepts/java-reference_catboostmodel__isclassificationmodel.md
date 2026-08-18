# isClassificationModel

```java
public boolean isClassificationModel()
```

#### {{ dl--purpose }}

Return `true` if the model has been trained with a classification loss function and, therefore, class probabilities can be calculated for it with [predictProba](java-reference_catboostmodel__predictproba.md).

Models that do not contain information about the loss function they have been trained with are considered to be classification models.
