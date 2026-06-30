package ai.catboost

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl

/** CatBoost is a machine learning algorithm that uses gradient boosting on decision trees.
 *
 *  ==Overview==
 *  This package provides classes that implement interfaces from
 *  [[https://spark.apache.org/mllib/ Apache Spark Machine Learning Library (MLLib)]].
 *
 *  For binary and multi- classification problems use [[CatBoostClassifier]],
 *  for regression use [[CatBoostRegressor]].
 *
 *  These classes implement usual `fit` method of [[org.apache.spark.ml.Predictor]] that accept a single
 *  [[org.apache.spark.sql.DataFrame]] for training, but you can also use other `fit` method that accepts
 *  additional datasets for computing evaluation metrics and overfitting detection similarily to CatBoost's
 *  other APIs.
 *
 *  This package also contains [[Pool]] class that is CatBoost's abstraction of a dataset.
 *  It contains additional information compared to simple [[org.apache.spark.sql.DataFrame]].
 *
 *  It is also possible to create [[Pool]] with quantized features before training by calling `quantize` method.
 *  This is useful if this dataset is used for training multiple times and quantization parameters do not
 *  change. Pre-quantized [[Pool]] allows to cache quantized features data and so do not re-run
 *  feature quantization step at the start of an each training.
 *
 *  Detailed documentation is available on [[https://catboost.ai/docs/]]
 */
package object spark {
  type EModelType = native_impl.EModelType
  type ECtrTableMergePolicy = native_impl.ECtrTableMergePolicy
}
