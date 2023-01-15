package ai.catboost.spark.params;

import org.apache.spark.ml.param._

import ai.catboost.spark.params.macros.ParamGetterSetter

trait IgnoredFeaturesParams extends Params {
  @ParamGetterSetter
  final val ignoredFeaturesIndices: IntArrayParam = new IntArrayParam(
    this,
    "ignoredFeaturesIndices",
    "Feature indices to exclude from the training"
  )

  @ParamGetterSetter
  final val ignoredFeaturesNames: StringArrayParam = new StringArrayParam(
    this,
    "ignoredFeaturesNames",
    "Feature names to exclude from the training"
  )
}

trait ThreadCountParams extends Params {
  @ParamGetterSetter
  final val threadCount: IntParam = new IntParam(
    this,
    "threadCount",
    "Number of CPU threads in parallel operations on client"
  )
}
