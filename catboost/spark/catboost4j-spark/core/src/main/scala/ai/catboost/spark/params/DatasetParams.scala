package ai.catboost.spark.params;

import org.apache.spark.ml.param.shared._

/** Common dataset columns parameters */
trait DatasetParamsTrait
  extends HasLabelCol with HasFeaturesCol with HasWeightCol
{}
