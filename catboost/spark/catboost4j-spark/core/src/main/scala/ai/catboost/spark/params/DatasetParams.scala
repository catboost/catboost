package ai.catboost.spark.params;

import org.apache.spark.ml.param.shared._


trait DatasetParamsTrait
  extends HasLabelCol with HasFeaturesCol with HasWeightCol
{}