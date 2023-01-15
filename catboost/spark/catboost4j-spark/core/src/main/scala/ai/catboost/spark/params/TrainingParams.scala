package ai.catboost.spark.params;

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable

import ai.catboost.spark.params.macros.ParamGetterSetter


trait TrainingParamsTrait
  extends QuantizationParamsTrait with HasLabelCol with HasFeaturesCol with HasWeightCol
{
  @ParamGetterSetter
  final val sparkPartitionCount: IntParam = new IntParam(
    this,
    "sparkPartitionCount",
    "The number of partitions used during training. Corresponds to the number of active parallel tasks."+
    " Set to the number of active executors by default"
  )

  @ParamGetterSetter
  final val workerInitializationTimeout: DurationParam = new DurationParam(
    this,
    "workerInitializationTimeout",
    "Timeout to wait until CatBoost workers on Spark executors are initalized and sent their info to master. "
    + "Depends on dataset size. Default is 10 minutes"
  )

  setDefault(workerInitializationTimeout, java.time.Duration.ofMinutes(10))
}


trait ClassifierTrainingParamsTrait extends TrainingParamsTrait {
}


trait RegressorTrainingParamsTrait extends TrainingParamsTrait {

}
