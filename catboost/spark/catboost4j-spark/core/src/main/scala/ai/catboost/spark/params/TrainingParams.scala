package ai.catboost.spark.params;

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable

import ai.catboost.spark.params.macros.ParamGetterSetter

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._ // enums


/** Params for training CatBoost. See documentation on [[https://catboost.ai/docs/]] for details. */
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
  final val trainingDriverListeningPort: IntParam = new IntParam(
    this,
    "trainingDriverListeningPort",
    "Port used for communication on the driver's side during training. Default is 0, that means automatic assignment"
    + ""
  )

  setDefault(trainingDriverListeningPort, 0)

  @ParamGetterSetter
  final val workerInitializationTimeout: DurationParam = new DurationParam(
    this,
    "workerInitializationTimeout",
    "Timeout to wait until CatBoost workers on Spark executors are initalized and sent their info to master. "
    + "Depends on dataset size. Default is 10 minutes"
  )

  setDefault(workerInitializationTimeout, java.time.Duration.ofMinutes(10))

  @ParamGetterSetter
  final val workerMaxFailures: IntParam = new IntParam(
    this,
    "workerMaxFailures",
    "Number of individual CatBoost workers failures before giving up training. "
    + "Should be greater than or equal to 1. Default is 4"
  )

  setDefault(workerMaxFailures, 4)

  @ParamGetterSetter
  final val workerListeningPort: IntParam = new IntParam(
    this,
    "workerListeningPort",
    "Port used for communication on the workers' side during training. Default is 0, that means automatic assignment"
    + ""
  )

  setDefault(workerListeningPort, 0)

  @ParamGetterSetter
  final val connectTimeout: DurationParam = new DurationParam(
    this,
    "connectTimeout",
    "Timeout to wait while establishing socket connections between TrainingDriver and workers."
    + "Default is 1 minute"
  )

  setDefault(connectTimeout, java.time.Duration.ofMinutes(1))

  @ParamGetterSetter
  final val lossFunction: Param[String] = new Param[String](
    this,
    "lossFunction",
    "The metric to use in training. The specified value also determines the machine learning problem to "
    + "solve. Some metrics support optional parameters (see the Objectives and metrics documentation section "
    + "for details on each metric)."
  )

  @ParamGetterSetter
  final val customMetric: StringArrayParam = new StringArrayParam(
    this,
    "customMetric",
    "Metric values to output during training. These functions are not optimized and are displayed for "
    + " informational purposes only. Some metrics support optional parameters (see the Objectives and "
    + " metrics documentation section for details on each metric)."
  )

  @ParamGetterSetter
  final val evalMetric: Param[String] = new Param[String](
    this,
    "evalMetric",
    "The metric used for overfitting detection (if enabled) and best model selection (if enabled). Some "
    + "metrics support optional parameters (see the Objectives and metrics documentation section for details "
    + "on each metric)."
  )

  @ParamGetterSetter
  final val iterations: IntParam = new IntParam(
    this,
    "iterations",
    "The maximum number of trees that can be built when solving machine learning problems. When using other "
    + "parameters that limit the number of iterations, the final number of trees may be less than the number "
    + "specified in this parameter. "
    + "Default value is 1000."

  )

  @ParamGetterSetter
  final val learningRate: FloatParam = new FloatParam(
    this,
    "learningRate",
    "The learning rate. Used for reducing the gradient step. "
    + "The default value is defined automatically for Logloss, MultiClass & RMSE loss functions depending on "
    + " the number of iterations if none of 'leaf_estimation_iterations', leaf_estimation_method', "
    + "'l2_leaf_reg' is set. In this case, the selected learning rate is printed to stdout and saved in the "
    + "model. In other cases, the default value is 0.03."
  )

  @ParamGetterSetter
  final val randomSeed: IntParam = new IntParam(
    this,
    "randomSeed",
    "The random seed used for training. Default value is 0."
  )

  @ParamGetterSetter
  final val l2LeafReg: FloatParam = new FloatParam(
    this,
    "l2LeafReg",
    "Coefficient at the L2 regularization term of the cost function. Any positive value is allowed. "
    + "Default value is 3.0."
  )

  @ParamGetterSetter
  final val bootstrapType: EnumParam[EBootstrapType] = new EnumParam[EBootstrapType](
    this,
    "bootstrapType",
    "Bootstrap type. Defines the method for sampling the weights of objects."
    + "The default value depends on the selected mode and processing unit type: "
    + "QueryCrossEntropy, YetiRankPairwise, PairLogitPairwise: Bernoulli with the subsample parameter set to 0.5."
    + " MultiClass and MultiClassOneVsAll: Bayesian."
    + " Other modes: MVS with the subsample parameter set to 0.8."
  )

  @ParamGetterSetter
  final val baggingTemperature: FloatParam = new FloatParam(
    this,
    "baggingTemperature",
    "This parameter can be used if the selected bootstrap type is Bayesian. "
    + "Possible values are in the range [0, +inf). The higher the value the more aggressive the bagging is."
    + "Default value in 1.0."
  )

  @ParamGetterSetter
  final val subsample: FloatParam = new FloatParam(
    this,
    "subsample",
    "Sample rate for bagging. "
    + "The default value depends on the dataset size and the bootstrap type, see documentation for details."
  )

  @ParamGetterSetter
  final val samplingFrequency: EnumParam[ESamplingFrequency] = new EnumParam[ESamplingFrequency](
    this,
    "samplingFrequency",
    "Frequency to sample weights and objects when building trees. "
    + "Default value is 'PerTreeLevel'"
  )

  @ParamGetterSetter
  final val samplingUnit: EnumParam[ESamplingUnit] = new EnumParam[ESamplingUnit](
    this,
    "samplingUnit",
    "The sampling scheme, see documentation for details. "
    + "Default value is 'Object'"
  )

  @ParamGetterSetter
  final val mvsReg: FloatParam = new FloatParam(
    this,
    "mvsReg",
    "Affects the weight of the denominator and can be used for balancing between the importance and "
    + "Bernoulli sampling (setting it to 0 implies importance sampling and to +Inf - Bernoulli)."
    + "Note: This parameter is supported only for the MVS sampling method."
  )

  @ParamGetterSetter
  final val randomStrength: FloatParam = new FloatParam(
    this,
    "randomStrength",
    "The amount of randomness to use for scoring splits when the tree structure is selected. Use this "
    + "parameter to avoid overfitting the model. See documentation for details. "
    + "Default value is 1.0"
  )

  @ParamGetterSetter
  final val useBestModel: BooleanParam = new BooleanParam(
    this,
    "useBestModel",
    "If this parameter is set, the number of trees that are saved in the resulting model is selected based"
    + " on the optimal value of the evalMetric. This option requires a validation dataset to be provided."
  )

  @ParamGetterSetter
  final val bestModelMinTrees: IntParam = new IntParam(
    this,
    "bestModelMinTrees",
    "The minimal number of trees that the best model should have. If set, the output model contains at least "
    + " the given number of trees even if the best model is located within these trees."
    + " Should be used with the useBestModel parameter."
    + " No limit by default."
  )

  @ParamGetterSetter
  final val depth: IntParam = new IntParam(
    this,
    "depth",
    "Depth of the trees."
    + "Default value is 6."
  )

  @ParamGetterSetter
  final val oneHotMaxSize: IntParam = new IntParam(
    this,
    "oneHotMaxSize",
    "Use one-hot encoding for all categorical features with a number of different values less than or equal "
    + "to the given parameter value. Ctrs are not calculated for such features."
  )

  @ParamGetterSetter
  final val hasTime: BooleanParam = new BooleanParam(
    this,
    "hasTime",
    "Use the order of objects in the input data (do not perform random permutations during Choosing the tree "
    + "structure stage)."
  )

  @ParamGetterSetter
  final val rsm: FloatParam = new FloatParam(
    this,
    "rsm",
    "Random subspace method. The percentage of features to use at each split selection, when features are "
    + "selected over again at random. "
    + "The value must be in the range (0;1]. Default value is 1."
  )

  @ParamGetterSetter
  final val foldPermutationBlock: IntParam = new IntParam(
    this,
    "foldPermutationBlock",
    "Objects in the dataset are grouped in blocks before the random permutations. This parameter defines the "
    + "size of the blocks. The smaller is the value, the slower is the training. Large values may result in "
    + "quality degradation. "
    + "Default value is 1."
  )

  @ParamGetterSetter
  final val leafEstimationMethod: EnumParam[ELeavesEstimation] = new EnumParam[ELeavesEstimation](
    this,
    "leafEstimationMethod",
    "The method used to calculate the values in leaves. See documentation for details."
  )

  @ParamGetterSetter
  final val leafEstimationIterations: IntParam = new IntParam(
    this,
    "leafEstimationIterations",
    "CatBoost might calculate leaf values using several gradient or newton steps instead of a single one. "
    + "This parameter regulates how many steps are done in every tree when calculating leaf values."
  )

  @ParamGetterSetter
  final val leafEstimationBacktracking: EnumParam[ELeavesEstimationStepBacktracking]
    = new EnumParam[ELeavesEstimationStepBacktracking](
      this,
      "leafEstimationBacktracking",
      "When the value of the leafEstimationIterations parameter is greater than 1, CatBoost makes several "
      + "gradient or newton steps when calculating the resulting leaf values of a tree. "
      + "The behaviour differs depending on the value of this parameter. See documentation for details. "
      + "Default value is 'AnyImprovement'"
    )

  @ParamGetterSetter
  final val foldLenMultiplier: FloatParam = new FloatParam(
    this,
    "foldLenMultiplier",
    "Coefficient for changing the length of folds. The value must be greater than 1. The best validation "
    + "result is achieved with minimum values. "
    + "Default value is 2.0."
  )

  @ParamGetterSetter
  final val approxOnFullHistory: BooleanParam = new BooleanParam(
    this,
    "approxOnFullHistory",
    "Use all the preceding rows in the fold for calculating the approximated values. This mode is slower and "
    + "in rare cases slightly more accurate."
  )

  @ParamGetterSetter
  final val diffusionTemperature: FloatParam = new FloatParam(
    this,
    "diffusionTemperature",
    "The diffusion temperature of the Stochastic Gradient Langevin Boosting mode. "
    + "Only non-negative values are supported. Default value is 10000."
  )

  @ParamGetterSetter
  final val allowConstLabel: BooleanParam = new BooleanParam(
    this,
    "allowConstLabel",
    "Use it to train models with datasets that have equal label values for all objects."
  )

  @ParamGetterSetter
  final val scoreFunction: EnumParam[EScoreFunction] = new EnumParam[EScoreFunction](
    this,
    "scoreFunction",
    "The score type used to select the next split during the tree construction. See documentation for details. "
    + "Default value is 'Cosine'"
  )

  @ParamGetterSetter
  final val featureWeightsMap: OrderedStringMapParam[Double] = new OrderedStringMapParam[Double](
    this,
    "featureWeightsMap",
    "Per-feature multiplication weights used when choosing the best split. Map is 'feature_name' -> weight. "
    + "The score of each candidate is multiplied by the weights of features from the current split."
    + "This parameter is mutually exclusive with featureWeightsList."
  )

  @ParamGetterSetter
  final val featureWeightsList: DoubleArrayParam = new DoubleArrayParam(
    this,
    "featureWeightsList",
    "Per-feature multiplication weights used when choosing the best split. Array indices correspond to "
    + "feature indices. The score of each candidate is multiplied by the weights of features from the current "
    + "split."
    + "This parameter is mutually exclusive with featureWeightsMap."
  )

  @ParamGetterSetter
  final val firstFeatureUsePenaltiesMap: OrderedStringMapParam[Double] = new OrderedStringMapParam[Double](
    this,
    "firstFeatureUsePenaltiesMap",
    "Per-feature penalties for the first occurrence of the feature in the model. The given value is "
    + "subtracted from the score if the current candidate is the first one to include the feature in the "
    + "model. Map is 'feature_name' -> penalty. See documentation for details. "
    + "This parameter is mutually exclusive with firstFeatureUsePenaltiesList."
  )

  @ParamGetterSetter
  final val firstFeatureUsePenaltiesList: DoubleArrayParam = new DoubleArrayParam(
    this,
    "firstFeatureUsePenaltiesList",
    "Per-feature penalties for the first occurrence of the feature in the model. The given value is "
    + "subtracted from the score if the current candidate is the first one to include the feature in the "
    + "model. Array indices correspond to feature indices. See documentation for details. "
    + "This parameter is mutually exclusive with firstFeatureUsePenaltiesMap."
  )

  @ParamGetterSetter
  final val penaltiesCoefficient: FloatParam = new FloatParam(
    this,
    "penaltiesCoefficient",
    "A single-value common coefficient to multiply all penalties. Non-negative values are supported. "
    + "Default value is 1.0."
  )

  @ParamGetterSetter
  final val perObjectFeaturePenaltiesMap: OrderedStringMapParam[Double] = new OrderedStringMapParam[Double](
    this,
    "perObjectFeaturePenaltiesMap",
    "Per-object penalties for the first use of the feature for the object. The given value is multiplied by "
    + "the number of objects that are divided by the current split and use the feature for the first time. "
    + "Map is 'feature_name' -> penalty. See documentation for details. "
    + "This parameter is mutually exclusive with perObjectFeaturePenaltiesList."
  )

  @ParamGetterSetter
  final val perObjectFeaturePenaltiesList: DoubleArrayParam = new DoubleArrayParam(
    this,
    "perObjectFeaturePenaltiesList",
    "Per-object penalties for the first use of the feature for the object. The given value is multiplied by "
    + "the number of objects that are divided by the current split and use the feature for the first time. "
    + "Array indices correspond to feature indices. See documentation for details. "
    + "This parameter is mutually exclusive with perObjectFeaturePenaltiesMap."
  )

  @ParamGetterSetter
  final val modelShrinkRate: FloatParam = new FloatParam(
    this,
    "modelShrinkRate",
    "The constant used to calculate the coefficient for multiplying the model on each iteration. "
    + "See documentation for details."
  )

  @ParamGetterSetter
  final val modelShrinkMode: EnumParam[EModelShrinkMode] = new EnumParam[EModelShrinkMode](
    this,
    "modelShrinkMode",
    "Determines how the actual model shrinkage coefficient is calculated at each iteration. See "
    + "documentation for details. "
    + "Default value is 'Constant'"
  )


  // Overfitting detection settings

  @ParamGetterSetter
  final val earlyStoppingRounds: IntParam = new IntParam(
    this,
    "earlyStoppingRounds",
    "Sets the overfitting detector type to Iter and stops the training after the specified number of "
    + "iterations since the iteration with the optimal metric value."
  )

  @ParamGetterSetter
  final val odType: EnumParam[EOverfittingDetectorType] = new EnumParam[EOverfittingDetectorType](
    this,
    "odType",
    "The type of the overfitting detector to use. See documentation for details. "
    + "Default value is 'IncToDec'"
  )

  @ParamGetterSetter
  final val odPval: FloatParam = new FloatParam(
    this,
    "odPval",
    "The threshold for the IncToDec overfitting detector type. The training is stopped when the specified "
    + "value is reached. Requires that a validation dataset was input. See documentation for details."
    + "Turned off by default."
  )

  @ParamGetterSetter
  final val odWait: IntParam = new IntParam(
    this,
    "odWait",
    "The number of iterations to continue the training after the iteration with the optimal metric value. "
    + "See documentation for details. "
    + "Default value is 20."
  )


  // Output settings

  @ParamGetterSetter
  final val loggingLevel: EnumParam[ELoggingLevel] = new EnumParam[ELoggingLevel](
    this,
    "loggingLevel",
    "The logging level to output to stdout. See documentation for details. "
    + "Default value is 'Verbose'"
  )

  @ParamGetterSetter
  final val metricPeriod: IntParam = new IntParam(
    this,
    "metricPeriod",
    "The frequency of iterations to calculate the values of objectives and metrics. The value should be a "
    + " positive integer. The usage of this parameter speeds up the training. "
    + "Default value is 1."
  )

  @ParamGetterSetter
  final val trainDir: Param[String] = new Param[String](
    this,
    "trainDir",
    "The directory for storing the files on Driver node generated during training. "
    + "Default value is 'catboost_info'"
  )

  @ParamGetterSetter
  final val allowWritingFiles: BooleanParam = new BooleanParam(
    this,
    "allowWritingFiles",
    "Allow to write analytical and snapshot files during training. "
    + "Enabled by default."
  )

  @ParamGetterSetter
  final val saveSnapshot: BooleanParam = new BooleanParam(
    this,
    "saveSnapshot",
    "Enable snapshotting for restoring the training progress after an interruption. If enabled, the default "
    + " period for making snapshots is 600 seconds. Use the snapshotInterval parameter to change this period."
  )

  @ParamGetterSetter
  final val snapshotFile: Param[String] = new Param[String](
    this,
    "snapshotFile",
    "The name of the file to save the training progress information in. This file is used for recovering "
    + "training after an interruption."
  )

  @ParamGetterSetter
  final val snapshotInterval: DurationParam = new DurationParam(
    this,
    "snapshotInterval",
    "The interval between saving snapshots. See documentation for details. "
    + "Default value is 600 seconds."
  )
}


/** Params for training [[CatBoostClassifier]]. See documentation on [[https://catboost.ai/docs/]]
 *  for details.
 */
trait ClassifierTrainingParamsTrait extends TrainingParamsTrait {
  @ParamGetterSetter
  final val classWeightsMap: OrderedStringMapParam[Double] = new OrderedStringMapParam[Double](
    this,
    "classWeightsMap",
    "Map from class name to weight. The values are used as multipliers for the object weights. "
    + " This parameter is mutually exclusive with classWeightsList."
  )

  @ParamGetterSetter
  final val classWeightsList: DoubleArrayParam = new DoubleArrayParam(
    this,
    "classWeightsList",
    "List of weights for each class. The values are used as multipliers for the object weights. "
    + " This parameter is mutually exclusive with classWeightsMap."
  )

  @ParamGetterSetter
  final val classNames: StringArrayParam = new StringArrayParam(
    this,
    "classNames",
    "Allows to redefine the default values (consecutive integers)."
  )

  @ParamGetterSetter
  final val autoClassWeights: EnumParam[EAutoClassWeightsType] = new EnumParam[EAutoClassWeightsType](
    this,
    "autoClassWeights",
    "Automatically calculate class weights based either on the total weight or the total number of objects in"
    + " each class. The values are used as multipliers for the object weights. "
    + "Default value is 'None'"
  )

  @ParamGetterSetter
  final val scalePosWeight: FloatParam = new FloatParam(
    this,
    "scalePosWeight",
    "The weight for class 1 in binary classification. The value is used as a multiplier for the weights of "
    + "objects from class 1. "
    + "Default value is 1 (both classes have equal weight)."
  )

  @ParamGetterSetter
  final val classesCount: IntParam = new IntParam(
    this,
    "classesCount",
    "The upper limit for the numeric class label. Defines the number of classes for multiclassification. "
    + "See documentation for details."
  )


  // Target quantization settings

  @ParamGetterSetter
  final val targetBorder: FloatParam = new FloatParam(
    this,
    "targetBorder",
    "If set, defines the border for converting target values to 0 and 1 classes."
  )
}


/** Params for training [[CatBoostRegressor]]. See documentation at [[https://catboost.ai/docs/]]
 *  for details.
 */
trait RegressorTrainingParamsTrait extends TrainingParamsTrait {

}
