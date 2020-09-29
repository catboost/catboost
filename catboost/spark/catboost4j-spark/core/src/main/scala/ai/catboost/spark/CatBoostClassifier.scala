package ai.catboost.spark

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel,ProbabilisticClassifier}
import org.apache.spark.ml.util.Identifiable

import ai.catboost.spark.params._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl


/** Classification model trained by CatBoost. Use [[CatBoostClassifier]] to train it */
class CatBoostClassificationModel (
  override val uid: String,
  protected var nativeModel: native_impl.TFullModel = null,
  protected var nativeDimension: Int
)
  extends ProbabilisticClassificationModel[Vector, CatBoostClassificationModel]
    with CatBoostModelTrait[CatBoostClassificationModel]
{
  def this(nativeModel : native_impl.TFullModel) = this(
    Identifiable.randomUID("CatBoostClassificationModel"),
    nativeModel,
    nativeModel.GetDimensionsCount.toInt
  )

  override def copy(extra: ParamMap): CatBoostClassificationModel = {
    val newModel = defaultCopy[CatBoostClassificationModel](extra)
    newModel.nativeModel = this.nativeModel
    newModel.nativeDimension = this.nativeDimension
    newModel
  }

  override def numClasses: Int = {
    if (nativeDimension == 1) 2 else nativeDimension
  }

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  override protected def predictRaw(features: Vector): Vector = {
    val nativePredictions = predictRawImpl(features)
    if (nativeDimension == 1) {
      Vectors.dense(-nativePredictions(0), nativePredictions(0))
    } else {
      Vectors.dense(nativePredictions)
    }
  }

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    val result = new Array[Double](rawPrediction.size)
    native_impl.native_impl.CalcSoftmax(rawPrediction.toDense.values, result)
    Vectors.dense(result)
  }
}


/** Class to train [[CatBoostClassificationModel]]
 *
 *  The default optimized loss function depends on various conditions:
 *
 *   - `Logloss` — The label column has only two different values or the targetBorder parameter is specified.
 *   - `MultiClass` — The label column has more than two different values and the targetBorder parameter is
 *     not specified.
 *
 * @example Binary classification.
 * {{{
 *  val spark = SparkSession.builder()
 *    .master("local[*]")
 *    .appName("ClassifierTest")
 *    .getOrCreate();
 *
 *  val srcDataSchema = Seq(
 *    StructField("features", SQLDataTypes.VectorType),
 *    StructField("label", StringType)
 *  )
 *
 *  val trainData = Seq(
 *    Row(Vectors.dense(0.1, 0.2, 0.11), "0"),
 *    Row(Vectors.dense(0.97, 0.82, 0.33), "1"),
 *    Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
 *    Row(Vectors.dense(0.8, 0.62, 0.0), "0")
 *  )
 *
 *  val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
 *  val trainPool = new Pool(trainDf)
 *
 *  val evalData = Seq(
 *    Row(Vectors.dense(0.22, 0.33, 0.9), "1"),
 *    Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
 *    Row(Vectors.dense(0.77, 0.0, 0.0), "1")
 *  )
 *
 *  val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
 *  val evalPool = new Pool(evalDf)
 *
 *  val classifier = new CatBoostClassifier
 *  val model = classifier.fit(trainPool, Array[Pool](evalPool))
 *  val predictions = model.transform(evalPool.data)
 *  predictions.show()
 * }}}
 *
 * @example Multiclassification.
 * {{{
 *  val spark = SparkSession.builder()
 *    .master("local[*]")
 *    .appName("ClassifierTest")
 *    .getOrCreate();
 *
 *  val srcDataSchema = Seq(
 *    StructField("features", SQLDataTypes.VectorType),
 *    StructField("label", StringType)
 *  )
 *
 *  val trainData = Seq(
 *    Row(Vectors.dense(0.1, 0.2, 0.11), "1"),
 *    Row(Vectors.dense(0.97, 0.82, 0.33), "2"),
 *    Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
 *    Row(Vectors.dense(0.8, 0.62, 0.0), "0")
 *  )
 *
 *  val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
 *  val trainPool = new Pool(trainDf)
 *
 *  val evalData = Seq(
 *    Row(Vectors.dense(0.22, 0.33, 0.9), "2"),
 *    Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
 *    Row(Vectors.dense(0.77, 0.0, 0.0), "1")
 *  )
 *
 *  val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
 *  val evalPool = new Pool(evalDf)
 *
 *  val classifier = new CatBoostClassifier
 *  val model = classifier.fit(trainPool, Array[Pool](evalPool))
 *  val predictions = model.transform(evalPool.data)
 *  predictions.show()
 * }}}
 */
class CatBoostClassifier (override val uid: String)
  extends ProbabilisticClassifier[Vector, CatBoostClassifier, CatBoostClassificationModel]
    with CatBoostPredictorTrait[CatBoostClassifier, CatBoostClassificationModel]
    with ClassifierTrainingParamsTrait
{
  def this() = this(Identifiable.randomUID("CatBoostClassifier"))

  override def copy(extra: ParamMap): CatBoostClassifier = defaultCopy(extra)

  protected override def preprocessBeforeTraining(
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool]
  ) : (Pool, Array[Pool]) = {
    if (!isDefined(lossFunction)) {
      if (isDefined(targetBorder)) {
        set(lossFunction, "Logloss")
      } else {
        val distinctLabelValuesCount = quantizedTrainPool.data.select(getLabelCol).distinct.count
        set(lossFunction, if (distinctLabelValuesCount > 2) "MultiClass" else "Logloss")
      }
    }

    (quantizedTrainPool, quantizedEvalPools)
  }

  protected override def createModel(nativeModel: native_impl.TFullModel): CatBoostClassificationModel = {
    new CatBoostClassificationModel(nativeModel)
  }
}
