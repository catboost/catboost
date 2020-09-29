package ai.catboost.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util.Identifiable

import org.apache.spark.ml.CatBoostRegressorBase // defined inside catboost4j-spark

import ai.catboost.spark.params._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl


/** Regression model trained by CatBoost. Use [[CatBoostRegressor]] to train it */
class CatBoostRegressionModel (
  override val uid: String,
  protected var nativeModel : native_impl.TFullModel = null,
  protected var nativeDimension: Int
)
  extends RegressionModel[Vector, CatBoostRegressionModel]
    with CatBoostModelTrait[CatBoostRegressionModel]
{
  def this(nativeModel : native_impl.TFullModel) = this(
    Identifiable.randomUID("CatBoostRegressionModel"),
    nativeModel,
    nativeDimension = 1 // always for regression
  )

  override def copy(extra: ParamMap): CatBoostRegressionModel = {
    val newModel = defaultCopy[CatBoostRegressionModel](extra)
    newModel.nativeModel = this.nativeModel
    newModel.nativeDimension = this.nativeDimension
    newModel
  }

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  override def predict(features: Vector): Double = {
    predictRawImpl(features)(0)
  }
}


/** Class to train [[CatBoostRegressionModel]]
 *   The default optimized loss function is `RMSE`
 *
 * @example Basic example.
 * {{{
 *  val spark = SparkSession.builder()
 *    .master("local[*]")
 *    .appName("RegressorTest")
 *    .getOrCreate();
 *
 *  val srcDataSchema = Seq(
 *    StructField("features", SQLDataTypes.VectorType),
 *    StructField("label", StringType)
 *  )
 *
 *  val trainData = Seq(
 *    Row(Vectors.dense(0.1, 0.2, 0.11), "0.12"),
 *    Row(Vectors.dense(0.97, 0.82, 0.33), "0.22"),
 *    Row(Vectors.dense(0.13, 0.22, 0.23), "0.34"),
 *    Row(Vectors.dense(0.8, 0.62, 0.0), "0.1")
 *  )
 *
 *  val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
 *  val trainPool = new Pool(trainDf)
 *
 *  val evalData = Seq(
 *    Row(Vectors.dense(0.22, 0.33, 0.9), "0.1"),
 *    Row(Vectors.dense(0.11, 0.1, 0.21), "0.9"),
 *    Row(Vectors.dense(0.77, 0.0, 0.0), "0.72")
 *  )
 *
 *  val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
 *  val evalPool = new Pool(evalDf)
 *
 *  val regressor = new CatBoostRegressor
 *  val model = regressor.fit(trainPool, Array[Pool](evalPool))
 *  val predictions = model.transform(evalPool.data)
 *  predictions.show()
 * }}}
 *
 * @example Example with alternative loss function.
 * {{{
 *  ...<initialize trainPool, evalPool>
 *  val regressor = new CatBoostRegressor().setLossFunction("MAE")
 *  val model = regressor.fit(trainPool, Array[Pool](evalPool))
 *  val predictions = model.transform(evalPool.data)
 *  predictions.show()
 * }}}
 */
class CatBoostRegressor (override val uid: String)
  extends CatBoostRegressorBase[Vector, CatBoostRegressor, CatBoostRegressionModel]
    with CatBoostPredictorTrait[CatBoostRegressor, CatBoostRegressionModel]
    with RegressorTrainingParamsTrait
{
  def this() = this(Identifiable.randomUID("CatBoostRegressor"))

  override def copy(extra: ParamMap): CatBoostRegressor = defaultCopy(extra)

  protected override def createModel(nativeModel: native_impl.TFullModel): CatBoostRegressionModel = {
    new CatBoostRegressionModel(nativeModel)
  }
}

