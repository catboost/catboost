package ai.catboost.spark

import collection.mutable

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import org.apache.spark.ml.CatBoostRegressorBase // defined inside catboost4j-spark

import ai.catboost.spark.params._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl


/** Regression model trained by CatBoost. Use [[CatBoostRegressor]] to train it
 *
 * ==Serialization==
 * Supports standard Spark MLLib serialization. Data can be saved to distributed filesystem like HDFS or
 * local files.
 * When saved to `path` two files are created:
 *   -`<path>/metadata` which contains Spark-specific metadata in JSON format
 *   -`<path>/model` which contains model in usual CatBoost format which can be read using other local
 *     CatBoost APIs (if stored in a distributed filesystem it has to be copied to the local filesystem first).
 *
 * Saving to and loading from local files in standard CatBoost model formats is also supported.
 *
 * @example Save model
 * {{{
 *   val trainPool : Pool = ... init Pool ...
 *   val regressor = new CatBoostRegressor
 *   val model = regressor.fit(trainPool)
 *   val path = "/home/user/catboost_spark_models/model0"
 *   model.write.save(path)
 * }}}
 *
 * @example Load model
 * {{{
 *   val dataFrameForPrediction : DataFrame = ... init DataFrame ...
 *   val path = "/home/user/catboost_spark_models/model0"
 *   val model = CatBoostRegressionModel.load(path)
 *   val predictions = model.transform(dataFrameForPrediction)
 *   predictions.show()
 * }}}
 *
 * @example Save as a native model
 * {{{
 *   val trainPool : Pool = ... init Pool ...
 *   val regressor = new CatBoostRegressor
 *   val model = regressor.fit(trainPool)
 *   val path = "/home/user/catboost_native_models/model0.cbm"
 *   model.saveNativeModel(path)
 * }}}
 *
 * @example Load native model
 * {{{
 *   val dataFrameForPrediction : DataFrame = ... init DataFrame ...
 *   val path = "/home/user/catboost_native_models/model0.cbm"
 *   val model = CatBoostRegressionModel.loadNativeModel(path)
 *   val predictions = model.transform(dataFrameForPrediction)
 *   predictions.show()
 * }}}
 */
class CatBoostRegressionModel (
  override val uid: String,
  private[spark] var nativeModel : native_impl.TFullModel = null,
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
    val that = new CatBoostRegressionModel(this.uid, this.nativeModel, this.nativeDimension)
    this.copyValues(that, extra).asInstanceOf[CatBoostRegressionModel]
  }

  override def transformImpl(dataset: Dataset[_]): DataFrame = {
    transformCatBoostImpl(dataset)
  }

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  override def predict(features: Vector): Double = {
    predictRawImpl(features)(0)
  }

  protected override def getAdditionalColumnsForApply : Seq[StructField] = {
    Seq(StructField($(predictionCol), DoubleType))
  }

  protected override def getResultIteratorForApply(
    objectsDataProvider: native_impl.SWIGTYPE_p_NCB__TObjectsDataProviderPtr,
    dstRows: mutable.ArrayBuffer[Array[Any]], // guaranteed to be non-empty
    localExecutor: native_impl.TLocalExecutor
  ) : Iterator[Row] = {
    val applyResults = new native_impl.TApplyResultIterator(
      nativeModel,
      objectsDataProvider,
      native_impl.EPredictionType.RawFormulaVal,
      localExecutor
    ).GetSingleDimensionalResults.toPrimitiveArray

    val applyResultRowIdx = dstRows(0).length - 1
    new ProcessRowsOutputIterator(
      dstRows,
      (rowArray: Array[Any], objectIdx: Int) => {
        rowArray(applyResultRowIdx) = applyResults(objectIdx)
        rowArray
      }
    )
  }
}

object CatBoostRegressionModel extends MLReadable[CatBoostRegressionModel] {
  override def read: MLReader[CatBoostRegressionModel] = new CatBoostRegressionModelReader
  override def load(path: String): CatBoostRegressionModel = super.load(path)

  private class CatBoostRegressionModelReader
    extends MLReader[CatBoostRegressionModel] with CatBoostModelReaderTrait
  {
      override def load(path: String) : CatBoostRegressionModel = {
        val (uid, nativeModel) = loadImpl(
          super.sparkSession.sparkContext,
          classOf[CatBoostRegressionModel].getName,
          path
        )
        new CatBoostRegressionModel(uid, nativeModel, 1)
      }
  }

  def loadNativeModel(
    fileName: String,
    format: EModelType = native_impl.EModelType.CatboostBinary
  ): CatBoostRegressionModel = {
    new CatBoostRegressionModel(native_impl.native_impl.ReadModel(fileName, format))
  }

  def sum(
    models: Array[CatBoostRegressionModel],
    weights: Array[Double] = null,
    ctrMergePolicy: ECtrTableMergePolicy = native_impl.ECtrTableMergePolicy.IntersectingCountersAverage
  ): CatBoostRegressionModel = {
    new CatBoostRegressionModel(CatBoostModel.sum(models.toArray[CatBoostModelTrait[CatBoostRegressionModel]], weights, ctrMergePolicy))
  }
}


/** Class to train [[CatBoostRegressionModel]]
 *   The default optimized loss function is `RMSE`
 *
 * ===Examples===
 * Basic example.
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
 * Example with alternative loss function.
 * {{{
 *  ...<initialize trainPool, evalPool>
 *  val regressor = new CatBoostRegressor().setLossFunction("MAE")
 *  val model = regressor.fit(trainPool, Array[Pool](evalPool))
 *  val predictions = model.transform(evalPool.data)
 *  predictions.show()
 * }}}
 *
 * ==Serialization==
 * Supports standard Spark MLLib serialization. Data can be saved to distributed filesystem like HDFS or
 * local files.
 *
 * ===Examples:===
 * Save:
 * {{{
 *   val regressor = new CatBoostRegressor().setLossFunction("MAE")
 *   val path = "/home/user/catboost_regressors/regressor0"
 *   regressor.write.save(path)
 * }}}
 *
 * Load:
 * {{{
 *   val path = "/home/user/catboost_regressors/regressor0"
 *   val regressor = CatBoostRegressor.load(path)
 *   val trainPool : Pool = ... init Pool ...
 *   val model = regressor.fit(trainPool)
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

object CatBoostRegressor extends DefaultParamsReadable[CatBoostRegressor] {
  override def load(path: String): CatBoostRegressor = super.load(path)
}

