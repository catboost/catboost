package ai.catboost.spark

import collection.mutable

import org.json4s.JObject

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel,ProbabilisticClassifier}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import ai.catboost.spark.impl.CtrsContext
import ai.catboost.spark.params._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl


/** Classification model trained by CatBoost. Use [[CatBoostClassifier]] to train it
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
 *   val classifier = new CatBoostClassifier
 *   val model = classifier.fit(trainPool)
 *   val path = "/home/user/catboost_spark_models/model0"
 *   model.write.save(path)
 * }}}
 *
 * @example Load model
 * {{{
 *   val dataFrameForPrediction : DataFrame = ... init DataFrame ...
 *   val path = "/home/user/catboost_spark_models/model0"
 *   val model = CatBoostClassificationModel.load(path)
 *   val predictions = model.transform(dataFrameForPrediction)
 *   predictions.show()
 * }}}
 *
 * @example Save as a native model
 * {{{
 *   val trainPool : Pool = ... init Pool ...
 *   val classifier = new CatBoostClassifier
 *   val model = classifier.fit(trainPool)
 *   val path = "/home/user/catboost_native_models/model0.cbm"
 *   model.saveNativeModel(path)
 * }}}
 *
 * @example Load native model
 * {{{
 *   val dataFrameForPrediction : DataFrame = ... init DataFrame ...
 *   val path = "/home/user/catboost_native_models/model0.cbm"
 *   val model = CatBoostClassificationModel.loadNativeModel(path)
 *   val predictions = model.transform(dataFrameForPrediction)
 *   predictions.show()
 * }}}
 */
class CatBoostClassificationModel (
  override val uid: String,
  private[spark] var nativeModel: native_impl.TFullModel = null,
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

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".transform() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    transformImpl(dataset)
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

  protected def getAdditionalColumnsForApply : Seq[StructField] = {
    var result = Seq[StructField]()
    if ($(rawPredictionCol).nonEmpty) {
      result = result :+ StructField($(rawPredictionCol), SQLDataTypes.VectorType)
    }
    if ($(probabilityCol).nonEmpty) {
      result = result :+ StructField($(probabilityCol), SQLDataTypes.VectorType)
    }
    if ($(predictionCol).nonEmpty) {
      result = result :+ StructField($(predictionCol), DoubleType)
    }
    result
  }

  protected def getResultIteratorForApply(
    objectsDataProvider: native_impl.SWIGTYPE_p_NCB__TObjectsDataProviderPtr,
    dstRows: mutable.ArrayBuffer[Array[Any]], // guaranteed to be non-empty
    localExecutor: native_impl.TLocalExecutor
  ) : Iterator[Row] = {
    val applyResultIterator = new native_impl.TApplyResultIterator(
      nativeModel,
      objectsDataProvider,
      native_impl.EPredictionType.RawFormulaVal,
      localExecutor
    )

    val rowLength = dstRows(0).length
    val numClassesValue = numClasses

    val callback = if ($(rawPredictionCol).nonEmpty && !$(probabilityCol).nonEmpty && !$(predictionCol).nonEmpty) {
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)
        rowArray(rowLength - 1) = Vectors.dense(rawPrediction)
        rowArray
      }
    } else if (!$(rawPredictionCol).nonEmpty && $(probabilityCol).nonEmpty && !$(predictionCol).nonEmpty) {
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)
        val probability = new Array[Double](numClassesValue)
        native_impl.native_impl.CalcSoftmax(rawPrediction, probability)
        rowArray(rowLength - 1) = Vectors.dense(probability)
        rowArray
      }
    } else if ($(rawPredictionCol).nonEmpty && $(probabilityCol).nonEmpty && !$(predictionCol).nonEmpty) {
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)
        rowArray(rowLength - 2) = Vectors.dense(rawPrediction)

        val probability = new Array[Double](numClassesValue)
        native_impl.native_impl.CalcSoftmax(rawPrediction, probability)
        rowArray(rowLength - 1) = Vectors.dense(probability)

        rowArray
      }
    } else if (!$(rawPredictionCol).nonEmpty && !$(probabilityCol).nonEmpty && $(predictionCol).nonEmpty) {
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)
        rowArray(rowLength - 1) = raw2prediction(Vectors.dense(rawPrediction))
        rowArray
      }
    } else if ($(rawPredictionCol).nonEmpty && !$(probabilityCol).nonEmpty && $(predictionCol).nonEmpty) {
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)
        val rawPredictionVector = Vectors.dense(rawPrediction)
        rowArray(rowLength - 2) = rawPredictionVector
        rowArray(rowLength - 1) = raw2prediction(rawPredictionVector)
        rowArray
      }
    } else if (!$(rawPredictionCol).nonEmpty && $(probabilityCol).nonEmpty && $(predictionCol).nonEmpty) {
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)

        val probability = new Array[Double](numClassesValue)
        native_impl.native_impl.CalcSoftmax(rawPrediction, probability)

        val probabilityVector = Vectors.dense(probability)
        rowArray(rowLength - 2) = probabilityVector
        rowArray(rowLength - 1) = probability2prediction(probabilityVector)
        rowArray
      }
    } else { // ($(rawPredictionCol).nonEmpty && $(probabilityCol).nonEmpty && $(predictionCol).nonEmpty)
      (rowArray: Array[Any], objectIdx: Int) => {
        val rawPrediction = new Array[Double](numClassesValue)
        applyResultIterator.GetMultiDimensionalResult(objectIdx, rawPrediction)
        rowArray(rowLength - 3) = Vectors.dense(rawPrediction)

        val probability = new Array[Double](numClassesValue)
        native_impl.native_impl.CalcSoftmax(rawPrediction, probability)
        val probabilityVector = Vectors.dense(probability)
        rowArray(rowLength - 2) = probabilityVector

        rowArray(rowLength - 1) = probability2prediction(probabilityVector)

        rowArray
      }
    }

    new ProcessRowsOutputIterator(dstRows, callback)
  }
}


object CatBoostClassificationModel extends MLReadable[CatBoostClassificationModel] {
  override def read: MLReader[CatBoostClassificationModel] = new CatBoostClassificationModelReader
  override def load(path: String): CatBoostClassificationModel = super.load(path)

  private class CatBoostClassificationModelReader
    extends MLReader[CatBoostClassificationModel] with CatBoostModelReaderTrait
  {
      override def load(path: String) : CatBoostClassificationModel = {
        val (uid, nativeModel) = loadImpl(
          super.sparkSession.sparkContext,
          classOf[CatBoostClassificationModel].getName,
          path
        )
        new CatBoostClassificationModel(uid, nativeModel, nativeModel.GetDimensionsCount.toInt)
      }
  }
  
  def loadNativeModel(
    fileName: String, 
    format: EModelType = native_impl.EModelType.CatboostBinary
  ): CatBoostClassificationModel = {
    new CatBoostClassificationModel(native_impl.native_impl.ReadModelWrapper(fileName, format))
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
 * ===Examples===
 * Binary classification.
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
 * Multiclassification.
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
 *
 * ==Serialization==
 * Supports standard Spark MLLib serialization. Data can be saved to distributed filesystem like HDFS or
 * local files.
 *
 * ===Examples==
 * Save:
 * {{{
 *   val classifier = new CatBoostClassifier().setIterations(100)
 *   val path = "/home/user/catboost_classifiers/classifier0"
 *   classifier.write.save(path)
 * }}}
 *
 * Load:
 * {{{
 *   val path = "/home/user/catboost_classifiers/classifier0"
 *   val classifier = CatBoostClassifier.load(path)
 *   val trainPool : Pool = ... init Pool ...
 *   val model = classifier.fit(trainPool)
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
  ) : (Pool, Array[Pool], JObject, CtrsContext) = {
    val classNamesFromLabelData = if (ai.catboost.spark.params.Helpers.classNamesAreKnown(this)) {
      None
    } else {
      Some(DataHelpers.getClassNamesFromLabelData(quantizedTrainPool.data, getLabelCol))
    }

    if (!isDefined(lossFunction)) {
      if (isDefined(targetBorder)) {
        set(lossFunction, "Logloss")
      } else {
        val distinctLabelValuesCount = if (classNamesFromLabelData.isDefined) {
          classNamesFromLabelData.get.length.toLong
        } else {
          quantizedTrainPool.data.select(getLabelCol).distinct.count
        }
        set(lossFunction, if (distinctLabelValuesCount > 2) "MultiClass" else "Logloss")
      }
      log.info(s"lossFunction has been inferred as '${this.getLossFunction}'")
    }

    val catBoostJsonParams = ai.catboost.spark.params.Helpers.sparkMlParamsToCatBoostJsonParams(
      this, 
      classNamesFromLabelData
    )
    val (preprocessedTrainPool, preprocessedEvalPools, ctrsContext) 
        = addEstimatedCtrFeatures(quantizedTrainPool, quantizedEvalPools, catBoostJsonParams)
    (
      preprocessedTrainPool, 
      preprocessedEvalPools, 
      catBoostJsonParams, 
      ctrsContext
    )
  }

  protected override def createModel(nativeModel: native_impl.TFullModel): CatBoostClassificationModel = {
    new CatBoostClassificationModel(nativeModel)
  }
}

object CatBoostClassifier extends DefaultParamsReadable[CatBoostClassifier] {
  override def load(path: String): CatBoostClassifier = super.load(path)
}

