package ai.catboost.spark

import scala.reflect.classTag

import collection.mutable
import collection.JavaConverters._

import org.apache.commons.lang3.tuple.Pair

import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.CatBoostError

import ai.catboost.spark.impl.RowEncoderConstructor


class FeatureImportance(
  val featureName: String = "",
  val importance: Double = 0.0
)

class FeatureInteractionScore(
  val firstFeatureIdx: Int = 0,
  val secondFeatureIdx: Int = 0,
  val score: Double = 0.0
)


private[spark] trait CatBoostModelTrait[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]]
  extends org.apache.spark.ml.PredictionModel[Vector, Model]
  with MLWritable
{
  private[spark] var nativeModel: TFullModel
  protected var nativeDimension: Int

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  final def predictRawImpl(features: Vector) : Array[Double] = {
    val result = new Array[Double](nativeDimension)
    features match {
      case denseFeatures: DenseVector => nativeModel.Calc(denseFeatures.values, result)
      case sparseFeatures: SparseVector =>
        nativeModel.CalcSparse(sparseFeatures.size, sparseFeatures.indices, sparseFeatures.values, result)
      case _ => throw new CatBoostError("Unknown Vector subtype")
    }
    result
  }

  protected def getAdditionalColumnsForApply : Seq[StructField]

  protected def getResultIteratorForApply(
    objectsDataProvider: SWIGTYPE_p_NCB__TObjectsDataProviderPtr,
    dstRows: mutable.ArrayBuffer[Array[Any]], // guaranteed to be non-empty
    localExecutor: TLocalExecutor
  ) : Iterator[Row]

  // cannot override transformImpl because it's made final in ClassificationModel since Spark 3.x+
  protected def transformCatBoostImpl(dataset: Dataset[_]): DataFrame = {
    val dataFrame = dataset.asInstanceOf[DataFrame]

    val featuresColumnIdx = dataset.schema.fieldIndex($(featuresCol));

    val additionalColumnsForApply = getAdditionalColumnsForApply
    if (additionalColumnsForApply.isEmpty) {
      this.logWarning(s"$uid: transform() was called as NOOP since no output columns were set.")
      dataFrame
    } else {
      val resultSchema = StructType(dataset.schema.toSeq ++ additionalColumnsForApply)

      val pool = new Pool(dataFrame).setFeaturesCol($(featuresCol))
      val featuresLayoutPtr = pool.getFeaturesLayout
      val maxUniqCatFeatureValues = Pool.getCatFeaturesMaxUniqValueCount(dataFrame, $(featuresCol))

      val dstRowLength = resultSchema.length
      val threadCountForTask = SparkHelpers.getThreadCountForTask(dataset.sparkSession)

      this.logInfo(s"$uid: schedule applying model.")

      // Cannot use mapPartitions directly with RowEncoder because it loses schema columns metadata
      val resultAsRDD = dataFrame.rdd.mapPartitions(
        rowsIterator => {
          if (rowsIterator.hasNext) {
            val localExecutor = new TLocalExecutor
            localExecutor.Init(threadCountForTask)

            val (dstRows, rawObjectsDataProviderPtr) = DataHelpers.processDatasetWithRawFeatures(
              rowsIterator,
              featuresColumnIdx,
              featuresLayoutPtr,
              maxUniqCatFeatureValues,
              keepRawFeaturesInDstRows = true,
              dstRowLength = dstRowLength,
              localExecutor = localExecutor
            )
            getResultIteratorForApply(rawObjectsDataProviderPtr.ToBase(), dstRows, localExecutor)
          } else {
            Iterator[Row]()
          }
        }
      )
      dataFrame.sparkSession.createDataFrame(resultAsRDD, resultSchema)
    }
  }

  /**
   * This function is useful when the dataset has been already quantized but works with any Pool
   */
  def transformPool(dataset: Pool) : DataFrame = {
    if (dataset.getFeaturesCol != this.getFeaturesCol) {
      throw new CatBoostError(
        s"model and dataset have incompatible features column names: ${this.getFeaturesCol} and "
        + dataset.getFeaturesCol
      )
    }
    if (dataset.isQuantized) {
      val additionalColumnsForApply = getAdditionalColumnsForApply
      if (additionalColumnsForApply.isEmpty) {
        this.logWarning(s"$uid: transform() was called as NOOP since no output columns were set.")
        dataset.data
      } else {
        val resultSchema = StructType(dataset.data.schema.toSeq ++ additionalColumnsForApply)

        dataset.mapQuantizedPartitions(
          selectedColumns=Seq(dataset.getFeaturesCol),
          includeEstimatedFeatures=false,
          includePairsIfPresent=false,
          dstColumnNames=null,
          dstRowLength=resultSchema.length,
          (
            dataProvider: TDataProviderPtr,
            estimatedDataProvider: TDataProviderPtr,
            dstRows: mutable.ArrayBuffer[Array[Any]],
            localExecutor: TLocalExecutor
          ) => {
            getResultIteratorForApply(
              dataProvider.GetQuantizedObjectsDataProvider().ToBase(),
              dstRows,
              localExecutor
            )
          }
        )(RowEncoderConstructor.construct(resultSchema), classTag[Row])
      }
    } else {
      transformImpl(dataset.data)
    }
  }


  override def write: MLWriter = new CatBoostModelWriter[Model](this)

  /**
   * Save the model to a local file.
   *
   * @param fileName The path to the output model.
   * @param format The output format of the model.
   *  Possible values:
   *
   *  <table>
   *  <tr>
   *  <td>CatboostBinary</td>
   *  <td>
   *  CatBoost binary format (default).
   *  </td>
   *  </tr>
   *  <tr>
   *  <td>AppleCoreML</td>
   *  <td>
   *  Apple CoreML format (only datasets without categorical features are currently supported).
   *  </td>
   *  </tr>
   *  <tr>
   *  <td>Cpp</td>
   *  <td>
   *  Standalone C++ code (multiclassification models are not currently supported).
   *  See the <a href="https://catboost.ai/docs/concepts/c-plus-plus-api_applycatboostmodel.html#c-plus-plus-api_applycatboostmodel">C++</a>
   *   section for details on applying the resulting model.
   *  </td>
   *  </tr>
   *  <tr>
   *  <td>Python</td>
   *  <td>
   *  Standalone Python code (multiclassification models are not currently supported).
   *  See the <a href="https://catboost.ai/docs/concepts/python-reference_apply_catboost_model.html#python-reference_apply_catboost_model">Python</a>
   *   section for details on applying the resulting model.
   *  </td>
   *  </tr>
   *  <tr>
   *  <td>Json</td>
   *  <td>
   *  JSON format. Refer to the <a href="https://github.com/catboost/tutorials/blob/master/model_analysis/model_export_as_json_tutorial.ipynb">CatBoost JSON model tutorial</a> for format details.
   *  </td>
   *  </tr>
   *  <tr>
   *  <td>Onnx</td>
   *  <td>
   *  ONNX-ML format (only datasets without categorical features are currently supported).
   *  Refer to <a href="https://onnx.ai">https://onnx.ai</a> for details.
   *  </td>
   *  </tr>
   *  <tr>
   *  <td>Pmml</td>
   *  <td>
   *  <a href="http://dmg.org/pmml/pmml-v4-3.html">PMML version 4.3</a> format.
   *   Categorical features must be interpreted as one-hot encoded during the training
   *   if present in the training dataset.  This can be accomplished by setting the --one-hot-max-size/one_hot_max_size parameter to a value that is greater
   *   than the maximum number of unique categorical feature values among all categorical features in the dataset.
   *   Note. Multiclassification models are not currently supported.
   *   See the <a href="https://catboost.ai/docs/concepts/apply-pmml.html">PMML</a> section for details on applying the resulting model.
   *  </td>
   *  </tr>
   *  </table>
   * @param exportParameters Additional format-dependent parameters for AppleCoreML, Onnx or Pmml formats.
   *  See <a href="https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html">python API documentation</a> for details.
   * @param pool The dataset previously used for training.
   *  This parameter is required if the model contains categorical features and the output format is Cpp, Python, or Json.
   *
   * @example
   * {{{
   * val spark = SparkSession.builder()
   *   .master("local[*]")
   *   .appName("testSaveLocalModel")
   *   .getOrCreate()
   *
   * val pool = Pool.load(
   *   spark,
   *   "dsv:///home/user/datasets/my_dataset/train.dsv",
   *   columnDescription = "/home/user/datasets/my_dataset/cd"
   * )
   *
   * val regressor = new CatBoostRegressor()
   * val model = regressor.fit(pool)
   *
   * // save in CatBoostBinary format
   * model.saveNativeModel("/home/user/model/model.cbm")
   *
   * // save in ONNX format with metadata
   * model.saveNativeModel(
   *   "/home/user/model/model.onnx",
   *   EModelType.Onnx,
   *   Map(
   *     "onnx_domain" -> "ai.catboost",
   *     "onnx_model_version" -> 1,
   *     "onnx_doc_string" -> "test model for regression",
   *     "onnx_graph_name" -> "CatBoostModel_for_regression"
   *   )
   * )
   * }}}
   */
  def saveNativeModel(
    fileName: String,
    format: EModelType = EModelType.CatboostBinary,
    exportParameters: java.util.Map[String, Any] = null,
    pool: Pool = null
  ) = {
    val exportParametersJsonString = if (exportParameters != null) {
      implicit val formats = org.json4s.DefaultFormats
      org.json4s.jackson.Serialization.write(exportParameters)
    } else {
      ""
    }
    val poolCatFeaturesMaxUniqValueCount = if (pool != null) {
      pool.getCatFeaturesUniqValueCounts.max
    } else {
      0
    }
    nativeModel.Save(fileName, format, exportParametersJsonString, poolCatFeaturesMaxUniqValueCount)
  }

  /**
   * @param fstrType Supported values are FeatureImportance, PredictionValuesChange, LossFunctionChange, PredictionDiff
   * @param data
   *  if fstrType is PredictionDiff it is required and must contain 2 samples
   *  if fstrType is PredictionValuesChange this param is required in case if model was explicitly trained
   *   with flag to store no leaf weights.
   *  otherwise it can be null
   * @param calcType Used only for PredictionValuesChange. Possible values:
   *   - Regular
   *       Calculate regular SHAP values
   *   - Approximate
   *       Calculate approximate SHAP values
   *   - Exact
   *       Calculate exact SHAP values
   * @return array of feature importances (index corresponds to the order of features in the model)
   */
  def getFeatureImportance(
    fstrType: EFstrType=EFstrType.FeatureImportance,
    data: Pool=null,
    calcType: ECalcTypeShapValues=ECalcTypeShapValues.Regular
  ) : Array[Double] = {
    (new impl.FeatureImportanceCalcer()).calc(this.nativeModel, fstrType, data, calcType)
  }

  /**
   * @param fstrType Supported values are FeatureImportance, PredictionValuesChange, LossFunctionChange, PredictionDiff
   * @param data
   *  if fstrType is PredictionDiff it is required and must contain 2 samples
   *  if fstrType is PredictionValuesChange this param is required in case if model was explicitly trained
   *   with flag to store no leaf weights.
   *  otherwise it can be null
   * @param calcType Used only for PredictionValuesChange. Possible values:
   *   - Regular
   *       Calculate regular SHAP values
   *   - Approximate
   *       Calculate approximate SHAP values
   *   - Exact
   *       Calculate exact SHAP values
   * @return array of feature importances sorted in descending order by importance
   */
  def getFeatureImportancePrettified(
    fstrType: EFstrType=EFstrType.FeatureImportance,
    data: Pool=null,
    calcType: ECalcTypeShapValues=ECalcTypeShapValues.Regular
  ) : Array[FeatureImportance]  = {
    val featureImportancesArray = getFeatureImportance(fstrType, data, calcType)
    val datasetFeaturesLayout = if (data != null) { data.getFeaturesLayout } else { new TFeaturesLayoutPtr }
    val featureNames = native_impl.GetMaybeGeneratedModelFeatureIds(nativeModel, datasetFeaturesLayout)
    featureNames.asScala.zip(featureImportancesArray).sortBy(-_._2).map{
      case (name, value) => new FeatureImportance(name, value)
    }.toArray
  }

  /**
   * @param data dataset to calculate SHAP values for
   * @param preCalcMode Possible values:
   *   - Auto
   *       Use direct SHAP Values calculation only if data size is smaller than average leaves number
   *       (the best of two strategies below is chosen).
   *   - UsePreCalc
   *       Calculate SHAP Values for every leaf in preprocessing. Final complexity is
   *       O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
   *       D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
   *       This is much faster (because of a smaller constant) than direct calculation when N >> L
   *   - NoPreCalc
   *       Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
   *       is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)
   * @param calcType Possible values:
   *   - Regular
   *       Calculate regular SHAP values
   *   - Approximate
   *       Calculate approximate SHAP values
   *   - Exact
   *       Calculate exact SHAP values
   * @param referenceData reference data for Independent Tree SHAP values from https://arxiv.org/abs/1905.04610v1
   *   if referenceData is not null, then Independent Tree SHAP values are calculated
   * @param outputColumns columns from data to add to output DataFrame, if null - add all columns
   * @return
   *   - for regression and binclass models:
   *     DataFrame which contains outputColumns and "shapValues" column with Vector of length (n_features + 1) with SHAP values
   *   - for multiclass models:
   *     DataFrame which contains outputColumns and "shapValues" column with Matrix of shape (n_classes x (n_features + 1)) with SHAP values
   */
  def getFeatureImportanceShapValues(
    data: Pool,
    preCalcMode: EPreCalcShapValues=EPreCalcShapValues.Auto,
    calcType: ECalcTypeShapValues=ECalcTypeShapValues.Regular,
    modelOutputType: EExplainableModelOutput=EExplainableModelOutput.Raw,
    referenceData: Pool=null,
    outputColumns: Array[String]=null
  ) : DataFrame = {
    (new impl.FeatureImportanceCalcer()).calcShapValues(
      this.nativeModel,
      data,
      preCalcMode,
      calcType,
      modelOutputType,
      referenceData,
      outputColumns
    )
  }

  /**
   * SHAP interaction values are calculated for all features pairs if nor featureIndices nor featureNames
   *   are specified.
   * @param data dataset to calculate SHAP interaction values
   * @param featureIndices (optional) pair of feature indices to calculate SHAP interaction values for.
   * @param featureNames (optional) pair of feature names to calculate SHAP interaction values for.
   * @param preCalcMode Possible values:
   *   - Auto
   *       Use direct SHAP Values calculation only if data size is smaller than average leaves number
   *       (the best of two strategies below is chosen).
   *   - UsePreCalc
   *       Calculate SHAP Values for every leaf in preprocessing. Final complexity is
   *       O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
   *       D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
   *       This is much faster (because of a smaller constant) than direct calculation when N >> L
   *   - NoPreCalc
   *       Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
   *       is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)
   * @param calcType Possible values:
   *   - Regular
   *       Calculate regular SHAP values
   *   - Approximate
   *       Calculate approximate SHAP values
   *   - Exact
   *       Calculate exact SHAP values
   * @param outputColumns columns from data to add to output DataFrame, if null - add all columns
   * @return
   *   - for binclass or regression:
   *       DataFrame which contains outputColumns and "featureIdx1", "featureIdx2", "shapInteractionValue" columns
   *   - for multiclass:
   *   		 DataFrame which contains outputColumns and "classIdx", "featureIdx1", "featureIdx2", "shapInteractionValue" columns
   */
  def getFeatureImportanceShapInteractionValues(
    data: Pool,
    featureIndices: Pair[Int, Int]=null,
    featureNames: Pair[String, String]=null,
    preCalcMode: EPreCalcShapValues=EPreCalcShapValues.Auto,
    calcType: ECalcTypeShapValues=ECalcTypeShapValues.Regular,
    outputColumns: Array[String]=null
  ) : DataFrame = {
    (new impl.FeatureImportanceCalcer()).calcShapInteractionValues(
      this.nativeModel,
      data,
      featureIndices,
      featureNames,
      preCalcMode,
      calcType,
      outputColumns
    )
  }

  /**
   * @return array of feature interaction scores
   */
  def getFeatureImportanceInteraction() : Array[FeatureInteractionScore] = {
    (new impl.FeatureImportanceCalcer()).calcInteraction(this.nativeModel)
  }
}

private[spark] object CatBoostModel {
  def sum[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]](
    models: Array[CatBoostModelTrait[Model]],
    weights: Array[Double] = null,
    ctrMergePolicy: ECtrTableMergePolicy = ECtrTableMergePolicy.IntersectingCountersAverage
  ) : TFullModel = {
    val nativeModels = new TVector_const_TFullModel_ptr
    for (model <- models) {
      nativeModels.add(model.nativeModel)
    }
    val weightsVector = if (weights != null) {
      new TVector_double(weights)
    } else {
      new TVector_double(Iterator.fill(models.length)(1.0).toArray)
    }
    native_impl.SumModels(nativeModels, weightsVector, /*modelParamsPrefixes*/ new TVector_TString, ctrMergePolicy)
  }
}

private[spark] trait CatBoostModelReaderTrait {
  case class Metadata(
    className: String,
    uid: String,
    timestamp: Long,
    sparkVersion: String
  )

  // based in DefaultParamsReader.loadMetadata
  private def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    val metadataPath = new org.apache.hadoop.fs.Path(path, "metadata").toString
    val metadataStr = sc.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr, expectedClassName)
  }

  // based in DefaultParamsReader.parseMetadata
  def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]

    if (expectedClassName.nonEmpty) {
      require(className == expectedClassName, s"Error loading metadata: Expected class name" +
        s" $expectedClassName but found class name $className")
    }

    Metadata(className, uid, timestamp, sparkVersion)
  }

  // returns (uid, nativeModel)
  protected def loadImpl(sc: SparkContext, className: String, path: String): (String, TFullModel) = {
    val metadata = loadMetadata(path, sc, className)

    val modelPath = new org.apache.hadoop.fs.Path(path, "model")
    val fileSystem = modelPath.getFileSystem(sc.hadoopConfiguration)
    val contentSummary = fileSystem.getContentSummary(modelPath)

    val nativeModel = new TFullModel

    val inputStream = fileSystem.open(modelPath)
    try {
      nativeModel.read(contentSummary.getLength.toInt, inputStream)
    } finally {
      inputStream.close()
    }
    (metadata.uid, nativeModel)
  }
}

private[spark] class CatBoostModelWriter[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]](
  instance: CatBoostModelTrait[Model]
) extends MLWriter {
  private val className = instance.getClass.getName

  // based on DefaultParamsWriter.getMetadataToSave
  private def getMetadataToSave(sc: SparkContext) : String = {
    val uid = instance.uid
    val cls = className
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid)
    val metadataJson: String = compact(render(basicMetadata))
    metadataJson
  }

  // based on DefaultParamsWriter.saveMetadata
  private def saveMetadata(path: String, sc: SparkContext) = {
    val metadataPath = new org.apache.hadoop.fs.Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(sc)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  override protected def saveImpl(path: String): Unit = {
    implicit val sc = super.sparkSession.sparkContext

    saveMetadata(path, sc)

    val modelPath = new org.apache.hadoop.fs.Path(path, "model")
    val outputStream = modelPath.getFileSystem(sc.hadoopConfiguration).create(modelPath)
    try {
      instance.nativeModel.write(outputStream)
    } finally {
      outputStream.close()
    }
  }
}
