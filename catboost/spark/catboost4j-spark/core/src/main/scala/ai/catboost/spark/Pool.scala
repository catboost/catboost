package ai.catboost.spark;

import java.nio.file.Path
import java.util.{Arrays,ArrayList,Collections};

import scala.reflect.ClassTag

import concurrent.duration.Duration
import concurrent.{Await,Future}
import concurrent.ExecutionContext.Implicits.global

import scala.jdk.CollectionConverters._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer,ArrayBuilder}

import org.apache.spark.internal.Logging
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{Vector,SparseVector,SQLDataTypes,Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.TaskContext

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._;

import ai.catboost.CatBoostError
import ai.catboost.spark.impl.RowEncoderConstructor
import ai.catboost.spark.params.{Helpers,PoolLoadParams,QuantizationParams,QuantizationParamsTrait}
import ai.catboost.spark.params.macros.ParamGetterSetter


private[spark] class QuantizedRowsOutputIterator(
    var objectIdx : Int,
    val dstRows : ArrayBuffer[Array[Any]],
    val featuresColumnIdx : Int,
    val quantizedRowAssembler : TQuantizedRowAssembler,
    val objectBlobSize : Int
) extends Iterator[Row] {
  def next() : Row = {
    val quantizedFeaturesBlob = new Array[Byte](objectBlobSize)
    quantizedRowAssembler.AssembleObjectBlob(objectIdx, quantizedFeaturesBlob)
    val dstRow = dstRows(objectIdx)
    dstRows(objectIdx) = null // to speed up cleanup
    dstRow(featuresColumnIdx) = quantizedFeaturesBlob
    objectIdx = objectIdx + 1
    return Row.fromSeq(dstRow)
  }

  def hasNext : Boolean = {
    return objectIdx < dstRows.size
  }
}

private[spark] object QuantizedRowsOutputIterator {
  def apply(
    dstRows : ArrayBuffer[Array[Any]],
    featuresColumnIdx : Int,
    quantizedObjectsDataProviderPtr : TQuantizedObjectsDataProviderPtr
  ) : QuantizedRowsOutputIterator = {
    val quantizedRowAssembler = new TQuantizedRowAssembler(quantizedObjectsDataProviderPtr)
    new QuantizedRowsOutputIterator(
      0,
      dstRows,
      featuresColumnIdx,
      quantizedRowAssembler,
      quantizedRowAssembler.GetObjectBlobSize()
    )
  }
}

/** Companion object for [[Pool]] class that is CatBoost's abstraction of a dataset
 */
object Pool {
  private def updateSparseFeaturesSize(data: DataFrame) : DataFrame = {
    val spark = data.sparkSession
    import spark.implicits._

    // extract max feature count from data
    val maxFeatureCountDF = data.mapPartitions {
      rows => {
        var maxFeatureCount = 0
        rows.foreach {
          row => {
            // features is always the 1st column after loading
            val featureCount = row.getAs[SparseVector](0).size
            if (featureCount > maxFeatureCount) {
              maxFeatureCount = featureCount
            }
          }
        }
        Iterator[Int](maxFeatureCount)
      }
    }

    var maxFeatureCount = 0
    for (featureCount <- maxFeatureCountDF.collect()) {
      if (featureCount > maxFeatureCount) {
        maxFeatureCount = featureCount
      }
    }

    val existingFeatureNames = getFeatureNames(data, "features")
    val extendedFeatureNames = Arrays.copyOf[String](existingFeatureNames, maxFeatureCount)
    Arrays.fill(extendedFeatureNames.asInstanceOf[Array[Object]], existingFeatureNames.length, maxFeatureCount, "")

    val updatedMetadata = DataHelpers.makeFeaturesMetadata(extendedFeatureNames)

    val updateFeaturesSize = udf(
      (features: Vector) => {
        val sparseFeatures = features.asInstanceOf[SparseVector]
        Vectors.sparse(maxFeatureCount, sparseFeatures.indices, sparseFeatures.values)
      }
    )

    data.withColumn("features", updateFeaturesSize($"features").as("_", updatedMetadata))
  }

  /**
   * Load dataset in one of CatBoost's natively supported formats:
   *
   *  - [[https://catboost.ai/docs/concepts/input-data_values-file.html dsv]]
   *  - [[https://catboost.ai/docs/concepts/input-data_libsvm.html libsvm]]
   *
   * @param spark SparkSession
   * @param dataPathWithScheme Path with scheme to dataset in CatBoost format.
   *  For example, `dsv:///home/user/datasets/my_dataset/train.dsv` or
   *  `libsvm:///home/user/datasets/my_dataset/train.libsvm`
   * @param columnDescription Path to
   *  [[https://catboost.ai/docs/concepts/input-data_column-descfile.html column description file]]
   * @param params Additional params specifying data format.
   * @param pairsDataPathWithScheme (optional) Path with scheme to dataset pairs in CatBoost format.
   *  Only "dsv-grouped" format is supported for now.
   *  For example, `dsv-grouped:///home/user/datasets/my_dataset/train_pairs.dsv`
   * @return [[Pool]] containing loaded data.
   *
   * @example
   * {{{
   * val spark = SparkSession.builder()
   *   .master("local[*]")
   *   .appName("testLoadDSVSimple")
   *   .getOrCreate()
   *
   * val pool = Pool.load(
   *   spark,
   *   "dsv:///home/user/datasets/my_dataset/train.dsv",
   *   columnDescription = "/home/user/datasets/my_dataset/cd"
   * )
   *
   *  val poolWithPairs = Pool.load(
   *   spark,
   *   "dsv:///home/user/datasets/my_dataset_with_pairs/train.dsv",
   *   columnDescription = "/home/user/datasets/my_dataset_with_pairs/cd",
   *   pairsDataPathWithScheme = "dsv-grouped:///home/user/datasets/my_dataset_with_pairs/train_pairs.dsv"
   * )
   * }}}
   */
  def load(
    spark: SparkSession,
    dataPathWithScheme: String,
    columnDescription: Path = null, // API for Java, so no Option[_] here.
    params: PoolLoadParams = new PoolLoadParams(),
    pairsDataPathWithScheme: String = null): Pool = {

    val pathParts = dataPathWithScheme.split("://", 2)
    val (dataScheme, dataPath) =
      if (pathParts.size == 1) ("dsv", pathParts(0)) else (pathParts(0), pathParts(1))

    val format = dataScheme match {
      case "dsv" | "libsvm" => "ai.catboost.spark.CatBoostTextFileFormat"
      case _ => throw new CatBoostError(s"Loading pool from scheme ${dataScheme} is not supported")
    }

    val dataSourceOptions = mutable.Map[String,String]()

    val pairsData = if (pairsDataPathWithScheme != null) {
      dataSourceOptions.update("addSampleId", "true")
      CatBoostPairsDataLoader.load(spark, pairsDataPathWithScheme)
    } else {
      null
    }

    dataSourceOptions.update("dataScheme", dataScheme)

    params.extractParamMap.toSeq.foreach {
      case ParamPair(param, value) => {
        dataSourceOptions.update(param.name, value.toString)
      }
    }
    if (columnDescription != null) {
      dataSourceOptions.update("columnDescription", columnDescription.toString)
    }
    dataSourceOptions.update(
        "catboostJsonParams",
        Helpers.sparkMlParamsToCatBoostJsonParamsString(params)
    )
    dataSourceOptions.update("uuid", java.util.UUID.randomUUID().toString)

    var data = spark.read.format(format).options(dataSourceOptions).load(dataPath)
    if (pairsData != null) {
      data = DataHelpers.mapSampleIdxToPerGroupSampleIdx(data)
    }

    val pool = new Pool(
      if (dataScheme == "libsvm") updateSparseFeaturesSize(data) else data,
      pairsData=pairsData
    )

    setColumnParamsFromLoadedData(pool)

    pool
  }

  /**
   * Returns a PoolReader that can be used to read Pool (API similar to Spark's DataFrameReader).
   */
  def read(spark: SparkSession) : PoolReader = {
    new PoolReader(spark)
  }

  private[spark] def setColumnParamsFromLoadedData(pool: Pool) {
    // CatBoost loaders always use standard names, column parameter name is taken by adding "Col" suffix
    for (name <- pool.data.columns) {
      pool.set(name + "Col", name)
    }
  }

  private[spark] def getFeatureCount(data : DataFrame, featuresCol : String) : Int = {
    val attributeGroup = AttributeGroup.fromStructField(data.schema(featuresCol))
    val optNumAttributes = attributeGroup.numAttributes
    if (optNumAttributes.isDefined) {
      optNumAttributes.get.toInt
    } else {
      val optAttributes = attributeGroup.attributes
      if (optAttributes.isDefined) {
        return optAttributes.get.size
      } else {
        if (data.count == 0) {
          throw new CatBoostError("Cannot get feature count from empty DataFrame without attributes")
        }
        data.first().getAs[Vector](featuresCol).size
      }
    }
  }

  private[spark] def getFeatureNames(data: DataFrame, featuresCol: String): Array[String] = {
    val featureCount = getFeatureCount(data, featuresCol)
    val attributes = AttributeGroup.fromStructField(data.schema(featuresCol)).attributes
    if (attributes.isEmpty) {
      val featureNames = new Array[String](featureCount)
      Arrays.fill(featureNames.asInstanceOf[Array[Object]], 0, featureCount, "")
      featureNames
    } else {
      if (attributes.get.size != featureCount) {
        throw new CatBoostError(
          s"number of attributes (${attributes.get.size}) is not equal to featureCount ($featureCount)"
        )
      }
      attributes.get.map { attribute => attribute.name.getOrElse("") }.toArray
    }
  }

  /**
   * @return array of flat feature index to uniq cat features values count
   */
  private[spark] def getCatFeaturesUniqValueCounts(data: DataFrame, featuresCol: String) : Array[Int] = {
    val featureCount = getFeatureCount(data, featuresCol)

    val attributes = AttributeGroup.fromStructField(data.schema(featuresCol)).attributes
    if (attributes.isEmpty) {
      val result = new Array[Int](featureCount)
      Arrays.fill(result, 0)
      result
    } else {
      if (attributes.get.size != featureCount) {
        throw new CatBoostError(
          s"number of attributes (${attributes.get.size}) is not equal to featureCount ($featureCount)"
        )
      }
      attributes.get.map {
        case nominalAttribute : NominalAttribute => {
          if (nominalAttribute.numValues.isDefined) {
            nominalAttribute.numValues.get
          } else {
            if (nominalAttribute.values.isEmpty) {
              throw new CatBoostError(
                "Neither numValues nor values is defined for categorical feature attribute"
              )
            }
            nominalAttribute.values.get.length
          }
        }
        case _ => 0
      }.toArray
    }
  }

  private[spark] def getCatFeaturesMaxUniqValueCount(data: DataFrame, featuresCol: String) : Int = {
    getCatFeaturesUniqValueCounts(data, featuresCol).max
  }
}


/** CatBoost's abstraction of a dataset.
 *
 *  Features data can be stored in raw (features column has [[org.apache.spark.ml.linalg.Vector]] type)
 *  or quantized (float feature values are quantized into integer bin values, features column has
 *  `Array[Byte]` type) form.
 *
 *  Raw [[Pool]] can be transformed to quantized form using `quantize` method.
 *  This is useful if this dataset is used for training multiple times and quantization parameters do not
 *  change. Pre-quantized [[Pool]] allows to cache quantized features data and so do not re-run
 *  feature quantization step at the start of an each training.
 *
 * @groupname persistence Caching and Persistence
 */
class Pool (
    override val uid: String,
    val data: DataFrame = null,
    protected var featuresLayout: TFeaturesLayoutPtr = null, // updated on demand if not initialized
    val quantizedFeaturesInfo: QuantizedFeaturesInfoPtr = null,
    val pairsData: DataFrame = null,
    val partitionedByGroups: Boolean = false)
  extends Params with HasLabelCol with HasFeaturesCol with HasWeightCol with Logging {

  private[spark] def this(
    uid: Option[String],
    data: DataFrame,
    pairsData: DataFrame,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    partitionedByGroups: Boolean
  ) =
    this(
      uid.getOrElse(Identifiable.randomUID("catboostPool")),
      data,
      if (quantizedFeaturesInfo != null) quantizedFeaturesInfo.GetFeaturesLayout() else null,
      quantizedFeaturesInfo,
      pairsData,
      partitionedByGroups
    )

  /** Construct [[Pool]] from [[DataFrame]]
   *  Call set*Col methods to specify non-default columns.
   *  Only features and label columns with "features" and "label" names are assumed by default.
   *
   * @example
   * {{{
   *   val spark = SparkSession.builder()
   *     .master("local[4]")
   *     .appName("PoolTest")
   *     .getOrCreate();
   *
   *   val srcData = Seq(
   *     Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0x0L, 0.12f),
   *     Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0x0L, 0.18f),
   *     Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x1L, 1.0f)
   *   )
   *
   *   val srcDataSchema = Seq(
   *     StructField("features", SQLDataTypes.VectorType),
   *     StructField("label", StringType),
   *     StructField("groupId", LongType),
   *     StructField("weight", FloatType)
   *   )
   *
   *   val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))
   *
   *   val pool = new Pool(df)
   *     .setGroupIdCol("groupId")
   *     .setWeightCol("weight")
   *
   *   pool.data.show()
   * }}}
   */
  def this(data: DataFrame) = this(None, data, null, null, false)

  /** Construct [[Pool]] from [[DataFrame]] also specifying pairs data in an additional [[DataFrame]]
   * @example
   * {{{
   *   val spark = SparkSession.builder()
   *     .master("local[4]")
   *     .appName("PoolWithPairsTest")
   *     .getOrCreate();
   *
   *   val srcData = Seq(
   *     Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0x0L, 0.12f, 0),
   *     Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0x0L, 0.18f, 1),
   *     Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x1L, 1.0f, 2),
   *     Row(Vectors.dense(0.23, 0.01, 0.0), "0.0", 0x1L, 1.2f, 3)
   *   )
   *
   *   val srcDataSchema = Seq(
   *     StructField("features", SQLDataTypes.VectorType),
   *     StructField("label", StringType),
   *     StructField("groupId", LongType),
   *     StructField("weight", FloatType)
   *     StructField("sampleId", LongType)
   *   )
   *
   *   val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))
   *
   *   val srcPairsData = Seq(
   *     Row(0x0L, 0, 1),
   *     Row(0x1L, 3, 2)
   *   )
   *
   *   val srcPairsDataSchema = Seq(
   *     StructField("groupId", LongType),
   *     StructField("winnerId", IntegerType),
   *     StructField("loserId", IntegerType)
   *   )
   *
   *   val pairsDf = spark.createDataFrame(
   *     spark.sparkContext.parallelize(srcPairsData),
   *     StructType(srcPairsDataSchema)
   *   )
   *
   *   val pool = new Pool(df, pairsDf)
   *     .setGroupIdCol("groupId")
   *     .setWeightCol("weight")
   *     .setSampleIdCol("sampleId")
   *
   *   pool.data.show()
   *   pool.pairsData.show()
   * }}}
   */
  def this(data: DataFrame, pairsData: DataFrame) = this(None, data, pairsData, null, false)

  def getFeaturesLayout : TFeaturesLayoutPtr = {
    if (featuresLayout == null) {
      if (isQuantized) {
        throw new CatBoostError("featuresLayout must be defined for quantized pool")
      }

      val featuresMetaInfo = new TVector_TFeatureMetaInfo

      val attributes = AttributeGroup.fromStructField(data.schema($(featuresCol))).attributes
      if (attributes.isEmpty) {
        val featureCount = Pool.getFeatureCount(data, $(featuresCol))
        for (i <- 0 until featureCount) {
          val featureMetaInfo = new TFeatureMetaInfo
          featureMetaInfo.setType(EFeatureType.Float)
          featuresMetaInfo.add(featureMetaInfo)
        }
      } else {
        for (attribute <- attributes.get) {
          val featureMetaInfo = new TFeatureMetaInfo
          attribute match {
            case nominal : NominalAttribute => {
              featureMetaInfo.setType(EFeatureType.Categorical)
            }
            case numerical => {
              featureMetaInfo.setType(EFeatureType.Float)
            }
          }
          featureMetaInfo.setName(attribute.name.getOrElse(""))
          featuresMetaInfo.add(featureMetaInfo)
        }
      }

      featuresLayout = new TFeaturesLayoutPtr(new TFeaturesLayout)
      featuresLayout.__deref__.Init(featuresMetaInfo)
    }
    featuresLayout
  }

  /** @group setParam */
  def setLabelCol(value: String): Pool = set(labelCol, value).asInstanceOf[Pool]

  /** @group setParam */
  def setFeaturesCol(value: String): Pool = set(featuresCol, value).asInstanceOf[Pool]

  /** @group setParam */
  def setWeightCol(value: String): Pool = set(weightCol, value).asInstanceOf[Pool]

  /**
   * Param for sampleId column name.
   * @group param
   */
  @ParamGetterSetter final val sampleIdCol: Param[String] =
    new Param[String](
      this,
      "sampleIdCol",
      "sampleId column name")

  /**
   * Param for groupWeight column name.
   * @group param
   */
  @ParamGetterSetter final val groupWeightCol: Param[String] =
    new Param[String](
      this,
      "groupWeightCol",
      "groupWeight column name")

  /**
   * Param for baseline column name.
   * @group param
   */
  @ParamGetterSetter final val baselineCol: Param[String] =
    new Param[String](
      this,
      "baselineCol",
      "baseline column name")

  /**
   * Param for groupId column name.
   * @group param
   */
  @ParamGetterSetter final val groupIdCol: Param[String] =
    new Param[String](
      this,
      "groupIdCol",
      "groupId column name")

  /**
   * Param for subgroupId column name.
   * @group param
   */
  @ParamGetterSetter final val subgroupIdCol: Param[String] =
    new Param[String](this, "subgroupIdCol", "subgroupId column name")


  /**
   * Param for timestamp column name.
   * @group param
   */
  @ParamGetterSetter final val timestampCol: Param[String] =
    new Param[String](this, "timestampCol", "timestamp column name")

  override def copy(extra: ParamMap): Pool = defaultCopy(extra)

  def isQuantized: Boolean = { quantizedFeaturesInfo != null }

  def getFeatureCount: Int = {
    getFeaturesLayout.__deref__.GetExternalFeatureCount.toInt
  }

  def getFeatureNames : Array[String] = {
    getFeaturesLayout.__deref__.GetExternalFeatureIds.toArray(new Array[String](0))
  }

  def getCatFeaturesUniqValueCounts : Array[Int] = {
    if (isQuantized) {
      native_impl.GetCategoricalFeaturesUniqueValuesCounts(quantizedFeaturesInfo.__deref__).toPrimitiveArray
    } else {
      Pool.getCatFeaturesUniqValueCounts(data, $(featuresCol))
    }
  }

  def getEstimatedFeatureCount: Int = {
    if (isQuantized) {
      if (data.schema.fieldNames.contains("_estimatedFeatures")) {
        if (data.count == 0) {
          throw new CatBoostError("Cannot get estimated feature count from empty DataFrame")
        }
        data.first().getAs[Array[Byte]]("_estimatedFeatures").length
      } else {
        0
      }
    } else {
      0
    }
  }

  /** @return Number of objects in the dataset, similar to the same method of
   *  [[org.apache.spark.sql.Dataset]]
   */
  def count : Long = data.count

  /** @return Number of pairs in the dataset
   */
  def pairsCount : Long = if (pairsData != null) pairsData.count else 0.toLong

  /**
   * @return dimension of formula baseline, 0 if no baseline specified
   */
  def getBaselineCount: Int = {
    if (isDefined(baselineCol)) {
      data.select(getOrDefault(baselineCol)).head.getAs[Vector](0).size
    } else {
      0
    }
  }

  def getTargetType : ERawTargetType = {
    if (isDefined(labelCol)) {
      val dataType = data.schema(getOrDefault(labelCol)).dataType
      dataType match {
        case DataTypes.DoubleType | DataTypes.FloatType => ERawTargetType.Float
        case DataTypes.IntegerType | DataTypes.LongType => ERawTargetType.Integer
        case DataTypes.StringType => ERawTargetType.String
        case _ => throw new CatBoostError(s"unsupported target column type: $dataType")
      }
    } else {
      ERawTargetType.None
    }
  }

  /**
   * Persist Datasets of this Pool with the default storage level (MEMORY_AND_DISK).
   *
   * @group persistence
   */
  def cache() : Pool = {
    val result = new Pool(
      None,
      this.data.cache(),
      if (this.pairsData != null) this.pairsData.cache() else null,
      this.quantizedFeaturesInfo,
      this.partitionedByGroups
    )
    copyValues(result)
  }

  /**
   * Returns Pool with checkpointed Datasets.
   *
   * @param eager Whether to checkpoint Datasets immediately
   *
   * @group persistence
   */
  def checkpoint(eager: Boolean) : Pool = {
    val result = new Pool(
      None,
      this.data.checkpoint(eager),
      if (this.pairsData != null) this.pairsData.checkpoint(eager) else null,
      this.quantizedFeaturesInfo,
      this.partitionedByGroups
    )
    copyValues(result)
  }

  /**
   * Returns Pool with eagerly checkpointed Datasets.
   *
   * @group persistence
   */
  def checkpoint() : Pool = {
    checkpoint(eager = true)
  }

  /**
   * Returns Pool with locally checkpointed Datasets.
   *
   * @param eager Whether to checkpoint Datasets immediately
   *
   * @group persistence
   */
  def localCheckpoint(eager: Boolean) : Pool = {
    val result = new Pool(
      None,
      this.data.localCheckpoint(eager),
      if (this.pairsData != null) this.pairsData.localCheckpoint(eager) else null,
      this.quantizedFeaturesInfo,
      this.partitionedByGroups
    )
    copyValues(result)
  }

  /**
   * Returns Pool with eagerly locally checkpointed Datasets.
   *
   * @group persistence
   */
  def localCheckpoint() : Pool = {
    localCheckpoint(eager = true)
  }

  /**
   * Returns Pool with Datasets persisted with the given storage level.
   *
   * @group persistence
   */
  def persist(storageLevel: StorageLevel) : Pool = {
    val result = new Pool(
      None,
      this.data.persist(storageLevel),
      if (this.pairsData != null) this.pairsData.persist(storageLevel) else null,
      this.quantizedFeaturesInfo,
      this.partitionedByGroups
    )
    copyValues(result)
  }

  /**
   * Persist Datasets of this Pool with the default storage level (MEMORY_AND_DISK).
   *
   * @group persistence
   */
  def persist() : Pool = {
    persist(StorageLevel.MEMORY_AND_DISK)
  }

  /**
   * Mark Datasets of this Pool as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @group persistence
   */
  def unpersist() : Pool = {
    unpersist(blocking = false)
  }

  /**
   * Mark Datasets of this Pool as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @param blocking Whether to block until all blocks are deleted.
   *
   * @group persistence
   */
  def unpersist(blocking: Boolean) : Pool = {
    val result = new Pool(
      None,
      this.data.unpersist(blocking),
      if (this.pairsData != null) this.pairsData.unpersist(blocking) else null,
      this.quantizedFeaturesInfo,
      this.partitionedByGroups
    )
    copyValues(result)
  }

  /**
   * Interface for saving the content out into external storage (API similar to Spark's Dataset).
   */
  def write() : PoolWriter = {
    new PoolWriter(this)
  }


  private[spark] def createDataMetaInfo(selectedColumnTypes: Seq[String] = null) : TIntermediateDataMetaInfo = {
    val result = new TIntermediateDataMetaInfo
    result.setObjectCount(java.math.BigInteger.valueOf(count))
    result.setFeaturesLayout(this.getFeaturesLayout)

    if (selectedColumnTypes == null) {
      val targetType = getTargetType
      if (targetType != ERawTargetType.None) {
        result.setTargetType(targetType)
        result.setTargetCount(1)
        //result.setClassLabelsFromJsonString()
      }
      result.setBaselineCount(getBaselineCount)
      result.setHasGroupId(isDefined(groupIdCol))
      result.setHasGroupWeight(isDefined(groupWeightCol))
      result.setHasSubgroupIds(isDefined(subgroupIdCol))
      result.setHasWeights(isDefined(weightCol))
      result.setHasTimestamp(isDefined(timestampCol))
    } else {
      if (selectedColumnTypes.contains("label")) {
        val targetType = getTargetType
        if (targetType != ERawTargetType.None) {
          result.setTargetType(targetType)
          result.setTargetCount(1)
          //result.setClassLabelsFromJsonString()
        }
        if (selectedColumnTypes.contains("baseline")) {
          result.setBaselineCount(getBaselineCount)
        }
        if (selectedColumnTypes.contains("groupId")) {
          result.setHasGroupId(isDefined(groupIdCol))
        }
        if (selectedColumnTypes.contains("groupWeight")) {
          result.setHasGroupWeight(isDefined(groupWeightCol))
        }
        if (selectedColumnTypes.contains("subgroupId")) {
          result.setHasSubgroupIds(isDefined(subgroupIdCol))
        }
        if (selectedColumnTypes.contains("weight")) {
          result.setHasWeights(isDefined(weightCol))
        }
        if (selectedColumnTypes.contains("timestamp")) {
          result.setHasTimestamp(isDefined(timestampCol))
        }
      }
    }

    result
  }

  protected def calcNanModesAndBorders(
    nanModeAndBordersBuilder: TNanModeAndBordersBuilder,
    quantizationParams: QuantizationParamsTrait
  ) = {
    log.info("calcNanModesAndBorders: start")

    val calcHasNansSeparately = count > QuantizationParams.MaxSubsetSizeForBuildBordersAlgorithms
    val calcHasNansFuture = Future {
      if (calcHasNansSeparately) {
        log.info("calcFeaturesHasNans: start")
        val result = DataHelpers.calcFeaturesHasNans(data, getFeaturesCol, this.getFeatureCount)
        log.info("calcFeaturesHasNans: finish")
        result
      } else {
        Array[Byte]()
      }
    }

    val dataForBuildBorders =
      if (count > QuantizationParams.MaxSubsetSizeForBuildBordersAlgorithms) {
        data.select(getFeaturesCol).sample(
          QuantizationParams.MaxSubsetSizeForBuildBordersAlgorithms.toDouble / count.toDouble,
          /*seed*/ 0
        )
      } else {
        data.select(getFeaturesCol)
      }.persist(StorageLevel.MEMORY_ONLY)

    nanModeAndBordersBuilder.SetSampleSize(dataForBuildBorders.count.toInt)

    log.info("calcNanModesAndBorders: reading data: start")
    for (row <- dataForBuildBorders.toLocalIterator.asScala) {
       nanModeAndBordersBuilder.AddSample(row.getAs[Vector](0).toArray)
    }
    log.info("calcNanModesAndBorders: reading data: end")

    dataForBuildBorders.unpersist()

    log.info("CalcBordersWithoutNans: start")
    nanModeAndBordersBuilder.CalcBordersWithoutNans(
      quantizationParams.get(quantizationParams.threadCount).getOrElse(
        SparkHelpers.getThreadCountForDriver(data.sparkSession)
      )
    )
    log.info("CalcBordersWithoutNans: finish")

    val hasNansArray = Await.result(calcHasNansFuture, Duration.Inf)
    nanModeAndBordersBuilder.Finish(hasNansArray)
    log.info("calcNanModesAndBorders: finish")
  }

  protected def updateCatFeaturesInfo(
    isInitialization: Boolean,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr
  ) = {
    val catFeaturesUniqValueCounts = Pool.getCatFeaturesUniqValueCounts(data, $(featuresCol))
    native_impl.UpdateCatFeaturesInfo(
      catFeaturesUniqValueCounts,
      isInitialization,
      quantizedFeaturesInfo.Get
    )
  }

  protected def createQuantizationSchema(quantizationParams: QuantizationParamsTrait)
    : QuantizedFeaturesInfoPtr = {

    val catBoostJsonParamsString = Helpers.sparkMlParamsToCatBoostJsonParamsString(quantizationParams)
    val quantizedFeaturesInfo = native_impl.PrepareQuantizationParameters(
      getFeaturesLayout.__deref__,
      catBoostJsonParamsString
    )

    val nanModeAndBordersBuilder = new TNanModeAndBordersBuilder(quantizedFeaturesInfo)

    if (nanModeAndBordersBuilder.HasFeaturesToCalc) {
      calcNanModesAndBorders(nanModeAndBordersBuilder, quantizationParams)
    }

    updateCatFeaturesInfo(isInitialization=true, quantizedFeaturesInfo=quantizedFeaturesInfo)

    quantizedFeaturesInfo
  }

  protected def createQuantized(quantizedFeaturesInfo: QuantizedFeaturesInfoPtr) : Pool = {
    var featuresColumnIdx = data.schema.fieldIndex($(featuresCol));
    val threadCountForTask = SparkHelpers.getThreadCountForTask(data.sparkSession)
    val catFeaturesMaxUniqValueCount = native_impl.CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(
      quantizedFeaturesInfo.__deref__()
    )

    val featuresColumnName = $(featuresCol)

    val quantizedDataSchema = StructType(
      data.schema.map {
        structField => {
          if (structField.name == featuresColumnName) {
            StructField(featuresColumnName, BinaryType, false)
          } else {
            structField
          }
        }
      }
    )
    val quantizedDataEncoder = RowEncoderConstructor.construct(quantizedDataSchema)

    val quantizedData = data.mapPartitions(
      rowsIterator => {
        if (rowsIterator.isEmpty) {
          Iterator[Row]()
        } else {
          val localExecutor = new TLocalExecutor
          localExecutor.Init(threadCountForTask)

          // source features column will be replaced by quantizedFeatures
          val (dstRows, rawObjectsDataProviderPtr) = DataHelpers.processDatasetWithRawFeatures(
            rowsIterator,
            featuresColumnIdx,
            quantizedFeaturesInfo.GetFeaturesLayout(),
            catFeaturesMaxUniqValueCount,
            keepRawFeaturesInDstRows = false,
            dstRowLength = quantizedDataSchema.length,
            localExecutor = localExecutor
          )

          val quantizedObjectsDataProvider = native_impl.Quantize(
            quantizedFeaturesInfo,
            rawObjectsDataProviderPtr,
            localExecutor
          )

          QuantizedRowsOutputIterator(dstRows, featuresColumnIdx, quantizedObjectsDataProvider)
        }
      }
    )(quantizedDataEncoder)

    val quantizedPool = new Pool(None, quantizedData, pairsData, quantizedFeaturesInfo, partitionedByGroups)
    copyValues(quantizedPool)
  }

  /**
   * Create [[Pool]] with quantized features from [[Pool]] with raw features
   *
   * @example
   * {{{
   *  val spark = SparkSession.builder()
   *    .master("local[*]")
   *    .appName("QuantizationTest")
   *    .getOrCreate();
   *
   *  val srcData = Seq(
   *    Row(Vectors.dense(0.1, 0.2, 0.11), "0.12"),
   *    Row(Vectors.dense(0.97, 0.82, 0.33), "0.22"),
   *    Row(Vectors.dense(0.13, 0.22, 0.23), "0.34")
   *  )
   *
   *  val srcDataSchema = Seq(
   *    StructField("features", SQLDataTypes.VectorType),
   *    StructField("label", StringType)
   *  )
   *
   *  val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))
   *
   *  val pool = new Pool(df)
   *
   *  val quantizedPool = pool.quantize(new QuantizationParams)
   *  val quantizedPoolWithTwoBinsPerFeature = pool.quantize(new QuantizationParams().setBorderCount(1))
   *  quantizedPool.data.show()
   *  quantizedPoolWithTwoBinsPerFeature.data.show()
   * }}}
   */
  def quantize(quantizationParams: QuantizationParamsTrait = new QuantizationParams()) : Pool = {
    if (isQuantized) {
      throw new CatBoostError("Pool is already quantized")
    }
    createQuantized(createQuantizationSchema(quantizationParams))
  }

  /**
   * Create [[Pool]] with quantized features from [[Pool]] with raw features.
   * This variant of the method is useful if QuantizedFeaturesInfo with data for quantization
   * (borders and nan modes) has already been computed.
   * Used, for example, to quantize evaluation datasets after the training dataset has been quantized.
   */
  def quantize(quantizedFeaturesInfo: QuantizedFeaturesInfoPtr) : Pool = {
    if (isQuantized) {
      throw new CatBoostError("Pool is already quantized")
    }

    // because there can be new values
    updateCatFeaturesInfo(isInitialization=false, quantizedFeaturesInfo=quantizedFeaturesInfo)

    createQuantized(quantizedFeaturesInfo)
  }

  private[spark] def quantizeForModelApplicationImpl(model: TFullModel) : Pool = {
    if (this.isQuantized) {
      native_impl.CheckModelAndDatasetCompatibility(model, this.quantizedFeaturesInfo.__deref__())
      this
    } else {
      this.quantize(
        native_impl.CreateQuantizedFeaturesInfoForModelApplication(model, this.getFeaturesLayout.__deref__)
      )
    }
  }

  /**
   * Create [[Pool]] with quantized features from [[Pool]] with raw features.
   * This variant of the method is used when we want to apply CatBoostModel on Pool
   */
  def quantizeForModelApplication[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]](
    model: CatBoostModelTrait[Model]
  ) : Pool = {
    this.quantizeForModelApplicationImpl(model.nativeModel)
  }

  /**
   * Repartition data to the specified number of partitions.
   * Useful to repartition data to create one partition per executor for training
   * (where each executor gets its' own CatBoost worker with a part of the training data).
   */
  def repartition(partitionCount: Int, byGroupColumnsIfPresent: Boolean = true) : Pool = {
    val maybeGroupIdCol = get(groupIdCol)
    var partitionedByGroups = false
    val newData = if (byGroupColumnsIfPresent && maybeGroupIdCol.isDefined) {
      partitionedByGroups = true
      val maybeSubgroupIdCol = get(subgroupIdCol)
      if (maybeSubgroupIdCol.isDefined) {
        data.repartition(partitionCount, new Column(maybeGroupIdCol.get)).sortWithinPartitions(
          new Column(maybeGroupIdCol.get),
          new Column(maybeSubgroupIdCol.get)
        )
      } else {
        data.repartition(partitionCount, new Column(maybeGroupIdCol.get)).sortWithinPartitions(
          new Column(maybeGroupIdCol.get)
        )
      }
    } else {
      data.repartition(partitionCount)
    }
    val result = new Pool(None, newData, pairsData, this.quantizedFeaturesInfo, partitionedByGroups)
    copyValues(result)
  }

  /**
   * Create subset of this pool with the fraction of the samples (or groups of samples if present)
   */
  def sample(fraction: Double) : Pool = {
    if ((fraction < 0.0) || (fraction > 1.0)) {
      throw new CatBoostError("sample: fraction must be in [0, 1] interval")
    }
    val spark = this.data.sparkSession
    val sampledPool = if (this.isDefined(groupIdCol)) {
      val mainDataGroupIdFieldIdx = this.data.schema.fieldIndex(this.getGroupIdCol)
      val groupedMainData = this.data.rdd.groupBy(row => row.getLong(mainDataGroupIdFieldIdx))
      if (this.pairsData != null) {
        val pairsGroupIdx = this.pairsData.schema.fieldIndex("groupId")
        val groupedPairsData = this.pairsData.rdd.groupBy(row => row.getLong(pairsGroupIdx))

        val sampledCogroupedData = groupedMainData.cogroup(groupedPairsData).sample(
          withReplacement=false,
          fraction=fraction
        )
        val sampledMainData = sampledCogroupedData.flatMap(
          (group: (Long, (Iterable[Iterable[Row]], Iterable[Iterable[Row]]))) => {
            group._2._1.flatMap((it : Iterable[Row]) => it)
          }
        )
        val sampledPairsData = sampledCogroupedData.flatMap(
          (group: (Long, (Iterable[Iterable[Row]], Iterable[Iterable[Row]]))) => {
            group._2._2.flatMap((it : Iterable[Row]) => it)
          }
        )
        new Pool(
          None,
          spark.createDataFrame(sampledMainData, this.data.schema),
          spark.createDataFrame(sampledPairsData, this.pairsData.schema),
          this.quantizedFeaturesInfo,
          true
        )
      } else {
        val sampledGroupedMainData = groupedMainData.sample(withReplacement=false, fraction=fraction)
        val sampledMainData = sampledGroupedMainData.flatMap(_._2)
        new Pool(
          None,
          spark.createDataFrame(sampledMainData, this.data.schema),
          null,
          this.quantizedFeaturesInfo,
          true
        )
      }
    } else {
      new Pool(None, this.data.sample(fraction), null, this.quantizedFeaturesInfo, false)
    }
    this.copyValues(sampledPool)
  }

  /**
   * ensure that if groups are present data in partitions contains whole groups in consecutive order
   */
  def ensurePartitionByGroupsIfPresent() : Pool = {
    if (!this.isDefined(this.groupIdCol) || this.partitionedByGroups) {
      this
    } else {
      this.repartition(partitionCount=this.data.rdd.getNumPartitions, byGroupColumnsIfPresent=true)
    }
  }

  /**
   * used to add additional columns to data (for example estimated features)
   * It is impossible to just write an external function for this because copyValues is protected
   */
  def copyWithModifiedData(modifiedData: DataFrame, partitionedByGroups: Boolean=false) : Pool = {
    val result = new Pool(None, modifiedData, this.pairsData, this.quantizedFeaturesInfo, partitionedByGroups)
    copyValues(result)
  }

  /**
   * Map over partitions for quantized Pool
   */
  def mapQuantizedPartitions[R : Encoder : ClassTag](
    selectedColumns: Seq[String],
    includeEstimatedFeatures: Boolean,
    includePairsIfPresent: Boolean,
    dstColumnNames: Array[String], // can be null, add all columns to dst in this case
    dstRowLength: Int,
    f : (TDataProviderPtr, TDataProviderPtr, mutable.ArrayBuffer[Array[Any]], TLocalExecutor) => Iterator[R]
  ) : Dataset[R] = {
    if (!this.isQuantized) {
      throw new CatBoostError("mapQuantizedPartitions requires quantized pool")
    }

    val preparedPool = if (selectedColumns.contains("groupId")) {
      this.ensurePartitionByGroupsIfPresent()
    } else {
      this
    }

    val (columnIndexMap, selectedColumnNames, dstColumnIndices, estimatedFeatureCount) = DataHelpers.selectColumnsAndReturnIndex(
      preparedPool,
      selectedColumns,
      includeEstimatedFeatures,
      dstColumnNames = if (dstColumnNames != null) { dstColumnNames } else { preparedPool.data.schema.fieldNames }
    )
    if (dstColumnIndices.size > dstRowLength) {
      throw new CatBoostError(s"dstRowLength ($dstRowLength) < dstColumnIndices.size (${dstColumnIndices.size})")
    }

    val selectedDF = preparedPool.data.select(selectedColumnNames.head, selectedColumnNames.tail: _*)

    val spark = preparedPool.data.sparkSession
    val threadCountForTask = SparkHelpers.getThreadCountForTask(spark)

    val quantizedFeaturesInfo = preparedPool.quantizedFeaturesInfo
    val dataMetaInfo = preparedPool.createDataMetaInfo(selectedColumns)

    val schema = preparedPool.data.schema

    if (includePairsIfPresent && (preparedPool.pairsData != null)) {
      val cogroupedData = DataHelpers.getCogroupedMainAndPairsRDD(
        selectedDF,
        columnIndexMap("groupId"),
        preparedPool.pairsData
      )
      val pairsSchema = preparedPool.pairsData.schema
      val resultRDD = cogroupedData.mapPartitions {
        groups : Iterator[DataHelpers.PreparedGroupData] => {
          if (groups.hasNext) {
            val localExecutor = new TLocalExecutor
            localExecutor.Init(threadCountForTask)

            val (dataProviders, estimatedFeaturesDataProviders, dstRows) = DataHelpers.loadQuantizedDatasetsWithPairs(
              /*datasetOffset*/ 0,
              /*datasetCount*/ 1,
              quantizedFeaturesInfo,
              columnIndexMap,
              dataMetaInfo,
              schema,
              pairsSchema,
              estimatedFeatureCount,
              localExecutor,
              groups,
              dstColumnIndices,
              dstRowLength
            )
            f(
              dataProviders.get(0),
              if (estimatedFeatureCount.isDefined) { estimatedFeaturesDataProviders.get(0) } else { null },
              dstRows(0),
              localExecutor
            )
          } else {
            Iterator[R]()
          }
        }
      }
      spark.createDataset(resultRDD)
    } else {
      selectedDF.mapPartitions[R]{
        rows : Iterator[Row] => {
          if (rows.hasNext) {
            val localExecutor = new TLocalExecutor
            localExecutor.Init(threadCountForTask)

            val (dataProviders, estimatedFeaturesDataProviders, dstRows) = DataHelpers.loadQuantizedDatasets(
              /*datasetCount*/ 1,
              quantizedFeaturesInfo,
              columnIndexMap,
              dataMetaInfo,
              schema,
              estimatedFeatureCount,
              localExecutor,
              rows,
              dstColumnIndices,
              dstRowLength
            )
            f(
              dataProviders.get(0),
              if (estimatedFeatureCount.isDefined) { estimatedFeaturesDataProviders.get(0) } else { null },
              dstRows(0),
              localExecutor
            )
          } else {
            Iterator[R]()
          }
        }
      }
    }
  }
}
