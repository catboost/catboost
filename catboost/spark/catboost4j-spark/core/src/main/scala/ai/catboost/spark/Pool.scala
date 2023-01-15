package ai.catboost.spark;

import java.nio.file.Path
import java.util.{Arrays,ArrayList,Collections};

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer,ArrayBuilder}

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{Vector,SparseVector,Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._;

import ai.catboost.CatBoostError
import ai.catboost.spark.params.{Helpers,PoolLoadParams,QuantizationParams}
import ai.catboost.spark.params.macros.ParamGetterSetter


class QuantizedRowsOutputIterator(
    var objectIdx : Int,
    val dstRows : ArrayBuffer[Array[Any]],
    val featuresColumnIdx : Int,
    val quantizedRowAssembler : TQuantizedRowAssembler,
    val objectBlobSize : Int
) extends Iterator[Row] {
  def next : Row = {
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

object QuantizedRowsOutputIterator {
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
  
  
  def load(
    spark: SparkSession,
    dataPathWithScheme: String,
    columnDescription: Path = null, // API for Java, so no Option[_] here.
    params: PoolLoadParams = new PoolLoadParams()): Pool = {
    
    val pathParts = dataPathWithScheme.split("://", 2)
    val (dataScheme, dataPath) =
      if (pathParts.size == 1) ("dsv", pathParts(0)) else (pathParts(0), pathParts(1))
      
    val format = dataScheme match {
      case "dsv" | "libsvm" => "ai.catboost.spark.CatBoostTextFileFormat"
      case _ => throw new CatBoostError(s"Loading pool from scheme ${dataScheme} is not supported")
    }
    
    val dataSourceOptions = mutable.Map[String,String]()
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
    
    val data = spark.read.format(format).options(dataSourceOptions).load(dataPath)

    new Pool(if (dataScheme == "libsvm") updateSparseFeaturesSize(data) else data)
  }
  
  def getFeatureCount(data : DataFrame, featuresCol : String) : Int = {
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

  def getFeatureNames(data: DataFrame, featuresCol: String): Array[String] = {
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
}


class Pool (
    override val uid: String,
    val data: DataFrame = null,
    protected var featuresLayout: TFeaturesLayout = null, // updated on demand if not initialized
    val quantizedFeaturesInfo: QuantizedFeaturesInfoPtr = null) 
  extends Params with HasLabelCol with HasFeaturesCol with HasWeightCol {
  
  ensureNativeLibLoaded;
  
  def this(
    data: DataFrame,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr
  ) = 
    this(
      Identifiable.randomUID("catboostPool"),
      data,
      if (quantizedFeaturesInfo != null) quantizedFeaturesInfo.GetFeaturesLayout().__deref__() else null,
      quantizedFeaturesInfo)
  
  def this(data: DataFrame) = this(data, null)
  
  def getFeaturesLayout : TFeaturesLayout = {
    if (featuresLayout == null) {
      if (isQuantized) {
        throw new CatBoostError("featuresLayout must be defined for quantized pool")
      }
      featuresLayout = native_impl.MakeFeaturesLayout(
        Pool.getFeatureCount(data, $(featuresCol)),
        new TVector_TString(Pool.getFeatureNames(data, $(featuresCol))),
        /*ignoredFeatures*/ new TVector_i32()
      )
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
    getFeaturesLayout.GetExternalFeatureCount.toInt
  }
  
  def getFeatureNames : Array[String] = {
    getFeaturesLayout.GetExternalFeatureIds.toArray(new Array[String](0))
  }

  def count : Long = data.count
  
  protected def createQuantizationSchema(quantizationParams: QuantizationParams) 
    : QuantizedFeaturesInfoPtr = {
    
    val dataForBuildBorders =
      if (count > QuantizationParams.MaxSubsetSizeForBuildBordersAlgorithms) {
        data.select(getFeaturesCol).sample(
          QuantizationParams.MaxSubsetSizeForBuildBordersAlgorithms.toDouble / count.toDouble,
          /*seed*/ 0
        )
      } else {
        data.select(getFeaturesCol)
      }
    
    val featureCount = getFeatureCount

    val catBoostJsonParamsString = Helpers.sparkMlParamsToCatBoostJsonParamsString(quantizationParams)
    
    val nanModeAndBordersBuilder = new TNanModeAndBordersBuilder(
        catBoostJsonParamsString,
        featureCount,
        new TVector_TString(getFeatureNames),
        dataForBuildBorders.count.toInt
    )

    for (row <- dataForBuildBorders.toLocalIterator) {
       nanModeAndBordersBuilder.AddSample(row.getAs[Vector](0).toArray)
    }
    
    nanModeAndBordersBuilder.Finish(quantizationParams.getThreadCount)
  }
    
  protected def createQuantized(quantizedFeaturesInfo: QuantizedFeaturesInfoPtr) : Pool = {
    var featuresColumnIdx = data.schema.fieldIndex($(featuresCol));
    val threadCountForTask = SparkHelpers.getThreadCountForTask(data.sparkSession)
    
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
    val quantizedDataEncoder = RowEncoder(quantizedDataSchema)
    
    val quantizedData = data.mapPartitions(
      rowsIterator => {
        ensureNativeLibLoaded;
        
        val availableFeaturesIndices = native_impl.GetAvailableFloatFeatures(
          quantizedFeaturesInfo.GetFeaturesLayout().__deref__()
        ).toPrimitiveArray
        
        // source features column is replaced by quantizedFeatures
        val dstRows = new ArrayBuffer[Array[Any]]
        
        // as columns
        var availableFeaturesData = new Array[ArrayBuilder[Float]](availableFeaturesIndices.size)
        for (i <- 0 until availableFeaturesData.size) {
          availableFeaturesData(i) = ArrayBuilder.make[Float]
        }
        
        rowsIterator.foreach {
          row => {
             val rowFields = new Array[Any](row.length)
             for (i <- 0 until row.length) {
               if (i == featuresColumnIdx) {
                 val featuresValues = row.getAs[Vector](i)
                 for (j <- 0 until availableFeaturesIndices.size) {
                   availableFeaturesData(j) += featuresValues(availableFeaturesIndices(j)).toFloat
                 }
               } else {
                 rowFields(i) = row(i)
               }
             }
             dstRows += rowFields
          }
        }
        
        val availableFeaturesDataForBuilder = new TVector_TMaybeOwningConstArrayHolder_float
        for (featureData <- availableFeaturesData) {
          val result = featureData.result
          //System.err.println("result=" + result)
          //System.err.flush();
          availableFeaturesDataForBuilder.add(result)
        }
        
        val rawObjectsDataProviderPtr = native_impl.CreateRawObjectsDataProvider(
          quantizedFeaturesInfo.GetFeaturesLayout(),
          dstRows.size.toLong,
          availableFeaturesDataForBuilder
        )
        
        // try to force cleanup of no longer used data
        availableFeaturesData = null
        System.gc()
        
        val quantizedObjectsDataProvider = native_impl.Quantize(
          quantizedFeaturesInfo,
          rawObjectsDataProviderPtr,
          threadCountForTask
        )
        
        QuantizedRowsOutputIterator(dstRows, featuresColumnIdx, quantizedObjectsDataProvider)
      }
    )(quantizedDataEncoder)
    
    val quantizedPool = new Pool(quantizedData, quantizedFeaturesInfo)
    copyValues(quantizedPool)
  }
    
  def quantize(quantizationParams: QuantizationParams) : Pool = {
    if (isQuantized) {
      throw new CatBoostError("Pool is already quantized")
    }
    createQuantized(createQuantizationSchema(quantizationParams))
  }
    
  def quantize(quantizedFeaturesInfo: QuantizedFeaturesInfoPtr) : Pool = {
    if (isQuantized) {
      throw new CatBoostError("Pool is already quantized")
    }
    createQuantized(quantizedFeaturesInfo)
  }
}
