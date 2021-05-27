package ai.catboost.spark.impl

import java.io.File

import scala.collection.JavaConverters._

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import ai.catboost.CatBoostError

import ai.catboost.spark._
import ai.catboost.spark.params.TrainingParamsTrait

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._


class CtrsContext(
    val catBoostOptions: TCatBoostOptions,
    val ctrHelper: SWIGTYPE_p_TCtrHelper,
    val targetStats: TTargetStatsForCtrs,
    val learnTarget: Array[Float],
    val precomputedOnlineCtrMetaDataAsJsonString: String,
    val localExecutor: TLocalExecutor
)


object CtrFeatures {
  protected def getDatasetWithIdsAndIds(df: DataFrame) : (DataFrame, Array[Long]) = {
    val dfWithId = df.withColumn("_id", monotonicallyIncreasingId)
    (dfWithId, dfWithId.select("_id").toLocalIterator().asScala.map(row => row.getLong(0)).toArray)
  }
  
  /**
   * @return array of flat feature indices
   */
  protected def getCatFeatureFlatIndicesForCtrs(pool: Pool, oneHotMaxSize: Int) : Array[Int] = {
    pool.getCatFeaturesUniqValueCounts.zipWithIndex.collect{
      case (uniqValueCount, i) if (uniqValueCount > oneHotMaxSize) => i
    }
  }
  
  protected def getLearnTarget(pool: Pool) : Array[Float] = {
    val spark = pool.data.sparkSession
    import spark.implicits._
    
    val labelDf = pool.data.select(pool.getLabelCol)
    (labelDf.schema(0).dataType match {
      case IntegerType => { labelDf.map(row => row.getAs[Int](0).toFloat) }
      case LongType => { labelDf.map(row => row.getAs[Long](0).toFloat) }
      case FloatType => { labelDf.map(row => row.getAs[Float](0)) }
      case DoubleType => { labelDf.map(row => row.getAs[Double](0).toFloat) }
      case StringType => { labelDf.map(row => row.getAs[String](0).toFloat) }
      case _ => throw new CatBoostError("Unsupported data type for Label")
    }).toLocalIterator.asScala.toArray
  }
  
  def downloadSubsetOfQuantizedFeatures(
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool],
    quantizedFeaturesIndices: QuantizedFeaturesIndices,
    selectedFlatFeaturesIndices: Set[Int],
    localExecutor: TLocalExecutor
  ) : (TQuantizedObjectsDataProviderPtr, TVector_TQuantizedObjectsDataProviderPtr) = {
    (
      DataHelpers.downloadSubsetOfQuantizedFeatures(
        quantizedTrainPool,
        quantizedFeaturesIndices,
        selectedFlatFeaturesIndices,
        localExecutor
      ),
      new TVector_TQuantizedObjectsDataProviderPtr(
        quantizedEvalPools.map(
          evalPool => {
            DataHelpers.downloadSubsetOfQuantizedFeatures(
              evalPool,
              quantizedFeaturesIndices,
              selectedFlatFeaturesIndices,
              localExecutor
            )
          }
        )
      )
    )
  }
  
  protected def uploadAndMerge(
    spark: SparkSession,
    schema: StructType,
    aggregateData: DataFrame,
    ids: Array[Long],
    estimatedData: TQuantizedObjectsDataProviderPtr
  ) : DataFrame = {
    val rowAssembler = new TQuantizedRowAssembler(estimatedData)
    val buffer = new Array[Byte](rowAssembler.GetObjectBlobSize())
    val dataToUpload = (0 until ids.length).map(
      i => {
        rowAssembler.AssembleObjectBlob(i, buffer)
        Row(ids(i), buffer)
      }
    )
    val df = spark.createDataFrame(spark.sparkContext.parallelize(dataToUpload), schema)
    if (aggregateData == null) {
      df
    } else {
      aggregateData.joinWith(df, aggregateData("_id") === df("_id")).map{
        case (row0, row1) => Row(row0.getLong(0), row0.getAs[Array[Byte]](1) ++ row1.getAs[Array[Byte]](1))
      }(RowEncoder(schema))
    }
  }


  /**
   * @return (trainPoolWithEstimatedFeatures, evalPoolsWithEstimatedFeatures, ctrsContext)
   */
  def addCtrsAsEstimated(
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool],
    params: TrainingParamsTrait,
    oneHotMaxSize: Int
  ) : (Pool, Array[Pool], CtrsContext) = {
    val spark = quantizedTrainPool.data.sparkSession
    
    // Cache pools data because it's heavily reused here
    quantizedTrainPool.data.cache()
    quantizedEvalPools.map(evalPool => evalPool.data.cache())
    
    val catBoostJsonParams = ai.catboost.spark.params.Helpers.sparkMlParamsToCatBoostJsonParams(params)
   
    var learnTarget = getLearnTarget(quantizedTrainPool)
    
    val catBoostOptions = new TCatBoostOptions(ETaskType.CPU)
    native_impl.InitCatBoostOptions(compact(catBoostJsonParams), catBoostOptions)
    val ctrHelper = native_impl.GetCtrHelper(
      catBoostOptions,
      quantizedTrainPool.getFeaturesLayout.__deref__(),
      learnTarget
    )
    
    val localExecutor = new TLocalExecutor
    localExecutor.Init(SparkHelpers.getThreadCountForDriver(spark))
    
    val targetStatsForCtrs = native_impl.ComputeTargetStatsForCtrs(ctrHelper, learnTarget, localExecutor)
    
    val (trainWithIds, trainIds) = getDatasetWithIdsAndIds(quantizedTrainPool.data)
    
    val (evalsWithIds, evalsIds) = quantizedEvalPools.map(
      evalPool => getDatasetWithIdsAndIds(evalPool.data)
    ).unzip
    
    val catFeaturesFlatIndicesForCtrs = getCatFeatureFlatIndicesForCtrs(quantizedTrainPool, oneHotMaxSize)
    
    val quantizedFeaturesInfo = quantizedTrainPool.quantizedFeaturesInfo
    val quantizedFeaturesIndices = QuantizedFeaturesIndices(
      quantizedFeaturesInfo.GetFeaturesLayout,
      quantizedFeaturesInfo
    )
    
    var aggregatedMetaData : TPrecomputedOnlineCtrMetaData = null
    var aggregateEstimatedTrainData : DataFrame = null
    var aggregateEstimatedEvalsData : Array[DataFrame] = new Array[DataFrame](quantizedEvalPools.length)
    val estimatedDataSchema = StructType(
      Seq(
        StructField("_id", LongType),
        StructField("_estimatedFeatures", BinaryType)
      )
    )
    
    for (catFeatureFlatIdx <- catFeaturesFlatIndicesForCtrs) {
      val (trainColumnData, evalsColumnData) = downloadSubsetOfQuantizedFeatures(
        quantizedTrainPool,
        quantizedEvalPools,
        quantizedFeaturesIndices,
        Set[Int](catFeatureFlatIdx),
        localExecutor
      )
      
      val estimatedData = new TEstimatedForCPUObjectsDataProviders
      val estimatedMetaData = new TPrecomputedOnlineCtrMetaData
      
      native_impl.ComputeEstimatedCtrFeatures(
        ctrHelper,
        catBoostOptions,
        targetStatsForCtrs,
        trainColumnData,
        evalsColumnData,
        localExecutor,
        estimatedData, 
        estimatedMetaData
      )
      if (aggregatedMetaData == null) {
        aggregatedMetaData = estimatedMetaData
      } else {
        aggregatedMetaData.Append(estimatedMetaData)
      }
      aggregateEstimatedTrainData = uploadAndMerge(
        spark,
        estimatedDataSchema,
        aggregateEstimatedTrainData,
        trainIds,
        estimatedData.getLearn
      )
      for (i <- 0 until quantizedEvalPools.length) {
        aggregateEstimatedEvalsData(i) = uploadAndMerge(
          spark,
          estimatedDataSchema,
          aggregateEstimatedEvalsData(i),
          evalsIds(i),
          estimatedData.getTest().get(i)
        )
      }
    }
    val trainPoolWithEstimatedFeatures = new Pool(
      trainWithIds.join(aggregateEstimatedTrainData, "_id"),
      quantizedTrainPool.pairsData,
      quantizedFeaturesInfo,
      false
    )
    val evalPoolsWithEstimatedFeatures = (0 until quantizedEvalPools.length).map{
      i => {
        new Pool(
          evalsWithIds(i).join(aggregateEstimatedEvalsData(i), "_id"),
          quantizedEvalPools(i).pairsData,
          quantizedEvalPools(i).quantizedFeaturesInfo,
          false
        )
      }
    }.toArray
    
    quantizedTrainPool.data.unpersist()
    quantizedEvalPools.map(evalPool => evalPool.data.unpersist())
      
    (
      trainPoolWithEstimatedFeatures,
      evalPoolsWithEstimatedFeatures,
      new CtrsContext(
        catBoostOptions,
        ctrHelper, 
        targetStatsForCtrs,
        learnTarget,
        aggregatedMetaData.SerializeToJson(), 
        localExecutor
      )
    )
  }
  
  
  def addCtrProviderToModel(
    model: TFullModel,
    ctrsContext: CtrsContext, // moved into
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool]
  ) : TFullModel = {
    val quantizedFeaturesInfo = quantizedTrainPool.quantizedFeaturesInfo
    val quantizedFeaturesIndices = QuantizedFeaturesIndices(
      quantizedFeaturesInfo.GetFeaturesLayout,
      quantizedFeaturesInfo
    )
    
    val finalCtrsCalcer = new TFinalCtrsCalcer(
      model,
      ctrsContext.catBoostOptions,
      quantizedFeaturesInfo.__deref__,
      ctrsContext.learnTarget,
      ctrsContext.targetStats,
      ctrsContext.ctrHelper,
      ctrsContext.localExecutor
    )
    val catFeatureFlatIndicesUsedForCtrs = finalCtrsCalcer.GetCatFeatureFlatIndicesUsedForCtrs.toPrimitiveArray
    
    for (catFeatureFlatIdx <- catFeatureFlatIndicesUsedForCtrs) {
      val (trainColumnData, evalsColumnData) = downloadSubsetOfQuantizedFeatures(
        quantizedTrainPool,
        quantizedEvalPools,
        quantizedFeaturesIndices,
        Set[Int](catFeatureFlatIdx),
        ctrsContext.localExecutor
      )
      finalCtrsCalcer.ProcessForFeature(catFeatureFlatIdx, trainColumnData, evalsColumnData)
    }
    
    finalCtrsCalcer.GetModelWithCtrData
  }
}