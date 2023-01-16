package ai.catboost.spark.impl

import collection.mutable
import collection.mutable.HashMap

import concurrent.duration.Duration
import concurrent.{Await,Future}

import java.net._
import java.util.concurrent.{ExecutorCompletionService,Executors}
import java.util.concurrent.atomic.AtomicBoolean

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.typedLit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame,Row,SparkSession}
import org.apache.spark.TaskContext

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.CatBoostError
import ai.catboost.spark._

import java.io.File


private[spark] object CatBoostWorker {
  val usedInCurrentProcess : AtomicBoolean = new AtomicBoolean
}

private[spark] class CatBoostWorker extends Logging {
  def processPartition(
    trainingDriverListeningAddress: InetSocketAddress,
    catBoostJsonParamsString: String,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    precomputedOnlineCtrMetaDataAsJsonString: String,
    threadCount: Int,
    
    // returns (quantizedDataProviders, estimatedQuantizedDataProviders, dstRows) can return null
    getDataProvidersCallback : (TLocalExecutor) => (TVector_TDataProviderPtr, TVector_TDataProviderPtr, Array[mutable.ArrayBuffer[Array[Any]]])
  ) = {
    if (!CatBoostWorker.usedInCurrentProcess.compareAndSet(false, true)) {
      throw new CatBoostError("An active CatBoost worker is already present in the current process")
    }

    try {
      log.info("processPartition: start")

      val localExecutor = new TLocalExecutor
      localExecutor.Init(threadCount)

      // TaskContext.getPartitionId will become invalid when iteration over rows is finished, so save it
      val partitionId = TaskContext.getPartitionId
      
      log.info("processPartition: get data providers: start")

      var (quantizedDataProviders, estimatedQuantizedDataProviders, _) = getDataProvidersCallback(localExecutor)

      log.info("processPartition: get data providers: finish")
      
      val partitionSize = if (quantizedDataProviders != null) native_impl.GetPartitionTotalObjectCount(quantizedDataProviders).toInt else 0

      if (partitionSize != 0) {
        log.info("processPartition: CreateTrainingDataForWorker: start")
        native_impl.CreateTrainingDataForWorker(
          partitionId,
          threadCount,
          catBoostJsonParamsString,
          quantizedDataProviders,
          quantizedFeaturesInfo,
          estimatedQuantizedDataProviders,
          if (precomputedOnlineCtrMetaDataAsJsonString != null) {
            precomputedOnlineCtrMetaDataAsJsonString
          } else {
            ""
          }
        )
        log.info("processPartition: CreateTrainingDataForWorker: finish")
      }

      val workerPort = TrainingDriver.getWorkerPort()

      val ecs = new ExecutorCompletionService[Unit](Executors.newFixedThreadPool(2))

      val sendWorkerInfoFuture = ecs.submit(
        new Runnable() {
          def run() = {
            TrainingDriver.waitForListeningPortAndSendWorkerInfo(
              trainingDriverListeningAddress,
              partitionId,
              partitionSize,
              workerPort
            )
          }
        },
        ()
      )

      val workerFuture = ecs.submit(
        new Runnable() {
          def run() = {
            if (partitionSize != 0) {
              native_impl.RunWorkerWrapper(threadCount, workerPort)
            }
          }
        },
        ()
      )

      val firstCompletedFuture = ecs.take()

      if (firstCompletedFuture == workerFuture) {
        impl.Helpers.checkOneFutureAndWaitForOther(workerFuture, sendWorkerInfoFuture, "native_impl.RunWorkerWrapper")
      } else { // firstCompletedFuture == sendWorkerInfoFuture
        impl.Helpers.checkOneFutureAndWaitForOther(
          sendWorkerInfoFuture,
          workerFuture,
          "TrainingDriver.waitForListeningPortAndSendWorkerInfo"
        )
      }

    } finally {
      CatBoostWorker.usedInCurrentProcess.set(false)
    }
  }
}


private[spark] class CatBoostWorkers(
  val spark: SparkSession,
  val workerCount: Int,
  val trainingDriverListeningPort: Int,
  val preprocessedTrainPool: Pool,
  val preprocessedEvalPools: Array[Pool],
  val catBoostJsonParams: JObject,
  val precomputedOnlineCtrMetaDataAsJsonString: String,
  
  // needed here because CatBoost master should get all its data before workers 
  // occupy all cores on their executors
  val masterSavedPoolsFuture: Future[(PoolFilesPaths, Array[PoolFilesPaths])]
) extends Runnable {
  def run() = {
    val trainingDriverListeningAddress = new InetSocketAddress(
      SparkHelpers.getDriverHost(spark),
      trainingDriverListeningPort
    )

    val quantizedFeaturesInfo = preprocessedTrainPool.quantizedFeaturesInfo

    val (trainDataForWorkers, columnIndexMapForWorkers, estimatedFeatureCount) 
      = DataHelpers.selectColumnsForTrainingAndReturnIndex(
          preprocessedTrainPool,
          includeFeatures = true,
          includeSampleId = (preprocessedTrainPool.pairsData != null),
          includeEstimatedFeatures = true
        )
    val evalDataForWorkers = preprocessedEvalPools.map {
      evalPool => {
        val (evalDatasetForWorkers, _, _) 
          = DataHelpers.selectColumnsForTrainingAndReturnIndex(
              evalPool,
              includeFeatures = true,
              includeSampleId = (evalPool.pairsData != null),
              includeEstimatedFeatures = true
            )
        evalDatasetForWorkers
      }
    }

    val threadCount = SparkHelpers.getThreadCountForTask(spark)

    var catBoostJsonParamsForWorkers = catBoostJsonParams ~ ("thread_count" -> threadCount)

    val executorNativeMemoryLimit = SparkHelpers.getExecutorNativeMemoryLimit(spark)
    if (executorNativeMemoryLimit.isDefined) {
      catBoostJsonParamsForWorkers
        = catBoostJsonParamsForWorkers ~ ("used_ram_limit" -> s"${executorNativeMemoryLimit.get / 1024}KB")
    }

    val catBoostJsonParamsForWorkersString = compact(catBoostJsonParamsForWorkers)

    val dataMetaInfo = preprocessedTrainPool.createDataMetaInfo()
    val schemaForWorkers = trainDataForWorkers.schema

    // copy needed because Spark will try to capture the whole CatBoostWorkers class and fail
    val precomputedOnlineCtrMetaDataAsJsonStringCopy = precomputedOnlineCtrMetaDataAsJsonString
    val totalDatasetCount = 1 + preprocessedEvalPools.size
    
    if (preprocessedTrainPool.pairsData != null) {
      val cogroupedTrainData = getCogroupedMainAndPairsRDDForAllDatasets(
        trainDataForWorkers, 
        preprocessedTrainPool.pairsData,
        evalDataForWorkers,
        preprocessedEvalPools.map(evalPool => evalPool.pairsData)
      ).repartition(workerCount).cache()
      
      // Force cache before starting worker processes
      cogroupedTrainData.count()
      
      // make sure CatBoost master downloaded all the necessary data from cluster before starting worker processes
      Await.result(masterSavedPoolsFuture, Duration.Inf)
      
      val pairsSchema = preprocessedTrainPool.pairsData.schema

      cogroupedTrainData.foreachPartition {
        groups : Iterator[((Byte, Long), (Iterable[Iterable[Row]], Iterable[Iterable[Row]]))] => {
          new CatBoostWorker().processPartition(
            trainingDriverListeningAddress,
            catBoostJsonParamsForWorkersString,
            quantizedFeaturesInfo,
            precomputedOnlineCtrMetaDataAsJsonStringCopy,
            threadCount,
            (localExecutor: TLocalExecutor) => {
              if (groups.hasNext) {
                DataHelpers.loadQuantizedDatasetsWithPairs(
                  totalDatasetCount,
                  quantizedFeaturesInfo,
                  columnIndexMapForWorkers,
                  dataMetaInfo,
                  schemaForWorkers,
                  pairsSchema,
                  estimatedFeatureCount,
                  localExecutor,
                  groups
                )
              } else {
                (null, null, null)
              }
           }
          )
        }
      }
    } else {
      val mergedTrainData = getMergedDataFrameForAllDatasets(
        trainDataForWorkers,
        evalDataForWorkers
      ).repartition(workerCount).cache()

      columnIndexMapForWorkers += ("_datasetIdx" -> columnIndexMapForWorkers.size)

      // Force cache before starting worker processes
      mergedTrainData.count()
      
      // make sure CatBoost master downloaded all the necessary data from cluster before starting worker processes
      Await.result(masterSavedPoolsFuture, Duration.Inf)
      
      mergedTrainData.foreachPartition {
        rows : Iterator[Row] => {
          new CatBoostWorker().processPartition(
            trainingDriverListeningAddress,
            catBoostJsonParamsForWorkersString,
            quantizedFeaturesInfo,
            precomputedOnlineCtrMetaDataAsJsonStringCopy,
            threadCount,
            (localExecutor: TLocalExecutor) => {
              if (rows.hasNext) {
                DataHelpers.loadQuantizedDatasets(
                  totalDatasetCount,
                  quantizedFeaturesInfo,
                  columnIndexMapForWorkers,
                  dataMetaInfo,
                  schemaForWorkers,
                  estimatedFeatureCount,
                  localExecutor,
                  rows
                )
              } else {
                (null, null, null)
              }
            }
          )
        }
      }
    }
  }


  private def getMergedDataFrameForAllDatasets(
    trainDataForWorkers: DataFrame,
    evalDataForWorkers: Seq[DataFrame]
  ) : DataFrame = {
    var mergedDataForWorkers = trainDataForWorkers.withColumn("_datasetIdx", typedLit(0.toByte))
    evalDataForWorkers.zipWithIndex.foreach {
      case (evalDataset, evalIdx) => {
        mergedDataForWorkers = mergedDataForWorkers.union(
          evalDataset.withColumn("_datasetIdx", typedLit((evalIdx + 1).toByte))
        )
      }
    }
    mergedDataForWorkers
  }

  private def getCogroupedMainAndPairsRDDForAllDatasets(
    trainDataForWorkers: DataFrame,
    trainPairsData: DataFrame,
    evalDataForWorkers: Seq[DataFrame],
    evalPairsData: Seq[DataFrame]
  ) : RDD[((Byte, Long), (Iterable[Iterable[Row]], Iterable[Iterable[Row]]))] = {
      val mainDataGroupIdIdx = trainDataForWorkers.schema.fieldIndex("groupId")
      val pairsGroupIdIdx = trainPairsData.schema.fieldIndex("groupId")

      val datasetIdx = 0.toByte
      var mergedMainDataRDD = trainDataForWorkers.rdd.groupBy(
        row => (datasetIdx, row.getLong(mainDataGroupIdIdx))
      )
      var mergedPairsDataRDD = trainPairsData.rdd.groupBy(
        row => (datasetIdx, row.getLong(pairsGroupIdIdx))
      )

      evalDataForWorkers.zip(evalPairsData).zipWithIndex.foreach {
        case ((mainData, pairsData), evalIdx) => {
          val datasetIdx = (evalIdx + 1).toByte
          mergedMainDataRDD = mergedMainDataRDD.union(
            mainData.rdd.groupBy(
              row => (datasetIdx, row.getLong(mainDataGroupIdIdx))
            )
          )
          mergedPairsDataRDD = mergedPairsDataRDD.union(
            pairsData.rdd.groupBy(
              row => (datasetIdx, row.getLong(pairsGroupIdIdx))
            )
          )
        }
      }
      
      mergedMainDataRDD.cogroup(mergedPairsDataRDD)
  }
}
