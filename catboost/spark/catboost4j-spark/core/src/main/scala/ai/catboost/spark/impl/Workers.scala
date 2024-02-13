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
import org.apache.spark.{SparkContext,TaskContext}

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.CatBoostError
import ai.catboost.spark._

import java.io.File


private[spark] object CatBoostWorker {
  val usedInCurrentProcess : AtomicBoolean = new AtomicBoolean
}

private[spark] class CatBoostWorker(partitionId : Int) extends Logging {
  // Method to get the logger name for this object
  protected override def logName = {
    s"CatBoostWorker[partitionId=${this.partitionId}]"
  }

  def processPartition(
    trainingDriverListeningAddress: InetSocketAddress,
    catBoostJsonParamsString: String,
    serializedLabelConverter: TVector_i8,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    precomputedOnlineCtrMetaDataAsJsonString: String,
    threadCount: Int,
    connectTimeout: java.time.Duration,
    workerInitializationTimeout: java.time.Duration,
    workerListeningPortParam: Int, // auto-assign if 0

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

      log.info("processPartition: get data providers: start")

      var (quantizedDataProviders, estimatedQuantizedDataProviders, _) = getDataProvidersCallback(localExecutor)

      log.info("processPartition: get data providers: finish")

      val partitionSize = if (quantizedDataProviders != null) native_impl.GetPartitionTotalObjectCount(quantizedDataProviders).toInt else 0

      if (partitionSize != 0) {
        log.info("processPartition: CreateTrainingDataForWorker: start")
        native_impl.CreateTrainingDataForWorker(
          this.partitionId,
          threadCount,
          catBoostJsonParamsString,
          serializedLabelConverter,
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
      } else {
        log.info("processPartition: data is empty")
      }

      val workerListeningPort = if (workerListeningPortParam != 0) { workerListeningPortParam } else { TrainingDriver.getWorkerPort() }

      val ecs = new ExecutorCompletionService[Unit](Executors.newFixedThreadPool(2))

      val partitionId = this.partitionId
      val sendWorkerInfoFuture = ecs.submit(
        new Runnable() {
          def run() = {
            TrainingDriver.waitForListeningPortAndSendWorkerInfo(
              trainingDriverListeningAddress,
              partitionId,
              partitionSize,
              workerListeningPort,
              connectTimeout,
              workerInitializationTimeout
            )
          }
        },
        ()
      )

      val workerFuture = ecs.submit(
        new Runnable() {
          def run() = {
            if (partitionSize != 0) {
              log.info("processPartition: start RunWorker")
              native_impl.RunWorker(threadCount, workerListeningPort)
              log.info("processPartition: end RunWorker")
            }
          }
        },
        ()
      )

      try {
        impl.Helpers.waitForTwoFutures(
          ecs,
          workerFuture,
          "native_impl.RunWorker",
          sendWorkerInfoFuture,
          "TrainingDriver.waitForListeningPortAndSendWorkerInfo"
        )
      } catch {
        case e : java.util.concurrent.ExecutionException => {
          e.getCause match {
            case connectException : CatBoostTrainingDriverConnectException => {
              /*
               *  Do not propagate it because:
               *  1) the reason for this error can be that this partition
               *    processing has been restarted by Spark but TrainingDriver had already failed.
               *    If it is a first time connection error it will be detected by TrainingDriver anyway
               *    (by not getting any info from this worker)
               *  2) We can count other worker errors (for max failures check) separately from connection errors
               */
              log.info(connectException.toString)
            }
            case _ => throw e
          }
        }
      }

      log.info("processPartition: end")

    } finally {
      CatBoostWorker.usedInCurrentProcess.set(false)
    }
  }
}


private[spark] class CatBoostWorkers (
    val sparkContext: SparkContext,
    val processPartitionsCallback : (Int) => Unit, // (trainingDriverListeningPort : Int) => Unit

    // count failed stage attempts as one worker failure, no easy way/need for more granularity for now
    var workerFailureCount : Int = 0
) {
  // can be called multiple times
  def run(trainingDriverListeningPort: Int) = {
    val jobGroup = "CatBoostWorkers." + Thread.currentThread.getId.toString
    sparkContext.setJobGroup(jobGroup, jobGroup)

    try {
      processPartitionsCallback(trainingDriverListeningPort)
      val failedStats = SparkJobGroupLastStagesFailedStats(sparkContext, jobGroup)
      workerFailureCount += failedStats.failedAttempts + failedStats.failedTasksInLastAttempt
    } catch {
      case e : Throwable => {
        sparkContext.cancelJobGroup(jobGroup)
        throw e
      }
    } finally {
      sparkContext.clearJobGroup()
    }
  }
}

private[spark] object CatBoostWorkers {
  def apply(
    spark: SparkSession,
    workerCount: Int,
    connectTimeout: java.time.Duration,
    workerInitializationTimeout: java.time.Duration,
    workerListeningPort: Int,
    preparedTrainPool: DatasetForTraining,
    preparedEvalPools: Seq[DatasetForTraining],
    catBoostJsonParams: JObject,
    serializedLabelConverter: TVector_i8,
    precomputedOnlineCtrMetaDataAsJsonString: String,

    // needed here because CatBoost master should get all its data before workers
    // occupy all cores on their executors
    masterSavedPoolsFuture: Future[(PoolFilesPaths, Array[PoolFilesPaths])]
  ) : CatBoostWorkers = {
    val quantizedFeaturesInfo = preparedTrainPool.srcPool.quantizedFeaturesInfo

    val (trainDataForWorkers, columnIndexMapForWorkers, estimatedFeatureCount)
      = DataHelpers.selectColumnsForTrainingAndReturnIndex(
          preparedTrainPool,
          includeFeatures = true,
          includeSampleId = preparedTrainPool.isInstanceOf[DatasetForTrainingWithPairs],
          includeEstimatedFeatures = true,
          includeDatasetIdx = true
        )
    val evalDataForWorkers = preparedEvalPools.map {
      evalPool => {
        val (evalDatasetForWorkers, _, _)
          = DataHelpers.selectColumnsForTrainingAndReturnIndex(
              evalPool,
              includeFeatures = true,
              includeSampleId = evalPool.isInstanceOf[DatasetForTrainingWithPairs],
              includeEstimatedFeatures = true,
              includeDatasetIdx = true
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

    val dataMetaInfo = preparedTrainPool.srcPool.createDataMetaInfo()
    val schemaForWorkers = trainDataForWorkers.mainDataSchema

    // copies are needed because Spark will try to capture the whole CatBoostWorkers class and fail
    val connectTimeoutCopy = connectTimeout
    val workerInitializationTimeoutCopy = workerInitializationTimeout
    val workerListeningPortParamCopy = workerListeningPort
    val precomputedOnlineCtrMetaDataAsJsonStringCopy = precomputedOnlineCtrMetaDataAsJsonString

    val totalDatasetCount = 1 + evalDataForWorkers.size

    val processPartitionsCallback = if (trainDataForWorkers.isInstanceOf[DatasetForTrainingWithPairs]) {
      val cogroupedTrainData = getCogroupedMainAndPairsRDDForAllDatasets(
        trainDataForWorkers.asInstanceOf[DatasetForTrainingWithPairs],
        evalDataForWorkers.map{ _.asInstanceOf[DatasetForTrainingWithPairs] }
      ).cache()

      // Force cache before starting worker processes
      cogroupedTrainData.count()

      // make sure CatBoost master downloaded all the necessary data from cluster before starting worker processes
      Await.result(masterSavedPoolsFuture, Duration.Inf)

      val pairsSchema = preparedTrainPool.srcPool.pairsData.schema

      (trainingDriverListeningPort : Int) => {
        val trainingDriverListeningAddress = new InetSocketAddress(
          SparkHelpers.getDriverHost(spark),
          trainingDriverListeningPort
        )
        cogroupedTrainData.foreachPartition {
          groups : Iterator[DataHelpers.PreparedGroupData] => {
            new CatBoostWorker(TaskContext.getPartitionId).processPartition(
              trainingDriverListeningAddress,
              catBoostJsonParamsForWorkersString,
              serializedLabelConverter,
              quantizedFeaturesInfo,
              precomputedOnlineCtrMetaDataAsJsonStringCopy,
              threadCount,
              connectTimeoutCopy,
              workerInitializationTimeoutCopy,
              workerListeningPortParamCopy,
              (localExecutor: TLocalExecutor) => {
                if (groups.hasNext) {
                  DataHelpers.loadQuantizedDatasetsWithPairs(
                    /*datasetOffset*/ 0,
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
      }
    } else {
      val mergedTrainData = getMergedDataFrameForAllDatasets(
        trainDataForWorkers.asInstanceOf[UsualDatasetForTraining],
        evalDataForWorkers.map{ _.asInstanceOf[UsualDatasetForTraining] }
      ).cache()

      // Force cache before starting worker processes
      mergedTrainData.count()

      // make sure CatBoost master downloaded all the necessary data from cluster before starting worker processes
      Await.result(masterSavedPoolsFuture, Duration.Inf)

      (trainingDriverListeningPort : Int) => {
        val trainingDriverListeningAddress = new InetSocketAddress(
          SparkHelpers.getDriverHost(spark),
          trainingDriverListeningPort
        )
        mergedTrainData.foreachPartition {
          rows : Iterator[Row] => {
            new CatBoostWorker(TaskContext.getPartitionId).processPartition(
              trainingDriverListeningAddress,
              catBoostJsonParamsForWorkersString,
              serializedLabelConverter,
              quantizedFeaturesInfo,
              precomputedOnlineCtrMetaDataAsJsonStringCopy,
              threadCount,
              connectTimeoutCopy,
              workerInitializationTimeoutCopy,
              workerListeningPortParamCopy,
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

    new CatBoostWorkers(spark.sparkContext, processPartitionsCallback)
  }


  private def getMergedDataFrameForAllDatasets(
    trainDataForWorkers: UsualDatasetForTraining,
    evalDataForWorkers: Seq[UsualDatasetForTraining]
  ) : RDD[Row] = {
    var mergedData = trainDataForWorkers.data.rdd

    for (evalDataset <- evalDataForWorkers) {
      mergedData = mergedData.zipPartitions(evalDataset.data.rdd, preservesPartitioning=true){
        (iterator1, iterator2) => iterator1 ++ iterator2
      }
    }

    mergedData
  }

  private def getCogroupedMainAndPairsRDDForAllDatasets(
    trainDataForWorkers: DatasetForTrainingWithPairs,
    evalDataForWorkers: Seq[DatasetForTrainingWithPairs]
  ) : RDD[DataHelpers.PreparedGroupData] = {
    var mergedData = trainDataForWorkers.data

    for (evalDataset <- evalDataForWorkers) {
      mergedData = mergedData.zipPartitions(evalDataset.data, preservesPartitioning=true){
        (iterator1, iterator2) => iterator1 ++ iterator2
      }
    }

    mergedData
  }
}
