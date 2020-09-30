package ai.catboost.spark.impl

import collection.mutable.HashMap

import java.net._

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row,SparkSession}
import org.apache.spark.TaskContext

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.spark._


private[spark] object Worker {
  def processPartition(
    trainingDriverListeningAddress: InetSocketAddress,
    catBoostJsonParamsString: String,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    columnsIndexMap: HashMap[String, Int], // column type -> idx in schema
    dataMetaInfo: TIntermediateDataMetaInfo,
    schema: StructType,
    threadCount: Int,
    rows: Iterator[Row]
  ) = {
    var quantizedDataProvider : TDataProviderPtr = null

    if (rows.hasNext) {
      quantizedDataProvider = DataHelpers.loadQuantizedDataset(
        quantizedFeaturesInfo,
        columnsIndexMap,
        dataMetaInfo,
        schema,
        threadCount,
        rows
      )
    }

    val partitionSize = if (quantizedDataProvider != null) quantizedDataProvider.GetObjectCount.toInt else 0

    val workerPort = TrainingDriver.getWorkerPortAndSendWorkerInfo(
      trainingDriverListeningAddress,
      TaskContext.getPartitionId,
      partitionSize
    )

    if (partitionSize != 0) {
      native_impl.RunWorker(
        TaskContext.getPartitionId,
        workerPort,
        threadCount,
        catBoostJsonParamsString,
        quantizedDataProvider,
        quantizedFeaturesInfo
      )
    }
  }
}


private[spark] class Workers(
  val spark: SparkSession,
  val trainingDriverListeningPort: Int,
  val preprocessedTrainPool: Pool,
  val catBoostJsonParams: JObject
) extends Runnable {
  def run() = {
    val trainingDriverListeningAddress = new InetSocketAddress(
      SparkHelpers.getDriverHost(spark),
      trainingDriverListeningPort
    )

    val quantizedFeaturesInfo = preprocessedTrainPool.quantizedFeaturesInfo

    val (trainDataForWorkers, columnIndexMapForWorkers) = DataHelpers.selectColumnsForTrainingAndReturnIndex(
      preprocessedTrainPool,
      includeFeatures = true
    )

    val threadCount = SparkHelpers.getThreadCountForTask(spark)

    var catBoostJsonParamsForWorkers = catBoostJsonParams ~ ("thread_count" -> threadCount)

    val executorNativeMemoryLimit = SparkHelpers.getExecutorNativeMemoryLimit(spark)
    if (executorNativeMemoryLimit.isDefined) {
      catBoostJsonParamsForWorkers
        = catBoostJsonParamsForWorkers ~ ("used_ram_limit" -> executorNativeMemoryLimit.get)
    }

    val catBoostJsonParamsForWorkersString = compact(catBoostJsonParamsForWorkers)

    val dataMetaInfo = preprocessedTrainPool.createDataMetaInfo
    val schemaForWorkers = trainDataForWorkers.schema

    trainDataForWorkers.foreachPartition {
      rows => {
        Worker.processPartition(
          trainingDriverListeningAddress,
          catBoostJsonParamsForWorkersString,
          quantizedFeaturesInfo,
          columnIndexMapForWorkers,
          dataMetaInfo,
          schemaForWorkers,
          threadCount,
          rows
        )
      }
    }
  }
}
