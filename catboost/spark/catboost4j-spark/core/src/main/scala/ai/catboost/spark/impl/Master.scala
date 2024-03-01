package ai.catboost.spark.impl

import collection.mutable
import concurrent.duration.Duration
import concurrent.{Await,Future}
import concurrent.ExecutionContext.Implicits.global

import scala.util.control.Breaks._

import java.io.{BufferedReader,InputStreamReader}
import java.nio.charset.StandardCharsets
import java.nio.file._

import java.util.concurrent.Callable
import java.util.regex.Pattern

import org.apache.commons.io.FileUtils

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl

import ai.catboost.CatBoostError
import ai.catboost.spark._

private[spark] object CatBoostMasterWrapper {
  // use this method to create Master instances
  def apply(
    preparedTrainPool: DatasetForTraining,
    preparedEvalPools: Seq[DatasetForTraining],
    catBoostJsonParamsForMasterString: String,
    precomputedOnlineCtrMetaDataAsJsonString: String
  ) : CatBoostMasterWrapper = {
    val spark = preparedTrainPool.srcPool.data.sparkSession

    val result = new CatBoostMasterWrapper(
      spark,
      catBoostJsonParamsForMasterString,
      precomputedOnlineCtrMetaDataAsJsonString
    )

    result.savedPoolsFuture = Future {
      val threadCount = SparkHelpers.getThreadCountForDriver(spark)
      val localExecutor = new native_impl.TLocalExecutor
      localExecutor.Init(threadCount)

      val trainPoolFiles = DataHelpers.downloadQuantizedPoolToTempFiles(
        preparedTrainPool,
        includeFeatures=false,
        includeEstimatedFeatures=false,
        localExecutor=localExecutor,
        dataPartName="Learn Dataset",
        log=result.log
      )
      val testPoolsFiles = preparedEvalPools.zipWithIndex.map {
        case (testPool, idx) => DataHelpers.downloadQuantizedPoolToTempFiles(
          testPool,
          includeFeatures=false,
          includeEstimatedFeatures=false,
          localExecutor=localExecutor,
          dataPartName=s"Eval Dataset #${idx}",
          log=result.log
        )
      }.toArray

      (trainPoolFiles, testPoolsFiles)
    }

    result
  }
}


private[spark] class CatBoostMasterWrapper (
  val spark: SparkSession,
  val catBoostJsonParamsForMasterString: String,
  val precomputedOnlineCtrMetaDataAsJsonString: String,

  var savedPoolsFuture : Future[(PoolFilesPaths, Array[PoolFilesPaths])] = null, // inited later

  // will be set in trainCallback, called from the trainingDriver's run()
  var nativeModelResult : native_impl.TFullModel = null
) extends Logging {

  /**
   * If master failed because of lost connection to workers throws  CatBoostWorkersConnectionLostException
   */
  def trainCallback(workersInfo: Array[WorkerInfo])  = {
    if (nativeModelResult != null) {
      throw new CatBoostError(
        "[Internal error] trainCallback is called again despite nativeModelResult already assigned"
      )
    }

    val tmpDirPath = Files.createTempDirectory("catboost_train")

    val hostsFilePath = tmpDirPath.resolve("worker_hosts.txt")
    TrainingDriver.saveHostsListToFile(hostsFilePath, workersInfo)
    val resultModelFilePath = tmpDirPath.resolve("result_model.cbm")

    val jsonParamsFile = tmpDirPath.resolve("json_params")
    Files.write(jsonParamsFile, catBoostJsonParamsForMasterString.getBytes(StandardCharsets.UTF_8))

    var precomputedOnlineCtrMetaDataFile: Path = null
    if (precomputedOnlineCtrMetaDataAsJsonString != null) {
      precomputedOnlineCtrMetaDataFile =  tmpDirPath.resolve("precomputed_online_ctr_metadata")
      Files.write(
        precomputedOnlineCtrMetaDataFile,
        precomputedOnlineCtrMetaDataAsJsonString.getBytes(StandardCharsets.UTF_8)
      )
    }

    val args = mutable.ArrayBuffer[String](
      "--node-type", "Master",
      "--thread-count", SparkHelpers.getThreadCountForDriver(spark).toString,
      "--params-file", jsonParamsFile.toString,
      "--file-with-hosts", hostsFilePath.toString,
      "--hosts-already-contain-loaded-data",
      /* permutations on master are impossible when data is preloaded on hosts, shuffling is performed in Spark
       * on the preprocessing phase
       */
      "--has-time",
      "--max-ctr-complexity", "1",
      "--final-ctr-computation-mode", "Skip", // final ctrs are computed in post-processing
      "--model-file", resultModelFilePath.toString
    )

    val driverNativeMemoryLimit = SparkHelpers.getDriverNativeMemoryLimit(spark)
    if (driverNativeMemoryLimit.isDefined) {
      args += ("--used-ram-limit", driverNativeMemoryLimit.get.toString)
    }
    if (precomputedOnlineCtrMetaDataAsJsonString != null) {
      args += ("--precomputed-data-meta", precomputedOnlineCtrMetaDataFile.toString)
    }

    log.info("Wait until Dataset data parts are ready.")

    val (savedTrainPool, savedEvalPools) = Await.result(savedPoolsFuture, Duration.Inf)

    log.info("Dataset data parts are ready. Start CatBoost Master process.")

    args += ("--learn-set", "spark-quantized://master-part:" + savedTrainPool.mainData.toString)
    if (savedTrainPool.pairsData.isDefined) {
      args += ("--learn-pairs", "dsv-grouped-with-idx://" + savedTrainPool.pairsData.get.toString)
    }
    if (!savedEvalPools.isEmpty) {
      args += (
        "--test-set",
        savedEvalPools.map(
            poolFilesPaths => "spark-quantized://master-part:" + poolFilesPaths.mainData
        ).mkString(",")
      )
      if (savedTrainPool.pairsData.isDefined) { // if train pool has pairs so do test pools
        args += (
          "--test-pairs",
          savedEvalPools.map(
              poolFilesPaths => "dsv-grouped-with-idx://" + poolFilesPaths.pairsData.get.toString
          ).mkString(",")
        )
      }
    }

    val masterAppProcess = RunClassInNewProcess(
      MasterApp.getClass,
      args = Some(args.toArray),
      inheritIO=false,
      redirectOutput = Some(ProcessBuilder.Redirect.INHERIT),
      redirectError = Some(ProcessBuilder.Redirect.PIPE)
    )

    /*
     * Parse PAR errors from stderr
     *  Very hackish but there's no other way to get information why the process was aborted
     */
    val failedBecauseOfWorkerConnectionLostRegexp = Pattern.compile(
      "^FAIL.*(got unexpected network error, no retries rest|reply isn't OK)$"
    )

    var failedBecauseOfWorkerConnectionLost = false

    val errorStreamReader = new BufferedReader(new InputStreamReader(masterAppProcess.getErrorStream()))
    try {
      breakable {
        while (true) {
          val line = errorStreamReader.readLine
          if (line == null) {
            break
          }
          System.err.println("[CatBoost Master] " + line)

          if (failedBecauseOfWorkerConnectionLostRegexp.matcher(line).matches) {
            failedBecauseOfWorkerConnectionLost = true
          }
        }
      }
    } finally {
      errorStreamReader.close
    }

    val returnValue = masterAppProcess.waitFor
    if (returnValue != 0) {
      if (failedBecauseOfWorkerConnectionLost) {
        throw new CatBoostWorkersConnectionLostException("")
      }
      throw new CatBoostError(s"CatBoost Master process failed: exited with code $returnValue")
    }

    log.info("CatBoost Master process finished successfully.")

    log.info("Trained model: start loading")
    nativeModelResult = native_impl.native_impl.ReadModel(resultModelFilePath.toString)
    log.info("Trained model: finish loading")

    FileUtils.deleteDirectory(tmpDirPath.toFile)
  }
}
