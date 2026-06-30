package ai.catboost.spark;

import util.control.Breaks._

import org.apache.spark.{SparkContext,SparkJobInfo,SparkStageInfo}
import org.apache.spark.sql.SparkSession

import ai.catboost.CatBoostError


private[spark] object SparkHelpers {
  def parseMemoryOverHeadOption(memoryOverhead: String) : Long = {
    val withUnitRE = "^([\\d]+)(k|m|g|t|K|M|G|T)$".r
    val withoutUnitRE = "^([\\d]+)$".r
    memoryOverhead match {
      case withUnitRE(value, unit) => {
        value.toLong << (
          unit match {
            case "k" | "K" => 10
            case "m" | "M" => 20
            case "g" | "G" => 30
            case "t" | "T" => 40
            case _ => throw new java.lang.RuntimeException("Internal error: Incorrect regex matching")
          }
        )
      }
      case withoutUnitRE(value) => { value.toLong << 20 } // default unit is Megabytes
      case _ => throw new CatBoostError(s"bad format for memory overhead string: $memoryOverhead")
    }
  }

  def getThreadCountForDriver(spark : SparkSession) : Int = {
    val taskCpusConfig = spark.sparkContext.getConf.getOption("spark.driver.cores")
    taskCpusConfig.getOrElse("1").toInt
  }

  def getThreadCountForTask(spark : SparkSession) : Int = {
    val taskCpusConfig = spark.sparkContext.getConf.getOption("spark.task.cpus")
    taskCpusConfig.getOrElse("1").toInt
  }

  // Technically Spark can run several executors on one worker but this is not a recommended case
  def getWorkerCount(spark: SparkSession) : Int = {
    if (spark.sparkContext.isLocal) {
      1
    } else {
      /* There's a period at the start of the Spark application when executors have not been started yet
       *  http://apache-spark-user-list.1001560.n3.nabble.com/Getting-the-number-of-slaves-tp10604p10816.html
       *  retry with sleep until the number stabilizes
       */
      val NumRetriesForExecutorsStartWait = 60 // wait for 1 second each time

      var currentExecutorCount = 0
      var retryCount = 0

      breakable {
        while (true) {
          if (retryCount == NumRetriesForExecutorsStartWait) {
            throw new java.lang.RuntimeException(
              s"Unable to get the number of Spark executors in ${NumRetriesForExecutorsStartWait} seconds"
            )
          }
          retryCount = 0

          val executorCount = spark.sparkContext.statusTracker.getExecutorInfos.length
          if (executorCount > 1) {
            if (executorCount == currentExecutorCount) {
              // heuristic: number is stable for 1 second, so it should be ok
              break
            } else {
              currentExecutorCount = executorCount
            }
          }
          Thread.sleep(1000) // 1 second
          retryCount = retryCount + 1
        }
      }
      currentExecutorCount - 1 // one is the driver
    }
  }

  def getDriverHost(spark: SparkSession): String = {
    spark.sparkContext.getConf.getOption("spark.driver.host").get
  }

  def getDriverNativeMemoryLimit(spark: SparkSession): Option[Long]  = {
    val optionalValue = spark.sparkContext.getConf.getOption("spark.driver.memoryOverhead")
    if (optionalValue.isDefined) { Some(parseMemoryOverHeadOption(optionalValue.get)) } else { None }
  }

  def getExecutorNativeMemoryLimit(spark: SparkSession): Option[Long] = {
    val optionalValue = spark.sparkContext.getConf.getOption("spark.executor.memoryOverhead")
    if (optionalValue.isDefined) { Some(parseMemoryOverHeadOption(optionalValue.get)) } else { None }
  }
}

private[spark] class SparkJobGroupLastStagesFailedStats(
  val failedAttempts : Int,
  val failedTasksInLastAttempt: Int
)

private[spark] object SparkJobGroupLastStagesFailedStats {
  def apply(sparkContext: SparkContext, jobGroup: String) : SparkJobGroupLastStagesFailedStats = {
    val statusTracker = sparkContext.statusTracker

    var failedAttempts = 0
    var failedTasksInLastAttempt = 0

    for (jobId <- statusTracker.getJobIdsForGroup(jobGroup)) {
      statusTracker.getJobInfo(jobId).foreach{
        case jobInfo : SparkJobInfo => {
          if (jobInfo.stageIds.nonEmpty) {
            statusTracker.getStageInfo(jobInfo.stageIds.max).foreach{
              case stageInfo : SparkStageInfo => {
                failedAttempts += stageInfo.currentAttemptId
                failedTasksInLastAttempt += stageInfo.numFailedTasks
              }
            }
          }
        }
      }
    }
    new SparkJobGroupLastStagesFailedStats(failedAttempts, failedTasksInLastAttempt)
  }
}
