package ai.catboost.spark;

import org.apache.spark.sql.SparkSession

object SparkHelpers {
  def getThreadCountForTask(spark : SparkSession) : Int = {
    val taskCpusConfig = spark.sparkContext.getConf.getOption("spark.task.cpus")
    taskCpusConfig.getOrElse("1").toInt
  }
}
