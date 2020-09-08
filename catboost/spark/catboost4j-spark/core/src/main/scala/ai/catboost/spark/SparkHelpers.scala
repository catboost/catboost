package ai.catboost.spark;

import org.apache.spark.sql.Dataset

object SparkHelpers {
  def getThreadCountForTask(dataset : Dataset[_]) : Int = {
    val spark = dataset.sparkSession
    val taskCpusConfig = spark.sparkContext.getConf.getOption("spark.task.cpus")
    taskCpusConfig.getOrElse("1").toInt
  }
}