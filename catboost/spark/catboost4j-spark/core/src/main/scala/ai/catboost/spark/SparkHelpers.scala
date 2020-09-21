package ai.catboost.spark;

import org.apache.spark.sql.SparkSession

import ai.catboost.CatBoostError


object SparkHelpers {
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
  
  def getWorkerCount(spark: SparkSession) : Int = {
    spark.sparkContext.statusTracker.getExecutorInfos.length - 1 // one is the driver
  }
  
  def getDriverHost(spark: SparkSession) : String = {
    spark.sparkContext.getConf.getOption("spark.driver.host").get
  }
  
  def getDriverNativeMemoryLimit(spark: SparkSession) : Long = {
    parseMemoryOverHeadOption(spark.sparkContext.getConf.getOption("spark.driver.memoryOverhead").get)
  }
  
  def getExecutorNativeMemoryLimit(spark: SparkSession) : Long = {
    parseMemoryOverHeadOption(spark.sparkContext.getConf.getOption("spark.executor.memoryOverhead").get)
  }
}
