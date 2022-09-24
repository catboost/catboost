package ai.catboost.spark;

import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}

import org.apache.hadoop.fs.{Path => HadoopFsPath}
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamMap,ParamPair}
import org.apache.spark.sql.{DataFrameReader,SparkSession}

import ai.catboost.spark.impl.SerializationHelpers

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._


class PoolReader (
  val spark : SparkSession,
  private val dataFrameReader : DataFrameReader
) {
  case class Metadata(
    partitionedByGroups: Boolean,
    uid: String,
    timestamp: Long,
    sparkVersion: String
  )

  // based in DefaultParamsReader.loadMetadata
  private def loadMetadata(path: String, sc: SparkContext): Metadata = {
    val metadataPath = new HadoopFsPath(path, "metadata").toString
    val metadataStr = sc.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr)
  }

  // based in DefaultParamsReader.parseMetadata
  def parseMetadata(metadataStr: String): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val uid = (metadata \ "uid").extract[String]
    val partitionedByGroups = (metadata \ "partitionedByGroups").extract[Boolean]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]

    Metadata(partitionedByGroups, uid, timestamp, sparkVersion)
  }

  def this(spark: SparkSession) = this(spark, spark.read)

  def dataFramesReaderFormat(source: String): PoolReader = {
    this.dataFrameReader.format(source)
    this
  }

  def dataFramesReaderOption(name: String, value: Object): PoolReader = {
    this.dataFrameReader.option(name, value.toString)
    this
  }

  def dataFramesReaderOptions(options: scala.collection.Map[String, String]): PoolReader = {
    this.dataFrameReader.options(options)
    this
  }

  def dataFramesReaderOptions(options: java.util.Map[String, String]): PoolReader = {
    this.dataFrameReader.options(options)
    this
  }

  def load(path: String) : Pool = {
    val sc = spark.sparkContext
    val fsPath = new HadoopFsPath(path)
    val fileSystem = fsPath.getFileSystem(sc.hadoopConfiguration)

    val metadata = loadMetadata(path, sc)

    val data = dataFrameReader.load(new HadoopFsPath(path, "data").toString)

    val pairsDataPath = new HadoopFsPath(path, "pairsData")
    val pairsData = if (fileSystem.exists(pairsDataPath)) { dataFrameReader.load(pairsDataPath.toString) } else { null }

    val quantizedFeaturesInfo = SerializationHelpers.readObject[QuantizedFeaturesInfoPtr](
      fileSystem,
      new HadoopFsPath(path, "quantizedFeaturesInfo"),
      true
    )
    val paramMap = SerializationHelpers.readObject[ParamMap](
      fileSystem,
      new HadoopFsPath(path, "paramMap"),
      false
    )

    val result = new Pool(Some(metadata.uid), data, pairsData, quantizedFeaturesInfo, metadata.partitionedByGroups)
    paramMap.toSeq.foreach{
      case ParamPair(param, value) => result.set(param, value)
    }
    result
  }
}
