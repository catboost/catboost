package ai.catboost.spark

import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util._

import ru.yandex.catboost.spark.catboost4j_spark.core.src._

import ai.catboost.CatBoostError


private[spark] trait CatBoostModelTrait[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]]
  extends org.apache.spark.ml.PredictionModel[Vector, Model]
  with MLWritable
{
  private[spark] var nativeModel: native_impl.TFullModel
  protected var nativeDimension: Int

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  final def predictRawImpl(features: Vector) : Array[Double] = {
    val result = new Array[Double](nativeDimension)
    features match {
      case denseFeatures: DenseVector => nativeModel.Calc(denseFeatures.values, result)
      case sparseFeatures: SparseVector =>
        nativeModel.CalcSparse(sparseFeatures.size, sparseFeatures.indices, sparseFeatures.values, result)
      case _ => throw new CatBoostError("Unknown Vector subtype")
    }
    result
  }

  override def write: MLWriter = new CatBoostModelWriter[Model](this)
}

private[spark] trait CatBoostModelReaderTrait {
  case class Metadata(
    className: String,
    uid: String,
    timestamp: Long,
    sparkVersion: String
  )

  // based in DefaultParamsReader.loadMetadata
  private def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    val metadataPath = new org.apache.hadoop.fs.Path(path, "metadata").toString
    val metadataStr = sc.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr, expectedClassName)
  }

  // based in DefaultParamsReader.parseMetadata
  def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]

    if (expectedClassName.nonEmpty) {
      require(className == expectedClassName, s"Error loading metadata: Expected class name" +
        s" $expectedClassName but found class name $className")
    }

    Metadata(className, uid, timestamp, sparkVersion)
  }

  // returns (uid, nativeModel)
  protected def loadImpl(sc: SparkContext, className: String, path: String): (String, native_impl.TFullModel) = {
    val metadata = loadMetadata(path, sc, className)

    val modelPath = new org.apache.hadoop.fs.Path(path, "model")
    val fileSystem = modelPath.getFileSystem(sc.hadoopConfiguration)
    val contentSummary = fileSystem.getContentSummary(modelPath)

    val nativeModel = new native_impl.TFullModel

    val inputStream = fileSystem.open(modelPath)
    try {
      nativeModel.read(contentSummary.getLength.toInt, inputStream)
    } finally {
      inputStream.close()
    }
    (metadata.uid, nativeModel)
  }
}

private[spark] class CatBoostModelWriter[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]](
  instance: CatBoostModelTrait[Model]
) extends MLWriter {
  private val className = instance.getClass.getName

  // based on DefaultParamsWriter.getMetadataToSave
  private def getMetadataToSave(sc: SparkContext) : String = {
    val uid = instance.uid
    val cls = className
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid)
    val metadataJson: String = compact(render(basicMetadata))
    metadataJson
  }

  // based on DefaultParamsWriter.saveMetadata
  private def saveMetadata(path: String, sc: SparkContext) = {
    val metadataPath = new org.apache.hadoop.fs.Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(sc)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  override protected def saveImpl(path: String): Unit = {
    implicit val sc = super.sparkSession.sparkContext

    saveMetadata(path, sc)

    val modelPath = new org.apache.hadoop.fs.Path(path, "model")
    val outputStream = modelPath.getFileSystem(sc.hadoopConfiguration).create(modelPath)
    try {
      instance.nativeModel.write(outputStream)
    } finally {
      outputStream.close()
    }
  }
}

