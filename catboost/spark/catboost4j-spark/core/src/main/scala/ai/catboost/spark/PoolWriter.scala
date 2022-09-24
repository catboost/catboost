package ai.catboost.spark;

import java.util.Locale

import collection.JavaConverters._
import collection.mutable

import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}

import org.apache.hadoop.fs.{Path => HadoopFsPath, FileSystem => HadoopFileSystem}

import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame,SaveMode}

import ai.catboost.CatBoostError

import ai.catboost.spark.impl.SerializationHelpers


class PoolWriter (
  val pool : Pool,
  private var dataFramesWriterFormatValue : Option[String] = None,
  private val dataFramesWriterOptionsValue : mutable.HashMap[String, String] = new mutable.HashMap[String, String](),
  private var saveMode : SaveMode = SaveMode.ErrorIfExists
) {
  // based on DefaultParamsWriter.getMetadataToSave
  private def getMetadataToSave(sc: SparkContext): String = {
    val uid = this.pool.uid
    val basicMetadata = {
      ("partitionedByGroups" -> this.pool.partitionedByGroups) ~
      ("uid" -> uid) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version)
    }

    val metadataJson: String = compact(render(basicMetadata))
    metadataJson
  }

  // based on DefaultParamsWriter.saveMetadata
  private def saveMetadata(path: String, sc: SparkContext, fileSystem: HadoopFileSystem) = {
    val metadataFsPath = new HadoopFsPath(path, "metadata")
    if (fileSystem.exists(metadataFsPath)) {
      fileSystem.delete(metadataFsPath, true)
    }
    val metadataJson = getMetadataToSave(sc)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataFsPath.toString)
  }

  def dataFramesWriterFormat(source: String): PoolWriter = {
    if (pool.isQuantized) {
      source match {
        case "csv" | "text" => throw new CatBoostError(
          f"format '$source' is not supported for quantized Pool serialization"
        )
        case "json"  => throw new CatBoostError(
          f"format '$source' is not supported for quantized Pool serialization as 'features' are returned as "
            + "StringType instead of BinaryType"
        )
        case _ => ()
       }
     } else {
       source match {
         case "csv" | "text" => throw new CatBoostError(f"format '$source' is not supported for Pool serialization")
         case "json" | "orc" => throw new CatBoostError(
            f"format '$source' is not supported for Pool serialization as 'features' are returned as "
                + "WrappedArray instead of VectorUDT"
         )
         case _ => ()
      }
    }
    this.dataFramesWriterFormatValue = Some(source)
    this
  }

  def dataFramesWriterOption(name: String, value: Object): PoolWriter = {
    this.dataFramesWriterOptionsValue += (name -> value.toString)
    this
  }

  def dataFramesWriterOptions(options: scala.collection.Map[String, String]): PoolWriter = {
    this.dataFramesWriterOptionsValue ++= options
    this
  }

  def dataFramesWriterOptions(options: java.util.Map[String, String]): PoolWriter = {
    this.dataFramesWriterOptionsValue ++= options.asScala
    this
  }

  def mode(saveModeArg: String) : PoolWriter = {
    this.saveMode = saveModeArg.toLowerCase(Locale.ROOT) match {
      case "ignore" => SaveMode.Ignore
      case "overwrite" => SaveMode.Overwrite
      case "error" | "errorifexists" | "default" => SaveMode.ErrorIfExists
      case _ => throw new IllegalArgumentException(
        s"Unknown save mode: $saveModeArg. " +
        "Accepted save modes are 'ignore', 'overwrite', 'error', 'errorifexists'."
      )
    }
    this
  }

  def mode(saveModeArg: SaveMode) : PoolWriter = {
    this.saveMode = saveModeArg match {
      case SaveMode.Append => throw new IllegalArgumentException (s"Unknown save mode: $saveModeArg. " +
        "Accepted save modes are Ignore, Overwrite, ErrorIfExists.")
      case _ => saveModeArg
    }
    this
  }

  private def saveDataFrame(dataFrame : DataFrame, path : String) = {
    val dataFrameWriter = dataFrame.write.options(this.dataFramesWriterOptionsValue).mode(this.saveMode)
    if (this.dataFramesWriterFormatValue.nonEmpty) {
      dataFrameWriter.format(this.dataFramesWriterFormatValue.get)
    }
    dataFrameWriter.save(path)
  }

  def save(path: String): Unit = {
    val sc = pool.data.sparkSession.sparkContext
    val fsPath = new HadoopFsPath(path)
    val fileSystem = fsPath.getFileSystem(sc.hadoopConfiguration)

    if (fileSystem.exists(fsPath)) {
      if (this.saveMode == SaveMode.Ignore) {
        return ()
      }
      if (this.saveMode == SaveMode.ErrorIfExists) {
        throw new CatBoostError(s"Data at path '$path' already exists")
      }
    } else {
      fileSystem.mkdirs(fsPath)
    }

    saveMetadata(path, sc, fileSystem)

    saveDataFrame(pool.data, new HadoopFsPath(path, "data").toString)
    if (pool.pairsData != null) {
      saveDataFrame(pool.pairsData, new HadoopFsPath(path, "pairsData").toString)
    }

    SerializationHelpers.writeObject(
      fileSystem,
      new HadoopFsPath(path, "quantizedFeaturesInfo"),
      pool.quantizedFeaturesInfo
    )

    SerializationHelpers.writeObject(
      fileSystem,
      new HadoopFsPath(path, "paramMap"),
      pool.extractParamMap()
    )

    ()
  }
}

