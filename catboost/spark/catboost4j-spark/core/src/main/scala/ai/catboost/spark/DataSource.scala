package ai.catboost.spark;

import java.io.IOException
import java.io.{IOException,ObjectInputStream, ObjectOutputStream}

import collection.mutable
import collection.JavaConverters._
import scala.math.max
import scala.util.control.NonFatal

import org.json4s._
import org.json4s.jackson.JsonMethods

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.mapreduce.Job

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.{SQLDataTypes,Vectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder,RowEncoder}
import org.apache.spark.sql.execution.datasources._
import org.apache.spark.sql.execution.datasources.text.TextFileFormat
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.util.TaskCompletionListener

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._
import ai.catboost.CatBoostError
import ai.catboost.spark.impl.{ExpressionEncoderSerializer,RowEncoderConstructor}


// copied from org.apache.spark.util because it's private there
private object Utils extends Logging {
  def tryOrIOException[T](block: => T): T = {
    try {
      block
    } catch {
      case e: IOException =>
        logError("Exception encountered", e)
        throw e
      case NonFatal(e) =>
        logError("Exception encountered", e)
        throw new IOException(e)
    }
  }
}


// copied from org.apache.spark.util because it's private there
/**
 * Hadoop configuration but serializable. Use `value` to access the Hadoop configuration.
 *
 * @param value Hadoop configuration
 */
private class SerializableConfiguration(@transient var value: Configuration) extends Serializable {
  private def writeObject(out: ObjectOutputStream): Unit = Utils.tryOrIOException {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream): Unit = Utils.tryOrIOException {
    value = new Configuration(false)
    value.readFields(in)
  }
}

private[spark] final class HadoopFileLinesReaderWrapper (
  val lineReader: HadoopFileLinesReader, // Iterator[Text]
  val lineCount: Long,
  val hasHeader: Boolean
) extends IJVMLineDataReader {
  override def GetDataLineCount() : java.math.BigInteger = {
    java.math.BigInteger.valueOf(if (hasHeader) { lineCount - 1 } else { lineCount })
  }

  override def GetHeader() : TMaybe_TString = {
    if (hasHeader) {
      val headerText = lineReader.next
      native_impl.MakeMaybeUtf8String(headerText.getBytes, headerText.getLength)
    } else {
      new TMaybe_TString()
    }
  }

  override def ReadLineJVM(line : TStringOutWrapper) : Boolean = {
    if (lineReader.hasNext) {
      val text = lineReader.next
      line.Assign(text.getBytes, text.getLength)
      true
    } else {
      false
    }
  }
}


private[spark] final class DatasetRowsReaderIterator (
  val rowsReader: TRawDatasetRowsReader,
  var currentBlockSize: Int,
  var currentBlockOffset: Int,
  var currentOutRow: Array[Any],
  val serializer: ExpressionEncoderSerializer,
  val callbacks: mutable.ArrayBuffer[TRawDatasetRow => Unit]
) extends Iterator[InternalRow] {
  private def updateBlock = {
    currentBlockSize = rowsReader.ReadNextBlock
    currentBlockOffset = 0
  }

  def hasNext: Boolean = {
    if (currentBlockOffset < currentBlockSize) {
      true
    } else {
      if (currentBlockSize > 0) {
        updateBlock
      }
      currentBlockSize > 0
    }
  }

  def next(): InternalRow = {
    if (currentBlockOffset >= currentBlockSize) {
      updateBlock
      if (currentBlockSize == 0) {
        throw new java.util.NoSuchElementException("next on empty iterator exception")
      }
    }
    val parsedRaw = rowsReader.GetRow(currentBlockOffset)
    currentBlockOffset = currentBlockOffset + 1
    callbacks.foreach(_(parsedRaw))
    serializer.toInternalRow(Row.fromSeq(currentOutRow))
  }
}

private[spark] object DatasetRowsReaderIterator {
  def apply(
    linesReader: HadoopFileLinesReader,
    dataSchema: StructType,
    intermediateDataMetaInfo: TIntermediateDataMetaInfo,
    options: Map[String, String],
    lineOffset: Long,
    lineCount: Long,
    hasHeader: Boolean,
    threadCount: Int
  ) : DatasetRowsReaderIterator = {

    val maybeColumnsInfo = intermediateDataMetaInfo.getColumnsInfo
    val columnsInfo = if (maybeColumnsInfo.Defined) {
      maybeColumnsInfo.GetRef.getColumns
    } else {
      new TVector_TColumn
    }

    val lineReader = new HadoopFileLinesReaderWrapper(linesReader, lineCount, hasHeader)
    lineReader.swigReleaseOwnership()

    val result = new DatasetRowsReaderIterator(
      rowsReader = new TRawDatasetRowsReader(
        options("dataScheme"),
        lineReader,
        /*columnDescriptionPathWithScheme*/ "",
        columnsInfo,
        options("catboostJsonParams"),
        hasHeader,
        options.getOrElse("blockSize", "10000").toInt,
        threadCount
      ),
      currentBlockSize = 0,
      currentBlockOffset = 0,
      currentOutRow = new Array[Any](dataSchema.length),
      serializer = ExpressionEncoderSerializer(dataSchema),
      callbacks = new mutable.ArrayBuffer[TRawDatasetRow => Unit]
    )

    val baselineCount = intermediateDataMetaInfo.getBaselineCount

    val knownFeatureCount = intermediateDataMetaInfo.GetFeatureCount().toInt

    val denseFeaturesBuffer = new Array[Double](knownFeatureCount)
    val baselineBuffer = new Array[Double](baselineCount.toInt)

    if (intermediateDataMetaInfo.HasSparseFeatures()) {
      result.callbacks += {
        datasetRow => {
          val indices = datasetRow.getSparseFloatFeaturesIndices.toPrimitiveArray
          val values = datasetRow.getSparseFloatFeaturesValues.toPrimitiveArray
          result.currentOutRow(0) = Vectors.sparse(
            max(knownFeatureCount, indices(indices.length - 1) + 1),
            indices,
            values
          )
        }
      }
    } else {
      result.callbacks += {
        datasetRow => {
          datasetRow.GetDenseFloatFeatures(denseFeaturesBuffer)
          result.currentOutRow(0) = Vectors.dense(denseFeaturesBuffer)
        }
      }
    }
    var fieldIdxCounter = 1

    if (intermediateDataMetaInfo.getTargetCount == 1) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        intermediateDataMetaInfo.getTargetType match {
          case ERawTargetType.Boolean => {
              datasetRow => {
                result.currentOutRow(fieldIdx) = (datasetRow.getFloatTarget == 1.0)
              }
            }
          case ERawTargetType.Integer => {
              datasetRow => {
                result.currentOutRow(fieldIdx) = datasetRow.getFloatTarget.toInt
              }
            }
          case ERawTargetType.Float => {
              datasetRow => {
                result.currentOutRow(fieldIdx) = datasetRow.getFloatTarget
              }
            }
          case ERawTargetType.String => {
              datasetRow => {
                result.currentOutRow(fieldIdx) = datasetRow.getStringTarget
              }
            }
          case ERawTargetType.None => throw new CatBoostError("Raw Target column has type None")
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (baselineCount > 0) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        datasetRow => {
          datasetRow.GetBaselines(baselineBuffer)
          result.currentOutRow(fieldIdx) = Vectors.dense(baselineBuffer)
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (intermediateDataMetaInfo.getHasGroupId) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        datasetRow => {
          result.currentOutRow(fieldIdx) = datasetRow.getGroupId
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (intermediateDataMetaInfo.getHasGroupWeight) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        datasetRow => {
          result.currentOutRow(fieldIdx) = datasetRow.getGroupWeight
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (intermediateDataMetaInfo.getHasSubgroupIds) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        datasetRow => {
          result.currentOutRow(fieldIdx) = datasetRow.getSubgroupId
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (intermediateDataMetaInfo.getHasWeights) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        datasetRow => {
          result.currentOutRow(fieldIdx) = datasetRow.getWeight
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (intermediateDataMetaInfo.getHasTimestamp) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      result.callbacks += {
        datasetRow => {
          result.currentOutRow(fieldIdx) = datasetRow.getTimestamp
        }
      }

      fieldIdxCounter = fieldIdxCounter + 1
    }

    if (options.contains("addSampleId")) {
      val fieldIdx = fieldIdxCounter // to capture fixed value

      var sampleId = lineOffset

      result.callbacks += {
        datasetRow => {
          result.currentOutRow(fieldIdx) = sampleId
          sampleId = sampleId + 1
        }
      }
    }

    result.updateBlock

    result
  }

}


private[spark] object CatBoostTextFileFormat {
  def hasHeader(catboostJsonParamsString: String) : Boolean = {
    val catBoostOptionsJson = JsonMethods.parse(catboostJsonParamsString)
    val hasHeaderParamValues = for {
      JBool(has_header) <- catBoostOptionsJson \\ "has_header"
    } yield has_header

    if (hasHeaderParamValues.isEmpty) { false } else { hasHeaderParamValues(0) }
  }

  // CatBoost parsers need at least one data line
  def getHeaderAndFirstLine(
    dataScheme: String,
    sparkSession: SparkSession,
    options: Map[String, String],
    files: Seq[FileStatus]
  ) : (Option[String], String) = {
    val hasHeaderParamValue = hasHeader(options("catboostJsonParams"))

    val lines = sparkSession.baseRelationToDataFrame(
      DataSource.apply(
        sparkSession,
        paths = files.map(_.getPath.toUri.toString),
        className = classOf[TextFileFormat].getName
      ).resolveRelation()
    ) //.select("value")

    import lines.sqlContext.implicits._

    dataScheme match {
      case "dsv" if (hasHeaderParamValue) =>
        val firstLines = lines.head(2)
        (Some(firstLines(0).getAs[String](0)), firstLines(1).getAs[String](0))
      case "dsv" | "libsvm" => (None, lines.head.getAs[String](0))
      case _ => throw new CatBoostError(s"unsupported dataScheme=$dataScheme")
    }
  }


  def makeSchema(intermediateDataMetaInfo: TIntermediateDataMetaInfo, addSampleId: Boolean) : StructType = {
    val fields = new mutable.ArrayBuffer[StructField]()

    fields += StructField(
      "features",
      SQLDataTypes.VectorType,
      false,
      DataHelpers.makeFeaturesMetadata(
        intermediateDataMetaInfo.getFeaturesLayout.GetExternalFeatureIds().toArray[String](new Array[String](0))
      )
    )

    val targetCount = intermediateDataMetaInfo.getTargetCount
    if (targetCount > 1) {
      throw new CatBoostError("Multiple target columns are not supported yet")
    }
    if (targetCount == 1) {
      val dataType = intermediateDataMetaInfo.getTargetType match {
        case ERawTargetType.Boolean => DataTypes.BooleanType
        case ERawTargetType.Integer => DataTypes.IntegerType
        case ERawTargetType.Float => DataTypes.FloatType
        case ERawTargetType.String => DataTypes.StringType
        case ERawTargetType.None => throw new CatBoostError("Raw Target column has type None")
      }
      fields += StructField("label", dataType, false)
    }
    if (intermediateDataMetaInfo.getBaselineCount > 0) {
      fields += StructField("baseline", SQLDataTypes.VectorType, nullable = false)
    }
    if (intermediateDataMetaInfo.getHasGroupId) {
      // always 64 bit
      fields += StructField("groupId", DataTypes.LongType, nullable = false)
    }
    if (intermediateDataMetaInfo.getHasGroupWeight) {
      fields += StructField("groupWeight", DataTypes.FloatType, nullable = false)
    }
    if (intermediateDataMetaInfo.getHasSubgroupIds) {
      fields += StructField("subgroupId", DataTypes.IntegerType, nullable = false)
    }
    if (intermediateDataMetaInfo.getHasWeights) {
      fields += StructField("weight", DataTypes.FloatType, nullable = false)
    }
    if (intermediateDataMetaInfo.getHasTimestamp) {
      fields += StructField("timestamp", DataTypes.LongType, nullable = false)
    }
    if (addSampleId) {
      fields += StructField("sampleId", DataTypes.LongType, nullable = false)
    }

    StructType(fields.toArray)
  }
}


private[spark] class CatBoostTextFileFormat
  extends TextBasedFileFormat
  with DataSourceRegister
  with Logging {

  override def shortName(): String = "catboost"

  override def toString: String = "CatBoost"

  // uuid String -> intermediateDataMetaInfo, to pass between inferSchema and buildReader
  var cachedMetaInfo = new mutable.HashMap[String, TIntermediateDataMetaInfo]


  /*
   * options below will contain the following fields:
   *  "dataScheme" -> CatBoost data scheme. "dsv" and "libsvm" are currently supported
   *  "columnDescription" -> optional. Path to column description file
   *  "catboostJsonParams" -> CatBoost plain params JSON serialized to String
   *  "uuid" -> uuid as String. generated to be able to get data from cachedColumnDescriptions
   *  "blockSize" -> optional. Block size for TRawDatasetRowsReader
   *  "addSampleId" -> optional. Add sampleId column with original file line index
   */


  override def inferSchema(
    sparkSession: SparkSession,
    options: Map[String, String],
    files: Seq[FileStatus]): Option[StructType] = {

    val (optionalHeader, firstDataLine) = CatBoostTextFileFormat.getHeaderAndFirstLine(
      options("dataScheme"),
      sparkSession,
      options,
      files
    )

    val intermediateDataMetaInfo = native_impl.GetIntermediateDataMetaInfo(
      options("dataScheme"),
      options.getOrElse("columnDescription", ""),
      options("catboostJsonParams"),
      if (optionalHeader.isDefined) { new TMaybe_TString(optionalHeader.get) } else { new TMaybe_TString() },
      firstDataLine
    )

    val uuidString = options("uuid")

    this.synchronized {
      cachedMetaInfo.update(uuidString, intermediateDataMetaInfo)
    }

    Some(CatBoostTextFileFormat.makeSchema(intermediateDataMetaInfo, options.contains("addSampleId")))
  }

  override def prepareWrite(
    sparkSession: SparkSession,
    job: Job,
    options: Map[String, String],
    dataSchema: StructType): OutputWriterFactory = {

    throw new CatBoostError("CatBoostTextFileFormat does not support writing")
  }

  override def buildReader(
    sparkSession: SparkSession,
    dataSchema: StructType,
    partitionSchema: StructType,
    requiredSchema: StructType,
    filters: Seq[Filter],
    options: Map[String, String],
    hadoopConf: Configuration): (PartitionedFile) => Iterator[InternalRow] = {

    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))

    val uuidString = options("uuid")
    val intermediateDataMetaInfo = this.synchronized {
      cachedMetaInfo(uuidString)
    }

    val hasHeaderParamValue = CatBoostTextFileFormat.hasHeader(options("catboostJsonParams"))

    val broadcastedDataMetaInfo = sparkSession.sparkContext.broadcast(intermediateDataMetaInfo)

    val threadCountForTask = SparkHelpers.getThreadCountForTask(sparkSession)

    (file: PartitionedFile) => {
      val linesReader = new HadoopFileLinesReader(file, broadcastedHadoopConf.value.value)
      Option(TaskContext.get()).foreach(
        _.addTaskCompletionListener(
          new TaskCompletionListener {
            override def onTaskCompletion(context: TaskContext): Unit = { linesReader.close() }
          }
        )
      )

      DatasetRowsReaderIterator(
        linesReader,
        dataSchema,
        broadcastedDataMetaInfo.value,
        options,
        if (hasHeaderParamValue) { file.start - 1 } else { file.start },
        file.length,
        hasHeaderParamValue && (file.start == 0),
        threadCountForTask
      )
    }
  }
}

private[spark] object CatBoostPairsDataLoader {
  /**
   * @param pairsDataPathWithScheme (optional) Path with scheme to dataset pairs in CatBoost format.
   * @return [[DataFrame]] containing loaded pairs.
   */
  def load(spark: SparkSession, pairsDataPathWithScheme: String) : DataFrame = {
    val pairsPathParts = pairsDataPathWithScheme.split("://", 2)
    val (pairsDataScheme, pairsDataPath) = if (pairsPathParts.size == 1) {
        ("dsv-flat", pairsPathParts(0))
      } else {
        (pairsPathParts(0), pairsPathParts(1))
      }
    if (pairsDataScheme != "dsv-grouped") {
      throw new CatBoostError("Only 'dsv-grouped' scheme is supported for pairs now")
    }
    var schemaWithGroupIdAsStringFields = Seq(
      StructField("groupId", StringType, false),
      StructField("winnerId", LongType, false),
      StructField("loserId", LongType, false)
    )

    import spark.implicits._
    val firstLineArray = spark.read.text(pairsDataPath).limit(1).as[String].collect()
    if (firstLineArray.isEmpty) {
      throw new CatBoostError(s"No data in pairs file ${pairsDataPath}")
    }
    val nFields = firstLineArray(0).split('\t').length
    schemaWithGroupIdAsStringFields = nFields match {
      case 3 => schemaWithGroupIdAsStringFields
      case 4 => schemaWithGroupIdAsStringFields :+ StructField("weight", FloatType, false)
      case nFields => throw new CatBoostError(
        s"Incorrect number of columns (must be 3 or 4) in pairs file ${pairsDataPath}"
      )
    }

    val dfWithStringGroupId = spark.read.schema(StructType(schemaWithGroupIdAsStringFields))
      .option("sep", "\t")
      .csv(pairsDataPath)

    def schema = StructType(
      Seq(StructField("groupId", LongType, false)) ++ schemaWithGroupIdAsStringFields.toSeq.tail
    )

    dfWithStringGroupId.map(
      row => Row.fromSeq(Seq(native_impl.CalcGroupIdForString(row.getString(0))) ++ row.toSeq.tail)
    )(RowEncoderConstructor.construct(schema))
  }
}
