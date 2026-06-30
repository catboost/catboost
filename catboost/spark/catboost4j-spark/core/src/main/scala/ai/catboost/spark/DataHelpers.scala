package ai.catboost.spark;

import collection.mutable
import collection.mutable.HashMap
import collection.JavaConverters._

import java.nio.file.{Files,Path}
import java.util.Arrays

import org.slf4j.Logger

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.functions.typedLit
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

import ai.catboost.CatBoostError

import ai.catboost.spark.impl.RowEncoderConstructor

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._


private[spark] class QuantizedFeaturesIndices(
  val ui8Indices : Array[Int],
  val ui16Indices : Array[Int],
  val ui32Indices : Array[Int]
) {
}

private[spark] object QuantizedFeaturesIndices {
  def apply(
    featuresLayout: TFeaturesLayoutPtr,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr
  ) : QuantizedFeaturesIndices = {
    val ui8FeatureIndicesVec = new TVector_i32
    val ui16FeatureIndicesVec = new TVector_i32
    val ui32FeatureIndicesVec = new TVector_i32

    native_impl.GetActiveFeaturesIndices(
      featuresLayout,
      quantizedFeaturesInfo,
      ui8FeatureIndicesVec,
      ui16FeatureIndicesVec,
      ui32FeatureIndicesVec
    )

    new QuantizedFeaturesIndices(
      ui8FeatureIndicesVec.toPrimitiveArray,
      ui16FeatureIndicesVec.toPrimitiveArray,
      ui32FeatureIndicesVec.toPrimitiveArray
    )
  }
}


// Offsets in source quantized features blob
private[spark] class SelectedFeaturesOffsets (
  val ui8Offsets : Array[Int],
  val ui16Offsets : Array[Int],
  val ui32Offsets : Array[Int]
) extends java.io.Serializable {
  def getByteSize: Int = {
    return ui8Offsets.length * 1 + ui16Offsets.length * 2 + ui32Offsets.length * 4
  }
}

private[spark] object SelectedFeaturesOffsets{
  def apply(
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    featuresIndices: QuantizedFeaturesIndices,
    selectedFeaturesFlatIndices: Set[Int]
  ) : SelectedFeaturesOffsets = {
    val offsetsUi8Builder = mutable.ArrayBuilder.make[Int]
    val offsetsUi16Builder = mutable.ArrayBuilder.make[Int]
    val offsetsUi32Builder = mutable.ArrayBuilder.make[Int]

    var offset = 0
    for (i <- featuresIndices.ui8Indices) {
      if (selectedFeaturesFlatIndices.contains(i)) {
        offsetsUi8Builder += offset
      }
      offset = offset + 1
    }
    for (i <- featuresIndices.ui16Indices) {
      if (selectedFeaturesFlatIndices.contains(i)) {
        offsetsUi16Builder += offset
      }
      offset = offset + 2
    }
    for (i <- featuresIndices.ui32Indices) {
      if (selectedFeaturesFlatIndices.contains(i)) {
        offsetsUi32Builder += offset
      }
      offset = offset + 4
    }

    new SelectedFeaturesOffsets(
      offsetsUi8Builder.result,
      offsetsUi16Builder.result,
      offsetsUi32Builder.result
    )
  }
}

private[spark] object FeaturesColumnStorage {
  def apply(
    featuresLayout: TFeaturesLayoutPtr,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr
  ) : FeaturesColumnStorage = {
    val featuresIndices = QuantizedFeaturesIndices(featuresLayout, quantizedFeaturesInfo)

    val buffersUi8 = new Array[TVector_i64](featuresIndices.ui8Indices.length)
    for (i <- 0 until featuresIndices.ui8Indices.length) {
      buffersUi8(i) = new TVector_i64
    }
    val buffersUi16 = new Array[TVector_i64](featuresIndices.ui16Indices.length)
    for (i <- 0 until featuresIndices.ui16Indices.length) {
      buffersUi16(i) = new TVector_i64
    }
    val buffersUi32 = new Array[TVector_i64](featuresIndices.ui32Indices.length)
    for (i <- 0 until featuresIndices.ui32Indices.length) {
      buffersUi32(i) = new TVector_i64
    }

    new FeaturesColumnStorage(
      quantizedFeaturesInfo.GetFeaturesLayout,
      featuresIndices.ui8Indices,
      featuresIndices.ui16Indices,
      featuresIndices.ui32Indices,
      buffersUi8,
      buffersUi16,
      buffersUi32,
      new Array[java.nio.ByteBuffer](featuresIndices.ui8Indices.length),
      new Array[java.nio.ByteBuffer](featuresIndices.ui16Indices.length),
      new Array[java.nio.ByteBuffer](featuresIndices.ui32Indices.length)
    )
  }

  def forEstimated(featuresLayout: TFeaturesLayoutPtr) : FeaturesColumnStorage = {
    val featureCount = featuresLayout.GetExternalFeatureCount.toInt
    val ui8Indices = (0 until featureCount).toArray

    val buffersUi8 = new Array[TVector_i64](featureCount)
    for (i <- 0 until featureCount) {
      buffersUi8(i) = new TVector_i64
    }

    new FeaturesColumnStorage(
      featuresLayout,
      ui8Indices,
      new Array[Int](0),
      new Array[Int](0),
      buffersUi8,
      null,
      null,
      new Array[java.nio.ByteBuffer](featureCount),
      null,
      null
    )
  }
}

/**
 * store quantized feature columns in C++'s TVectors<ui64> to be zero-copy passed to TQuantizedDataProvider
 * expose their memory via JVM's java.nio.ByteBuffer.
 * size grows dynamically by adding rows' features data
 */
private[spark] class FeaturesColumnStorage (
  val featuresLayoutPtr: TFeaturesLayoutPtr,
  val indicesUi8: Array[Int],
  val indicesUi16: Array[Int],
  val indicesUi32: Array[Int],
  val buffersUi8: Array[TVector_i64],
  val buffersUi16: Array[TVector_i64],
  val buffersUi32: Array[TVector_i64],
  var javaBuffersUi8: Array[java.nio.ByteBuffer],
  var javaBuffersUi16: Array[java.nio.ByteBuffer],
  var javaBuffersUi32: Array[java.nio.ByteBuffer],
  var pos: Int = 0,
  var bufferSize: Int = 0
) {
  def addRowFeatures(quantizedValues: Array[Byte]) = {
    if (pos == bufferSize) {
      realloc(if (bufferSize == 0) { 16000 } else { bufferSize * 2 })
    }
    val byteBuffer = java.nio.ByteBuffer.wrap(quantizedValues)
    byteBuffer.order(java.nio.ByteOrder.nativeOrder)
    for (i <- 0 until indicesUi8.length) {
      javaBuffersUi8(i).put(pos, byteBuffer.get)
    }
    for (i <- 0 until indicesUi16.length) {
      javaBuffersUi16(i).putShort(2 * pos, byteBuffer.getShort)
    }
    for (i <- 0 until indicesUi32.length) {
      javaBuffersUi32(i).putInt(4 * pos, byteBuffer.getInt)
    }

    pos = pos + 1
  }

  private def ceilDiv(x: Int, y: Int) : Int = (x + y - 1) /  y

  private def realloc(newSize: Int) = {
    val sizeForUi8 = ceilDiv(newSize, 8)
    val sizeForUi16 = ceilDiv(newSize, 4)
    val sizeForUi32 = ceilDiv(newSize, 2)
    for (i <- 0 until indicesUi8.length) {
      buffersUi8(i).yresize(sizeForUi8)
      javaBuffersUi8(i) = buffersUi8(i).asDirectByteBuffer
      javaBuffersUi8(i).order(java.nio.ByteOrder.nativeOrder)
    }
    for (i <- 0 until indicesUi16.length) {
      buffersUi16(i).yresize(sizeForUi16)
      javaBuffersUi16(i) = buffersUi16(i).asDirectByteBuffer
      javaBuffersUi16(i).order(java.nio.ByteOrder.nativeOrder)
    }
     for (i <- 0 until indicesUi32.length) {
      buffersUi32(i).yresize(sizeForUi32)
      javaBuffersUi32(i) = buffersUi32(i).asDirectByteBuffer
      javaBuffersUi32(i).order(java.nio.ByteOrder.nativeOrder)
    }
    bufferSize = newSize
  }

  def addToVisitor(visitor: IQuantizedFeaturesDataVisitor) = {
    val featuresLayout = featuresLayoutPtr.Get
    for (i <- 0 until indicesUi8.length) {
      visitor.AddFeature(featuresLayout, indicesUi8(i), pos, 8, buffersUi8(i))
    }
    for (i <- 0 until indicesUi16.length) {
      visitor.AddFeature(featuresLayout, indicesUi16(i), pos, 16, buffersUi16(i))
    }
    for (i <- 0 until indicesUi32.length) {
      visitor.AddFeature(featuresLayout, indicesUi32(i), pos, 32, buffersUi32(i))
    }
  }

}


private[spark] class ProcessRowsOutputIterator(
    val dstRows : mutable.ArrayBuffer[Array[Any]],
    val processRowCallback: (Array[Any], Int) => Array[Any], // add necessary data to row
    var objectIdx : Int = 0
) extends Iterator[Row] {
  def next() : Row = {
    val dstRow = processRowCallback(dstRows(objectIdx), objectIdx)
    dstRows(objectIdx) = null // to speed up cleanup
    objectIdx = objectIdx + 1
    return Row.fromSeq(dstRow)
  }

  def hasNext : Boolean = {
    return objectIdx < dstRows.size
  }
}


private[spark] class PoolFilesPaths(
  val mainData : Path,
  val pairsData : Option[Path],
  val estimatedCtrData : Option[Path]
) {}


private[spark] class EstimatedFeaturesLoadingContext(
  var dataProviderBuilderClosure : TDataProviderClosureForJVM = null,
  var visitor : IQuantizedFeaturesDataVisitor = null,
  var dataMetaInfo : TIntermediateDataMetaInfo = null,
  var quantizedFeaturesInfo : QuantizedFeaturesInfoPtr = null
) {
  def start(objectCount: Int) = {
    dataMetaInfo.setObjectCount(java.math.BigInteger.valueOf(objectCount))
    visitor.Start(dataMetaInfo, objectCount, quantizedFeaturesInfo.__deref__)
  }

  def finish() = {
    visitor.Finish
  }

  def getResult() : TDataProviderPtr = {
    dataProviderBuilderClosure.GetResult()
  }
}

private[spark] object EstimatedFeaturesLoadingContext {
  def createAndUpdateCallbacks(
    estimatedFeatureCount: Int,
    estimatedFeaturesColumnIdxInSchema: Int,
    localExecutor: TLocalExecutor,
    mainDataRowCallbacks: mutable.ArrayBuffer[Row => Unit],
    postprocessingCallbacks: mutable.ArrayBuffer[() => Unit]
  ) : EstimatedFeaturesLoadingContext = {

    val (dataProviderBuilder, visitor) = DataHelpers.getDataProviderBuilderAndVisitor(
      /*hasFeatures*/ true,
      localExecutor
    )

    val result = new EstimatedFeaturesLoadingContext(dataProviderBuilder, visitor)

    result.quantizedFeaturesInfo = native_impl.MakeEstimatedQuantizedFeaturesInfo(
      estimatedFeatureCount
    )

    result.dataMetaInfo = new TIntermediateDataMetaInfo
    result.dataMetaInfo.setFeaturesLayout(result.quantizedFeaturesInfo.GetFeaturesLayout)

    val featuresColumnStorage = FeaturesColumnStorage.forEstimated(
      result.quantizedFeaturesInfo.GetFeaturesLayout
    )

    mainDataRowCallbacks += {
      row => {
         featuresColumnStorage.addRowFeatures(row.getAs[Array[Byte]](estimatedFeaturesColumnIdxInSchema))
      }
    }
    postprocessingCallbacks += {
      () => featuresColumnStorage.addToVisitor(result.visitor)
    }

    result
  }
}

private[spark] class DatasetLoadingContext(
  val dataProviderBuilderClosure : TDataProviderClosureForJVM,
  val visitor : IQuantizedFeaturesDataVisitor,
  val dataMetaInfo : TIntermediateDataMetaInfo,
  val quantizedFeaturesInfo : QuantizedFeaturesInfoPtr,
  val mainDataRowCallbacks : mutable.ArrayBuffer[Row => Unit],
  val mainDataPostprocessingCallbacks : mutable.ArrayBuffer[() => Unit],
  val pairsDataRowCallback : (Int, HashMap[Long,Int], Row) => Unit,
  val pairsDataPostprocessingCallback : () => Unit,
  val dstRows : mutable.ArrayBuffer[Array[Any]],
  val estimatedFeaturesLoadingContext : EstimatedFeaturesLoadingContext
) {
  // call after processing all rows
  def postprocessAndGetResults(objectCount: Int) : (TDataProviderPtr, TDataProviderPtr, mutable.ArrayBuffer[Array[Any]]) = {
    dataMetaInfo.setObjectCount(java.math.BigInteger.valueOf(objectCount))

    visitor.Start(dataMetaInfo, objectCount, quantizedFeaturesInfo.__deref__)

    if (estimatedFeaturesLoadingContext != null) {
      estimatedFeaturesLoadingContext.start(objectCount)
    }

    mainDataPostprocessingCallbacks.foreach(_())
    if (pairsDataPostprocessingCallback != null) {
      pairsDataPostprocessingCallback()
    }

    visitor.Finish

    if (estimatedFeaturesLoadingContext != null) {
      estimatedFeaturesLoadingContext.finish()
      (dataProviderBuilderClosure.GetResult(), estimatedFeaturesLoadingContext.getResult(), dstRows)
    } else {
      (dataProviderBuilderClosure.GetResult(), null, dstRows)
    }
  }
}

private[spark] object DatasetLoadingContext {
  def apply(
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    columnIndexMap: HashMap[String, Int], // column type -> idx in schema
    dataMetaInfo: TIntermediateDataMetaInfo,
    mainDatasetSchema: StructType,
    pairsDatasetSchema: StructType, // can be null
    estimatedFeatureCount: Option[Int],
    localExecutor: TLocalExecutor,
    dstRowsColumnIndices: Array[Int] = null,
    dstRowLength: Int = 0
  ) : DatasetLoadingContext = {
    val ownedDataMetaInfo = dataMetaInfo.Clone()

    val (dataProviderBuilderClosure, visitor) = DataHelpers.getDataProviderBuilderAndVisitor(
      columnIndexMap.contains("features"),
      localExecutor
    );

    val (mainDataRowCallbacks, mainDataPostprocessingCallbacks) = getMainDataProcessingCallbacks(
      quantizedFeaturesInfo,
      columnIndexMap,
      ownedDataMetaInfo,
      visitor,
      mainDatasetSchema
    )

    val (pairsDataRowCallback, pairsDataPostprocessingCallback) = if (pairsDatasetSchema != null) {
      getPairsDataProcessingCallbacks(
        visitor,
        pairsDatasetSchema
      )
    } else {
      (null, null)
    }

    val dstRows = addDstRowsCallback(mainDataRowCallbacks, dstRowsColumnIndices, dstRowLength)

    val estimatedFeaturesLoadingContext =
      if (estimatedFeatureCount.isDefined) {
        EstimatedFeaturesLoadingContext.createAndUpdateCallbacks(
          estimatedFeatureCount.get,
          columnIndexMap("_estimatedFeatures"),
          localExecutor,
          mainDataRowCallbacks,
          mainDataPostprocessingCallbacks
        )
      } else {
        null
      }

    new DatasetLoadingContext(
      dataProviderBuilderClosure,
      visitor,
      ownedDataMetaInfo,
      quantizedFeaturesInfo,
      mainDataRowCallbacks,
      mainDataPostprocessingCallbacks,
      pairsDataRowCallback,
      pairsDataPostprocessingCallback,
      dstRows,
      estimatedFeaturesLoadingContext
    )
  }

  private def getLabelCallback(
    stringLabelData: TVector_TString,
    floatLabelData: mutable.ArrayBuilder.ofFloat,
    fieldIdx: Int,
    schema: StructType
  ) : (Row => Unit) = {
    schema(fieldIdx).dataType match {
      case DataTypes.IntegerType => {
        row => {
           floatLabelData += row.getAs[Int](fieldIdx).toFloat
        }
      }
      case DataTypes.LongType => {
        row => {
           floatLabelData += row.getAs[Long](fieldIdx).toFloat
        }
      }
      case DataTypes.FloatType => {
        row => {
           floatLabelData += row.getAs[Float](fieldIdx)
        }
      }
      case DataTypes.DoubleType => {
        row => {
           floatLabelData += row.getAs[Double](fieldIdx).toFloat
        }
      }
      case DataTypes.StringType => {
        row => {
           stringLabelData.add(row.getAs[String](fieldIdx))
        }
      }
      case DataTypes.BooleanType => {
        row => {
           floatLabelData += (if (row.getAs[Boolean](fieldIdx)) 1.0f else 0.0f)
        }
      }
      case _ => throw new CatBoostError("Unsupported data type for Label")
    }
  }

  private def getFloatCallback(
    floatData: mutable.ArrayBuilder.ofFloat,
    fieldIdx: Int,
    schema: StructType
  ) : (Row => Unit) = {
    schema(fieldIdx).dataType match {
      case DataTypes.FloatType => {
        row => {
           floatData += row.getAs[Float](fieldIdx)
        }
      }
      case DataTypes.DoubleType => {
        row => {
           floatData += row.getAs[Double](fieldIdx).toFloat
        }
      }
      case _ => throw new CatBoostError("Unsupported data type for float column")
    }
  }

  /**
   * @returns (row callbacks, postprocessing callbacks)
   */
  private def getMainDataProcessingCallbacks(
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    columnIndexMap: HashMap[String, Int], // column type -> idx in schema
    dataMetaInfo: TIntermediateDataMetaInfo,
    visitor: IQuantizedFeaturesDataVisitor,
    schema: StructType
  ) : (mutable.ArrayBuffer[Row => Unit], mutable.ArrayBuffer[() => Unit]) = {

    val callbacks = new mutable.ArrayBuffer[Row => Unit]
    val postprocessingCallbacks = new mutable.ArrayBuffer[() => Unit]

    if (columnIndexMap.contains("features")) {
      val fieldIdx = columnIndexMap("features")

      val featuresColumnStorage = FeaturesColumnStorage(dataMetaInfo.getFeaturesLayout, quantizedFeaturesInfo)

      callbacks += {
        row => {
           featuresColumnStorage.addRowFeatures(row.getAs[Array[Byte]](fieldIdx))
        }
      }
      postprocessingCallbacks += {
        () => featuresColumnStorage.addToVisitor(visitor)
      }
    }


    if (columnIndexMap.contains("label")) {
      val fieldIdx = columnIndexMap("label")

      val stringLabelData = new TVector_TString
      val floatLabelData = new mutable.ArrayBuilder.ofFloat

      callbacks += getLabelCallback(
        stringLabelData,
        floatLabelData,
        fieldIdx,
        schema
      )
      postprocessingCallbacks += {
        () => dataMetaInfo.getTargetType match {
          case ERawTargetType.Float | ERawTargetType.Integer | ERawTargetType.Boolean =>
            visitor.AddTarget(floatLabelData.result)
          case ERawTargetType.String => visitor.AddTarget(stringLabelData)
          case _ =>
        }
      }
    }

    if (columnIndexMap.contains("weight")) {
      val fieldIdx = columnIndexMap("weight")
      val weightData = new mutable.ArrayBuilder.ofFloat
      callbacks += getFloatCallback(weightData, fieldIdx, schema)
      postprocessingCallbacks += {
        () => visitor.AddWeight(weightData.result)
      }
    }

    if (columnIndexMap.contains("groupWeight")) {
      val fieldIdx = columnIndexMap("groupWeight")
      val groupWeightData = new mutable.ArrayBuilder.ofFloat
      callbacks += getFloatCallback(groupWeightData, fieldIdx, schema)
      postprocessingCallbacks += {
        () => visitor.AddGroupWeight(groupWeightData.result)
      }
    }


    if (columnIndexMap.contains("baseline")) {
      val fieldIdx = columnIndexMap("baseline")
      val baselineCount = dataMetaInfo.getBaselineCount.toInt
      val baselineData = new Array[mutable.ArrayBuilder.ofFloat](baselineCount)
      callbacks += {
        row => {
           val baselineRow = row.getAs[Vector](fieldIdx).toDense
           for (i <- 0 until baselineCount) {
             baselineData(i) += baselineRow(i).toFloat
           }
        }
      }
      postprocessingCallbacks += {
        () => {
          for (i <- 0 until baselineCount) {
            visitor.AddBaseline(i, baselineData(i).result)
          }
        }
      }
    }

    if (columnIndexMap.contains("groupId")) {
      val fieldIdx = columnIndexMap("groupId")
      val groupIdData = new mutable.ArrayBuilder.ofLong
      callbacks += {
        row => {
          groupIdData += row.getAs[Long](fieldIdx)
        }
      }
      postprocessingCallbacks += {
        () => visitor.AddGroupId(groupIdData.result)
      }
    }

    if (columnIndexMap.contains("subgroupId")) {
      val fieldIdx = columnIndexMap("subgroupId")
      val subgroupIdData = new mutable.ArrayBuilder.ofInt
      callbacks += {
        row => {
          subgroupIdData += row.getAs[Int](fieldIdx)
        }
      }
      postprocessingCallbacks += {
        () => visitor.AddSubgroupId(subgroupIdData.result)
      }
    }

    if (columnIndexMap.contains("timestamp")) {
      val fieldIdx = columnIndexMap("timestamp")
      val timestampData = new mutable.ArrayBuilder.ofLong
      callbacks += {
        row => {
          timestampData += row.getAs[Long](fieldIdx)
        }
      }
      postprocessingCallbacks += {
        () => visitor.AddTimestamp(timestampData.result)
      }
    }

    (callbacks, postprocessingCallbacks)
  }

  /**
   * @return (row callback, postprocessing callback)
   * 	row callback has (groupIdx: Int, sampleIdToIdxInGroup: HashMap[Long,Int], row: Row) arguments.
   */
  private def getPairsDataProcessingCallbacks(
    visitor: IQuantizedFeaturesDataVisitor,
    schema: StructType
  ) : ((Int, HashMap[Long,Int], Row) => Unit, () => Unit) = {
    val pairsDataBuilder = new TPairsDataBuilder

    val winnerIdIdx = schema.fieldIndex("winnerId")
    val loserIdIdx = schema.fieldIndex("loserId")
    var maybeWeightIdx : Option[Int] = None

    for ((structField, idx) <- schema.zipWithIndex) {
      structField.name match {
        case "weight" => { maybeWeightIdx = Some(idx) }
        case _ => {}
      }
    }

    val rowCallback = maybeWeightIdx match {
      case Some(weightIdx) => {
        (groupIdx: Int, sampleIdToIdxInGroup: HashMap[Long,Int], row: Row) => {
          pairsDataBuilder.Add(
            groupIdx,
            sampleIdToIdxInGroup(row.getAs[Long](winnerIdIdx)),
            sampleIdToIdxInGroup(row.getAs[Long](loserIdIdx)),
            row.getAs[Float](weightIdx)
          )
        }
      }
      case None => {
        (groupIdx: Int, sampleIdToIdxInGroup: HashMap[Long,Int], row: Row) => {
          pairsDataBuilder.Add(
            groupIdx,
            sampleIdToIdxInGroup(row.getAs[Long](winnerIdIdx)),
            sampleIdToIdxInGroup(row.getAs[Long](loserIdIdx))
          )
        }
      }
    }

    (rowCallback, () => { pairsDataBuilder.AddToResult(visitor) })
  }


  /**
   * return src rows with selected dstRowsColumnIndices, null if dstRowsColumnIndices is null
   */
  private def addDstRowsCallback(
    mainDataProcessingCallbacks : mutable.ArrayBuffer[Row => Unit],
    dstRowsColumnIndices: Array[Int], // can be null
    dstRowLength: Int
  ) : mutable.ArrayBuffer[Array[Any]]  = {
    if (dstRowLength > 0) {
      val dstRows = new mutable.ArrayBuffer[Array[Any]]
      if (dstRowsColumnIndices != null) {
        mainDataProcessingCallbacks += {
          row => {
             val rowFields = new Array[Any](dstRowLength)
             for (i <- 0 until dstRowsColumnIndices.size) {
               rowFields(i) = row(dstRowsColumnIndices(i))
             }
             dstRows += rowFields
          }
        }
      } else {
        mainDataProcessingCallbacks += {
          row => { dstRows += new Array[Any](dstRowLength) }
        }
      }
      dstRows
    } else {
      null
    }
  }
}


private[spark] abstract class DatasetForTraining(
  val srcPool : Pool,
  val mainDataSchema : StructType,
  val datasetIdx : Byte
)

private[spark] case class UsualDatasetForTraining(
  override val srcPool : Pool,
  val data : DataFrame,
  override val datasetIdx : Byte
) extends DatasetForTraining(srcPool, data.schema, datasetIdx)

private[spark] case class DatasetForTrainingWithPairs(
  override val srcPool : Pool,
  val data : RDD[DataHelpers.PreparedGroupData],
  override val mainDataSchema : StructType,
  override val datasetIdx : Byte
) extends DatasetForTraining(srcPool, mainDataSchema, datasetIdx)


private[spark] object DataHelpers {
  def selectSchemaFields(srcSchema: StructType, fieldNames: Array[String] = null) : Seq[StructField] = {
    if (fieldNames == null) {
      srcSchema.toSeq
    } else {
      srcSchema.filter(field => fieldNames.contains(field.name))
    }
  }

  def mapSampleIdxToPerGroupSampleIdx(data: DataFrame) : DataFrame = {
    val groupIdIdx = data.schema.fieldIndex("groupId")
    val sampleIdIdx = data.schema.fieldIndex("sampleId")

    // Cannot use DataFrame API directly with RowEncoder because it loses schema columns metadata
    val resultAsRDD = data.rdd.groupBy(row => row.getLong(groupIdIdx)).flatMap{
      case (groupId, rows) => {
        var startSampleId : Long = Long.MaxValue
        val rowsCopy = rows.map(
          row => {
            startSampleId = startSampleId.min(row.getLong(sampleIdIdx))
            row
          }
        ).toSeq
        rowsCopy.map(
          row => {
            var fields = row.toSeq.toArray
            fields(sampleIdIdx) = fields(sampleIdIdx).asInstanceOf[Long] - startSampleId
            Row.fromSeq(fields)
          }
        )
      }
    }

    data.sparkSession.createDataFrame(resultAsRDD, data.schema)
  }


  // first element is (datasetIdx, groupId) pair
  // second element is (Iterable of main dataset data, Iterable of pairs data)
  type PreparedGroupData = ((Byte, Long), (Iterable[Iterable[Row]], Iterable[Iterable[Row]]))
  type GroupsIterator = Iterator[PreparedGroupData]

  def makeFeaturesMetadata(initialFeatureNames: Array[String]) : Metadata = {
    val featureNames = new Array[String](initialFeatureNames.length)

    val featureNamesSet = new mutable.HashSet[String]()

    for (i <- 0 until featureNames.size) {
      val name = initialFeatureNames(i)
      if (name.isEmpty) {
        val generatedName = i.toString
        if (featureNamesSet.contains(generatedName)) {
          throw new CatBoostError(
            s"""Unable to use generated name "$generatedName" for feature with unspecified name because"""
            + " it has been already used for another feature"
          )
        }
        featureNames(i) = generatedName
      } else {
        featureNames(i) = name
      }
      featureNamesSet.add(featureNames(i))
    }

    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = featureNames.map {
      name => defaultAttr.withName(name).asInstanceOf[Attribute]
    }.toArray
    val attrGroup = new AttributeGroup("userFeatures", attrs)
    attrGroup.toMetadata
  }

  def getDistinctIntLabelValues(data: DataFrame, labelColumn: String) : Array[Int] = {
    val iterator = data.select(labelColumn).distinct.toLocalIterator.asScala
    data.schema(labelColumn).dataType match {
      case DataTypes.IntegerType => iterator.map{ _.getAs[Int](0) }.toSeq.sorted.toArray
      case DataTypes.LongType => iterator.map{ _.getAs[Long](0) }.toSeq.sorted.map{ _.toInt }.toArray
      case _ => throw new CatBoostError("Unsupported data type for Integer Label")
    }
  }

  def getDistinctFloatLabelValues(data: DataFrame, labelColumn: String) : Array[Float] = {
    val iterator = data.select(labelColumn).distinct.toLocalIterator.asScala
    data.schema(labelColumn).dataType match {
      case DataTypes.FloatType => iterator.map{ _.getAs[Float](0) }.toSeq.sorted.toArray
      case DataTypes.DoubleType => iterator.map{ _.getAs[Double](0) }.toSeq.sorted.map{ _.toFloat }.toArray
      case _ => throw new CatBoostError("Unsupported data type for Float Label")
    }
  }

  def getDistinctStringLabelValues(data: DataFrame, labelColumn: String) : TVector_TString = {
    val iterator = data.select(labelColumn).distinct.toLocalIterator.asScala
    data.schema(labelColumn).dataType match {
      case DataTypes.StringType => new TVector_TString(
        iterator.map{ _.getString(0) }.toSeq.sorted.toIterable.asJava
      )
      case _ => throw new CatBoostError("Unsupported data type for String Label")
    }
  }

  // returns Array[Byte] because it is easier to pass to native code
  def calcFeaturesHasNans(data: DataFrame, featuresColumn: String, featureCount: Int) : Array[Byte] = {
    val featuresColIdx = data.schema.fieldIndex(featuresColumn)

    import data.sparkSession.implicits._
    val partialResultDf = data.mapPartitions(
      rows => {
        var result = new Array[Byte](featureCount)
        Arrays.fill(result, 0.toByte)
        for (row <- rows) {
          val featureValues = row.getAs[Vector](featuresColIdx).toArray
          for (i <- 0 until featureCount) {
            if (featureValues(i) != featureValues(i)) { // inequality check is fast IsNan
              result(i) = 1.toByte
            }
          }
        }
        Iterator[Array[Byte]](result)
      }
    ).persist(StorageLevel.MEMORY_ONLY)

    var result = new Array[Byte](featureCount)
    Arrays.fill(result, 0.toByte)

    for (partialResult <- partialResultDf.toLocalIterator.asScala) {
      for (i <- 0 until featureCount) {
        if (partialResult(i) == 1.toByte) {
          result(i) = 1.toByte
        }
      }
    }

    partialResultDf.unpersist()

    result
  }

  /**
   * @return (dstRows, rawObjectDataProvider)
   */
  def processDatasetWithRawFeatures(
    rows: Iterator[Row],
    featuresColumnIdx: Int,
    featuresLayout: TFeaturesLayoutPtr,
    maxUniqCatFeatureValues: Int,
    keepRawFeaturesInDstRows: Boolean,
    dstRowLength: Int,
    localExecutor: TLocalExecutor
  ) : (mutable.ArrayBuffer[Array[Any]], TRawObjectsDataProviderPtr) = {
    val dstRows = new mutable.ArrayBuffer[Array[Any]]

    val availableFloatFeaturesFlatIndices
      = native_impl.GetAvailableFeaturesFlatIndices_Float(featuresLayout.__deref__()).toPrimitiveArray
    val availableCatFeaturesFlatIndices
      = native_impl.GetAvailableFeaturesFlatIndices_Categorical(featuresLayout.__deref__()).toPrimitiveArray

    // features data as columns
    var availableFloatFeaturesData = new Array[mutable.ArrayBuilder[Float]](availableFloatFeaturesFlatIndices.size)
    for (i <- 0 until availableFloatFeaturesData.size) {
      availableFloatFeaturesData(i) = mutable.ArrayBuilder.make[Float]
    }
    var availableCatFeaturesData = new Array[mutable.ArrayBuilder[Int]](availableCatFeaturesFlatIndices.size)
    for (i <- 0 until availableCatFeaturesData.size) {
      availableCatFeaturesData(i) = mutable.ArrayBuilder.make[Int]
    }

    rows.foreach {
      row => {
         val rowFields = new Array[Any](dstRowLength)
         for (i <- 0 until row.length) {
           if (i == featuresColumnIdx) {
             val featuresValues = row.getAs[Vector](i)
             for (j <- 0 until availableFloatFeaturesFlatIndices.size) {
               availableFloatFeaturesData(j) += featuresValues(availableFloatFeaturesFlatIndices(j)).toFloat
             }
             for (j <- 0 until availableCatFeaturesFlatIndices.size) {
               availableCatFeaturesData(j) += featuresValues(availableCatFeaturesFlatIndices(j)).toInt
             }
             if (keepRawFeaturesInDstRows) {
               rowFields(i) = row(i)
             }
           } else {
             rowFields(i) = row(i)
           }
         }
         dstRows += rowFields
      }
    }

    val availableFloatFeaturesDataForBuilder = new TVector_TMaybeOwningConstArrayHolder_float
    for (featureData <- availableFloatFeaturesData) {
      val result = featureData.result
      availableFloatFeaturesDataForBuilder.add(result)
    }
    val availableCatFeaturesDataForBuilder = new TVector_TMaybeOwningConstArrayHolder_i32
    for (featureData <- availableCatFeaturesData) {
      val result = featureData.result
      availableCatFeaturesDataForBuilder.add(result)
    }

    val rawObjectsDataProviderPtr = native_impl.CreateRawObjectsDataProvider(
      featuresLayout,
      dstRows.size.toLong,
      availableFloatFeaturesDataForBuilder,
      availableCatFeaturesDataForBuilder,
      maxUniqCatFeatureValues,
      localExecutor
    )

    // try to force cleanup of no longer used data
    availableFloatFeaturesData = null
    availableCatFeaturesData = null
    System.gc()

    (dstRows, rawObjectsDataProviderPtr)
  }


  // Note: do not repartition the resulting data for master and workers separately
  def prepareDatasetForTraining(pool: Pool, datasetIdx: Byte, workerCount: Int) : DatasetForTraining = {
    if (pool.pairsData != null) {
      val cogroupedData = getCogroupedMainAndPairsRDD(
        pool.data,
        pool.data.schema.fieldIndex(pool.getOrDefault(pool.groupIdCol)),
        pool.pairsData,
        datasetIdx,
        numPartitions=Some(workerCount)
      ).cache()
      DatasetForTrainingWithPairs(pool, cogroupedData, pool.data.schema, datasetIdx)
    } else {
      val repartitionedPool = pool.repartition(workerCount, byGroupColumnsIfPresent=true)
      val data = repartitionedPool.data.withColumn("_datasetIdx", typedLit(datasetIdx)).cache()
      UsualDatasetForTraining(pool, data, datasetIdx)
    }
  }


  def getDataProviderBuilderAndVisitor(
    hasFeatures: Boolean,
    localExecutor: TLocalExecutor
  ) : (TDataProviderClosureForJVM, IQuantizedFeaturesDataVisitor) = {
    val dataProviderBuilderOptions = new TDataProviderBuilderOptions

    val dataProviderClosure = new TDataProviderClosureForJVM(
      EDatasetVisitorType.QuantizedFeatures,
      dataProviderBuilderOptions,
      hasFeatures,
      localExecutor
    )
    val visitor = dataProviderClosure.GetQuantizedVisitor
    if (visitor == null) {
      throw new CatBoostError("Failure to create IQuantizedFeaturesDataVisitor")
    }

    (dataProviderClosure, visitor)
  }

  def getLoadedDatasets(
    datasetLoadingContexts: Seq[DatasetLoadingContext],
    objectCountPerDataset : Array[Int]
  ) : (TVector_TDataProviderPtr, TVector_TDataProviderPtr, Array[mutable.ArrayBuffer[Array[Any]]]) = {
    val dstDataProviders = new TVector_TDataProviderPtr
    val dstEstimatedDataProviders = new TVector_TDataProviderPtr
    val dstDatasetsRows = new Array[mutable.ArrayBuffer[Array[Any]]](datasetLoadingContexts.size)

    datasetLoadingContexts.zipWithIndex.map {
      case (datasetLoadingContext, i) => {
        val (dataProvider, estimatedDataProvider, dstRows)
          = datasetLoadingContext.postprocessAndGetResults(objectCountPerDataset(i))
        dstDataProviders.add(dataProvider)
        if (estimatedDataProvider != null) {
          dstEstimatedDataProviders.add(estimatedDataProvider)
        }
        dstDatasetsRows(i) = dstRows
      }
    }

    (dstDataProviders, dstEstimatedDataProviders, dstDatasetsRows)
  }


  /**
   * Create quantized data providers from iterating over DataFrame's Rows.
   * @returns (quantized data provider, quantized estimated features provider, dstRows).
   *  types of quantized data providers are TDataProviderPtr because that's generic interface that
   *  clients (like training, prediction, feature quality estimators) accept.
   *  Quantized estimated features provider is created if estimatedFeatureCount is defined
   *  dstRows - src rows with selected dstRowsColumnIndices, null if dstRowsColumnIndices is null
   */
  def loadQuantizedDatasets(
    datasetCount: Int,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    columnIndexMap: HashMap[String, Int], // column type -> idx in schema
    dataMetaInfo: TIntermediateDataMetaInfo,
    schema: StructType,
    estimatedFeatureCount: Option[Int],
    localExecutor: TLocalExecutor,
    rows: Iterator[Row],
    dstRowsColumnIndices: Array[Int] = null,
    dstRowLength: Int = 0
  ) : (TVector_TDataProviderPtr, TVector_TDataProviderPtr, Array[mutable.ArrayBuffer[Array[Any]]]) = {
    val datasetLoadingContexts = (0 until datasetCount).map{
      _ => DatasetLoadingContext(
        quantizedFeaturesInfo,
        columnIndexMap,
        dataMetaInfo,
        schema,
        /*pairsDatasetSchema*/ null,
        estimatedFeatureCount,
        localExecutor,
        dstRowsColumnIndices,
        dstRowLength
      )
    }

    var objectCountPerDataset = new Array[Int](datasetCount)
    Arrays.fill(objectCountPerDataset, 0)

    val datasetIdxColumnIdx = columnIndexMap.getOrElse("_datasetIdx", -1)
    rows.foreach {
      row => {
        val datasetIdx = if (datasetIdxColumnIdx == -1) { 0 } else { row.getAs[Byte](datasetIdxColumnIdx).toInt }
        datasetLoadingContexts(datasetIdx).mainDataRowCallbacks.foreach(_(row))
        objectCountPerDataset(datasetIdx) = objectCountPerDataset(datasetIdx) + 1
      }
    }

    getLoadedDatasets(datasetLoadingContexts, objectCountPerDataset)
  }

  /**
   * Create quantized data providers from iterating over cogrouped merged main dataset and pairs data.
   * @returns (quantized data providers, quantized estimated features providers, dstRows).
   *  types of quantized data providers are TDataProviderPtr because that's generic interface that
   *  clients (like training, prediction, feature quality estimators) accept.
   *  Quantized estimated features provider is created if estimatedFeatureCount is defined
   *  dstRows - src rows with selected dstRowsColumnIndices, null if dstRowsColumnIndices is null
   */
  def loadQuantizedDatasetsWithPairs(
    datasetOffset: Int,
    datasetCount: Int,
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    columnIndexMap: HashMap[String, Int], // column type -> idx in schema
    dataMetaInfo: TIntermediateDataMetaInfo,
    datasetSchema: StructType,
    pairsDatasetSchema: StructType,
    estimatedFeatureCount: Option[Int],
    localExecutor: TLocalExecutor,
    groupsIterator: GroupsIterator,
    dstRowsColumnIndices: Array[Int] = null,
    dstRowLength: Int = 0
  ) : (TVector_TDataProviderPtr, TVector_TDataProviderPtr, Array[mutable.ArrayBuffer[Array[Any]]]) = {
    val datasetLoadingContexts = (0 until datasetCount).map{
      _ => DatasetLoadingContext(
        quantizedFeaturesInfo,
        columnIndexMap,
        dataMetaInfo,
        datasetSchema,
        pairsDatasetSchema,
        estimatedFeatureCount,
        localExecutor,
        dstRowsColumnIndices,
        dstRowLength
      )
    }

    var objectCountPerDataset = new Array[Int](datasetCount)
    Arrays.fill(objectCountPerDataset, 0)
    var groupIdxPerDataset = new Array[Int](datasetCount)
    Arrays.fill(groupIdxPerDataset, 0)

    val sampleIdIdx = columnIndexMap("sampleId")

    groupsIterator.foreach(
      (group: PreparedGroupData) => {
        val datasetIdx = group._1._1.toInt - datasetOffset
        val groupIdx = groupIdxPerDataset(datasetIdx)
        val sampleIdToIdxInGroup = new HashMap[Long,Int]
        var objectIdxInGroup = 0
        group._2._1.foreach(
          (it : Iterable[Row]) => {
            it.foreach(
              row => {
                datasetLoadingContexts(datasetIdx).mainDataRowCallbacks.foreach(_(row))

                val sampleId = row.getLong(sampleIdIdx)
                sampleIdToIdxInGroup.put(sampleId, objectIdxInGroup)

                objectIdxInGroup = objectIdxInGroup + 1
              }
            )
          }
        )
        objectCountPerDataset(datasetIdx) = objectCountPerDataset(datasetIdx) + objectIdxInGroup
        group._2._2.foreach(
          (it : Iterable[Row]) => {
            it.foreach(
              row => {
                datasetLoadingContexts(datasetIdx).pairsDataRowCallback(groupIdx, sampleIdToIdxInGroup, row)
              }
            )
          }
        )
        groupIdxPerDataset(datasetIdx) = groupIdx + 1
      }
    )

    getLoadedDatasets(datasetLoadingContexts, objectCountPerDataset)
  }


  /**
   * @returns (
   *  map of column type -> index in dst main data columns,
   *  selected column names,
   *  dst column indices,
   *  estimatedFeatureCount
   * )
   */
  def selectColumnsAndReturnIndex(
    pool: Pool,
    columnTypeNames: Seq[String],
    includeEstimatedFeatures: Boolean,
    includeDatasetIdx: Boolean = false,
    dstColumnNames: Seq[String] = Seq()
  ) : (HashMap[String, Int], Array[String], Array[Int], Option[Int]) = {
    val columnTypesMap = new mutable.HashMap[String, Int]()
    var columnsList = new mutable.ArrayBuffer[String]()
    var i = 0

    val updateColumnWithType = (columnName : String, columnTypeName : String) => {
      columnsList += columnName
      columnTypesMap.update(columnTypeName, i)
      i = i + 1
    }

    for (columnTypeName <- columnTypeNames) {
      val param = pool.getParam(columnTypeName + "Col").asInstanceOf[Param[String]]
      if (pool.isDefined(param)) {
        updateColumnWithType(pool.getOrDefault(param), columnTypeName)
      }
    }
    val estimatedFeatureCount
      = if (includeEstimatedFeatures && pool.data.schema.fieldNames.contains("_estimatedFeatures")) {
          updateColumnWithType("_estimatedFeatures", "_estimatedFeatures")
          Some(pool.getEstimatedFeatureCount)
        } else {
          None
        }
    if (includeDatasetIdx) {
      updateColumnWithType("_datasetIdx", "_datasetIdx")
    }

    val dstColumnIndices = new mutable.ArrayBuffer[Int]()
    for (dstColumnName <- dstColumnNames) {
      val selectedIdx = columnsList.indexOf(dstColumnName)
      if (selectedIdx == -1) {
        columnsList += dstColumnName
        dstColumnIndices += i
        i = i + 1
      } else {
        dstColumnIndices += selectedIdx
      }
    }

    (columnTypesMap, columnsList.toArray, dstColumnIndices.toArray, estimatedFeatureCount)
  }


  /**
   * @returns (
   *  data with columns for training,
   *  map of column type -> index in selected schema,
   *  estimatedFeatureCount
   * )
   */
  def selectColumnsForTrainingAndReturnIndex(
    data: DatasetForTraining,
    includeFeatures: Boolean,
    includeSampleId: Boolean,
    includeEstimatedFeatures: Boolean,
    includeDatasetIdx: Boolean
  ) : (DatasetForTraining, HashMap[String, Int], Option[Int]) = {
    // Pool param name is columnTypeName + "Col"
    val columnTypeNames = mutable.ArrayBuffer[String](
      "label",
      "weight",
      "groupWeight",
      "baseline",
      "groupId",
      "subgroupId",
      "timestamp"
    )
    if (includeFeatures) {
      columnTypeNames += "features"
    }
    if (includeSampleId) {
      columnTypeNames += "sampleId"
    }
    val (columnIndexMap, selectedColumnNames, _, estimatedFeatureCount) = selectColumnsAndReturnIndex(
      data.srcPool,
      columnTypeNames.toSeq,
      includeEstimatedFeatures,
      includeDatasetIdx=includeDatasetIdx && data.isInstanceOf[UsualDatasetForTraining]
    )

    val selectedData = data match {
      case UsualDatasetForTraining(srcPool, dataFrame, datasetIdx) => {
        UsualDatasetForTraining(
          srcPool,
          dataFrame.select(selectedColumnNames.head, selectedColumnNames.tail: _*),
          datasetIdx
        )
      }
      case DatasetForTrainingWithPairs(srcPool, groupData, mainDataSchema, datasetIdx) => {
        val selectedColumnIndices = selectedColumnNames.map{ mainDataSchema.fieldIndex(_) }

        val selectedGroupData = groupData.map {
          case (key, (mainPart, pairsPart)) => {
            (
              key,
              (
                mainPart.map {
                  case mainGroupData => mainGroupData.map {
                    case row => Row.fromSeq(selectedColumnIndices.map{ row(_) }.toSeq)
                  }
                },
                pairsPart
              )
            )
          }
        }

        DatasetForTrainingWithPairs(
          srcPool,
          selectedGroupData,
          StructType(selectedColumnIndices.map{ mainDataSchema(_) }),
          datasetIdx
        )
      }
    }
    (selectedData, columnIndexMap, estimatedFeatureCount)
  }

  def getCogroupedMainAndPairsRDD(
    mainData: DataFrame,
    mainDataGroupIdFieldIdx: Int,
    pairsData: DataFrame,
    datasetIdx : Byte = 0,
    numPartitions : Option[Int] = None
  ) : RDD[PreparedGroupData] = {
      val groupedMainData = mainData.rdd.groupBy(row => (datasetIdx, row.getLong(mainDataGroupIdFieldIdx)))

      val pairsGroupIdx = pairsData.schema.fieldIndex("groupId")
      val groupedPairsData = pairsData.rdd.groupBy(row => (datasetIdx, row.getLong(pairsGroupIdx)))

      numPartitions match {
        case Some(numPartitions) => groupedMainData.cogroup(groupedPairsData, numPartitions)
        case None => groupedMainData.cogroup(groupedPairsData)
      }
  }

  /**
   * @return (path to main quantized features file, optional path to pairs data (in 'dsv-grouped' format), path to estimated quantized features file)
   *  Path to estimated quantized features file will be null if
   *  	includeEstimatedFeatures = false or no actual estimated features data is present is pool.
   */
  def downloadQuantizedPoolToTempFiles(
    data: DatasetForTraining,
    includeFeatures: Boolean,
    includeEstimatedFeatures: Boolean,
    localExecutor: TLocalExecutor,
    dataPartName: String,
    log: Logger,
    tmpFilePrefix: String = null,
    tmpFileSuffix: String = null
  ) : PoolFilesPaths = {
    log.info(s"downloadQuantizedPoolToTempFiles for ${dataPartName}: start")

    val (selectedData, columnIndexMap, estimatedFeatureCount) = selectColumnsForTrainingAndReturnIndex(
      data,
      includeFeatures,
      includeSampleId = data.isInstanceOf[DatasetForTrainingWithPairs],
      includeEstimatedFeatures,
      includeDatasetIdx=false
    )

    val (mainDataProviders, estimatedDataProviders, _) = selectedData match {
      case UsualDatasetForTraining(srcPool, selectedDF, _) => {
        log.info(s"loadQuantizedDatasets for ${dataPartName}: start")

        val result = loadQuantizedDatasets(
          /*datasetCount*/ 1,
          srcPool.quantizedFeaturesInfo,
          columnIndexMap,
          srcPool.createDataMetaInfo(),
          selectedDF.schema,
          estimatedFeatureCount,
          localExecutor,
          selectedDF.toLocalIterator.asScala
        )
        log.info(s"loadQuantizedDatasets for ${dataPartName}: finish")
        result
      }
      case DatasetForTrainingWithPairs(srcPool, selectedGroupData, selectedMainDataSchema, _) => {
        log.info(s"loadQuantizedDatasetsWithPairs for ${dataPartName}: start")

        val result = loadQuantizedDatasetsWithPairs(
          /*datasetOffset*/ data.datasetIdx,
          /*datasetCount*/ 1,
          srcPool.quantizedFeaturesInfo,
          columnIndexMap,
          srcPool.createDataMetaInfo(),
          selectedMainDataSchema,
          srcPool.pairsData.schema,
          estimatedFeatureCount,
          localExecutor,
          selectedGroupData.toLocalIterator
        )
        log.info(s"loadQuantizedDatasetsWithPairs for ${dataPartName}: finish")
        result
      }
    }

    log.info(s"${dataPartName}: save loaded data to files: start")

    val mainDataProvider = mainDataProviders.get(0)
    val tmpMainFilePath = Files.createTempFile(tmpFilePrefix, tmpFileSuffix)
    tmpMainFilePath.toFile.deleteOnExit
    native_impl.SaveQuantizedPool(mainDataProvider, tmpMainFilePath.toString)

    var tmpPairsDataFilePath : Option[Path] = None
    if (data.isInstanceOf[DatasetForTrainingWithPairs]) {
      tmpPairsDataFilePath = Some(Files.createTempFile(tmpFilePrefix, tmpFileSuffix))
      tmpPairsDataFilePath.get.toFile.deleteOnExit
      native_impl.SavePairsInGroupedDsvFormat(mainDataProvider, tmpPairsDataFilePath.get.toString)
    }

    var tmpEstimatedFilePath : Option[Path] = None
    if (estimatedFeatureCount.isDefined) {
      tmpEstimatedFilePath = Some(Files.createTempFile(tmpFilePrefix, tmpFileSuffix))
      tmpEstimatedFilePath.get.toFile.deleteOnExit
      native_impl.SaveQuantizedPool(estimatedDataProviders.get(0), tmpEstimatedFilePath.get.toString)
    }

    log.info(s"${dataPartName}: save loaded data to files: finish")

    log.info(s"downloadQuantizedPoolToTempFiles for ${dataPartName}: finish")

    new PoolFilesPaths(tmpMainFilePath, tmpPairsDataFilePath, tmpEstimatedFilePath)
  }


  def downloadSubsetOfQuantizedFeatures(
    pool: Pool,
    quantizedFeaturesIndices: QuantizedFeaturesIndices,
    selectedFeaturesFlatIndices: Set[Int],
    localExecutor: TLocalExecutor
  ) : TQuantizedObjectsDataProviderPtr = {
    if (!pool.isQuantized) {
      throw new CatBoostError("downloadSubsetOfQuantizedFeatures is applicable only for quantized pools")
    }

    val selectedFeaturesOffsets = SelectedFeaturesOffsets(
      pool.quantizedFeaturesInfo,
      quantizedFeaturesIndices,
      selectedFeaturesFlatIndices)

    val selectedFeaturesByteSize = selectedFeaturesOffsets.getByteSize

    val selectedFeaturesSchema = StructType(Seq(StructField("features", BinaryType, false)))
    val selectedFeaturesEncoder = RowEncoderConstructor.construct(selectedFeaturesSchema)

    val selectedFeaturesDf = pool.data.select(pool.getFeaturesCol).mapPartitions(
      rows => {
        val buffer = new Array[Byte](selectedFeaturesByteSize)
        rows.map(
          row => {
            val srcByteBuffer = java.nio.ByteBuffer.wrap(row.getAs[Array[Byte]](0))
            srcByteBuffer.order(java.nio.ByteOrder.nativeOrder)
            val dstByteBuffer = java.nio.ByteBuffer.wrap(buffer)
            dstByteBuffer.order(java.nio.ByteOrder.nativeOrder)
            for (offset <- selectedFeaturesOffsets.ui8Offsets) {
              dstByteBuffer.put(srcByteBuffer.get(offset))
            }
            for (offset <- selectedFeaturesOffsets.ui16Offsets) {
              dstByteBuffer.putShort(srcByteBuffer.getShort(offset))
            }
            for (offset <- selectedFeaturesOffsets.ui32Offsets) {
              dstByteBuffer.putInt(srcByteBuffer.getInt(offset))
            }
            Row(buffer)
          }
        )
      }
    )(selectedFeaturesEncoder)

    val dataMetaInfo = new TIntermediateDataMetaInfo
    dataMetaInfo.setFeaturesLayout(
      native_impl.CloneWithSelectedFeatures(
        pool.quantizedFeaturesInfo.GetFeaturesLayout.__deref__,
        selectedFeaturesFlatIndices.toArray
      )
    )

    loadQuantizedDatasets(
      /*datasetCount*/ 1,
      pool.quantizedFeaturesInfo,
      HashMap[String,Int]("features" -> 0),
      dataMetaInfo,
      selectedFeaturesSchema,
      /*estimatedFeatureCount*/ None,
      localExecutor,
      selectedFeaturesDf.toLocalIterator.asScala
    )._1.get(0).GetQuantizedObjectsDataProvider()
  }
}
