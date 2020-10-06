package ai.catboost.spark;

import collection.mutable
import collection.mutable.HashMap
import collection.JavaConverters._

import java.nio.file.{Files,Path}

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import ai.catboost.CatBoostError

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._


private[spark] object FeaturesColumnStorage {
  def apply(quantizedFeaturesInfo: QuantizedFeaturesInfoPtr) : FeaturesColumnStorage = {
    val ui8FeatureIndicesVec = new TVector_i32
    val ui16FeatureIndicesVec = new TVector_i32

    native_impl.GetActiveFloatFeaturesIndices(
      quantizedFeaturesInfo,
      ui8FeatureIndicesVec,
      ui16FeatureIndicesVec
    )

    val ui8FeatureIndices = ui8FeatureIndicesVec.toPrimitiveArray
    val ui16FeatureIndices = ui16FeatureIndicesVec.toPrimitiveArray

    val buffersUi8 = new Array[TVector_i64](ui8FeatureIndices.length)
    for (i <- 0 until ui8FeatureIndices.length) {
      buffersUi8(i) = new TVector_i64
    }
    val buffersUi16 = new Array[TVector_i64](ui16FeatureIndices.length)
    for (i <- 0 until ui16FeatureIndices.length) {
      buffersUi16(i) = new TVector_i64
    }

    new FeaturesColumnStorage(
      ui8FeatureIndices,
      ui16FeatureIndices,
      buffersUi8,
      buffersUi16,
      new Array[java.nio.ByteBuffer](ui8FeatureIndices.length),
      new Array[java.nio.ByteBuffer](ui16FeatureIndices.length)
    )
  }
}

/**
 * store quantized feature columns in C++'s TVectors<ui64> to be zero-copy passed to TQuantizedDataProvider
 * expose their memory via JVM's java.nio.ByteBuffer.
 * size grows dynamically by adding rows' features data
 */
private[spark] class FeaturesColumnStorage (
  val indicesUi8: Array[Int],
  val indicesUi16: Array[Int],
  val buffersUi8: Array[TVector_i64],
  val buffersUi16: Array[TVector_i64],
  var javaBuffersUi8: Array[java.nio.ByteBuffer],
  var javaBuffersUi16: Array[java.nio.ByteBuffer],
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

    pos = pos + 1
  }

  private def ceilDiv(x: Int, y: Int) : Int = (x + y - 1) /  y

  private def realloc(newSize: Int) = {
    val sizeForUi8 = ceilDiv(newSize, 8)
    val sizeForUi16 = ceilDiv(newSize, 4)
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
  }

  def addToVisitor(visitor: IQuantizedFeaturesDataVisitor) = {
    for (i <- 0 until indicesUi8.length) {
      visitor.AddFloatFeature(indicesUi8(i), pos, 8, buffersUi8(i))
    }
    for (i <- 0 until indicesUi16.length) {
      visitor.AddFloatFeature(indicesUi16(i), pos, 16, buffersUi16(i))
    }
  }

}


private[spark] class ProcessRowsOutputIterator(
    val dstRows : mutable.ArrayBuffer[Array[Any]],
    val processRowCallback: (Array[Any], Int) => Array[Any], // add necessary data to row
    var objectIdx : Int = 0
) extends Iterator[Row] {
  def next : Row = {
    val dstRow = processRowCallback(dstRows(objectIdx), objectIdx)
    dstRows(objectIdx) = null // to speed up cleanup
    objectIdx = objectIdx + 1
    return Row.fromSeq(dstRow)
  }

  def hasNext : Boolean = {
    return objectIdx < dstRows.size
  }
}


private[spark] object DataHelpers {
  def makeFeaturesMetadata(initialFeatureNames: Array[String]) : Metadata = {
    val featureNames = new Array[String](initialFeatureNames.length)

    val featureNamesSet = new mutable.HashSet[String]()

    for (i <- 0 until featureNames.size) {
      val name = initialFeatureNames(i)
      if (name.isEmpty) {
        val generatedName = s"_f$i"
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

  def getClassNamesFromLabelData(data: DataFrame, labelColumn: String) : Array[String] = {
    val iterator = data.select(labelColumn).distinct.toLocalIterator.asScala
    data.schema(labelColumn).dataType match {
      case DataTypes.IntegerType => iterator.map{ _.getAs[Int](0) }.toSeq.sorted.map{ _.toString }.toArray
      case DataTypes.LongType => iterator.map{ _.getAs[Long](0) }.toSeq.sorted.map{ _.toString }.toArray
      case DataTypes.FloatType => iterator.map{ _.getAs[Float](0) }.toSeq.sorted.map{ _.toString }.toArray
      case DataTypes.DoubleType => iterator.map{ _.getAs[Double](0) }.toSeq.sorted.map{ _.toString }.toArray
      case DataTypes.StringType => iterator.map{ _.getString(0) }.toArray
      case _ => throw new CatBoostError("Unsupported data type for Label")
    }
  }

  /**
   * @return (dstRows, rawObjectDataProvider)
   */
  def processDatasetWithRawFeatures(
    rows: Iterator[Row],
    featuresColumnIdx: Int,
    featuresLayout: TFeaturesLayoutPtr,
    availableFeaturesIndices: Array[Int],
    keepRawFeaturesInDstRows: Boolean,
    dstRowLength: Int
  ) : (mutable.ArrayBuffer[Array[Any]], SWIGTYPE_p_NCB__TRawObjectsDataProviderPtr) = {
    val dstRows = new mutable.ArrayBuffer[Array[Any]]

    // as columns
    var availableFeaturesData = new Array[mutable.ArrayBuilder[Float]](availableFeaturesIndices.size)
    for (i <- 0 until availableFeaturesData.size) {
      availableFeaturesData(i) = mutable.ArrayBuilder.make[Float]
    }

    rows.foreach {
      row => {
         val rowFields = new Array[Any](dstRowLength)
         for (i <- 0 until row.length) {
           if (i == featuresColumnIdx) {
             val featuresValues = row.getAs[Vector](i)
             for (j <- 0 until availableFeaturesIndices.size) {
               availableFeaturesData(j) += featuresValues(availableFeaturesIndices(j)).toFloat
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

    val availableFeaturesDataForBuilder = new TVector_TMaybeOwningConstArrayHolder_float
    for (featureData <- availableFeaturesData) {
      val result = featureData.result
      availableFeaturesDataForBuilder.add(result)
    }

    val rawObjectsDataProviderPtr = native_impl.CreateRawObjectsDataProvider(
      featuresLayout,
      dstRows.size.toLong,
      availableFeaturesDataForBuilder
    )

    // try to force cleanup of no longer used data
    availableFeaturesData = null
    System.gc()

    (dstRows, rawObjectsDataProviderPtr)
  }

  def getLabelCallback(
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
      case _ => throw new CatBoostError("Unsupported data type for Label")
    }
  }

  def getFloatCallback(
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
   * Create quantized data provider from iterating over DataFrame's Rows.
   * @returns quantized data provider. type is TDataProviderPtr because that's generic interface that
   *  clients (like training, prediction, feature quality estimators) accept
   */
  def loadQuantizedDataset(
    quantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
    columnIndexMap: HashMap[String, Int], // column type -> idx in schema
    dataMetaInfo: TIntermediateDataMetaInfo,
    schema: StructType,
    threadCount: Int,
    rows: Iterator[Row]
  ) : TDataProviderPtr = {

    val callbacks = new mutable.ArrayBuffer[Row => Unit]
    val postprocessingCallbacks = new mutable.ArrayBuffer[() => Unit]

    val dataProviderBuilderOptions = new TDataProviderBuilderOptions

    val dataProviderClosure = new TDataProviderClosureForJVM(
      EDatasetVisitorType.QuantizedFeatures,
      dataProviderBuilderOptions,
      columnIndexMap.contains("features"),
      threadCount
    )
    val visitor = dataProviderClosure.GetQuantizedVisitor
    if (visitor == null) {
      throw new CatBoostError("Failure to create IQuantizedFeaturesDataVisitor")
    }

    // Add column handlers

    if (columnIndexMap.contains("features")) {
      val fieldIdx = columnIndexMap("features")

      val featuresColumnStorage = FeaturesColumnStorage(quantizedFeaturesInfo)

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
          case ERawTargetType.Float | ERawTargetType.Integer =>
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

    var objectCount = 0

    rows.foreach {
      row => {
        callbacks.foreach(_(row))
        objectCount = objectCount + 1
      }
    }

    dataMetaInfo.setObjectCount(java.math.BigInteger.valueOf(objectCount))

    visitor.Start(dataMetaInfo, objectCount, quantizedFeaturesInfo.__deref__)

    postprocessingCallbacks.foreach(_())

    visitor.Finish

    dataProviderClosure.GetResult()
  }


  /**
   * @returns (pool with columns for training, map of column type -> index in schema)
   */
  def selectColumnsForTrainingAndReturnIndex(
    pool: Pool,
    includeFeatures: Boolean
  ) : (DataFrame, HashMap[String, Int]) = {
    val columnTypesMap = new mutable.HashMap[String, Int]()

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
    var columnsList = new mutable.ArrayBuffer[String]()
    var i = 0
    for (columnTypeName <- columnTypeNames) {
      val param = pool.getParam(columnTypeName + "Col").asInstanceOf[Param[String]]
      if (pool.isDefined(param)) {
        val paramValue = pool.getOrDefault(param)
        columnsList += paramValue
        columnTypesMap.update(paramValue, i)
        i = i + 1
      }
    }

    val dfWithColumnsForTraining = pool.data.select(columnsList.head, columnsList.tail: _*)
    (dfWithColumnsForTraining, columnTypesMap)
  }

  def downloadQuantizedPoolToTempFile(
    pool: Pool,
    includeFeatures: Boolean,
    threadCount: Int = 1,
    tmpFilePrefix: String = null,
    tmpFileSuffix: String = null
  ) : Path = {
    val (selectedDF, columnIndexMap) = selectColumnsForTrainingAndReturnIndex(pool, includeFeatures)

    val dataProvider = loadQuantizedDataset(
      pool.quantizedFeaturesInfo,
      columnIndexMap,
      pool.createDataMetaInfo,
      selectedDF.schema,
      threadCount,
      selectedDF.toLocalIterator.asScala
    )

    val tmpFilePath = Files.createTempFile(tmpFilePrefix, tmpFileSuffix)
    tmpFilePath.toFile.deleteOnExit

    native_impl.SaveQuantizedPoolWrapper(dataProvider, tmpFilePath.toString)
    tmpFilePath
  }
}
