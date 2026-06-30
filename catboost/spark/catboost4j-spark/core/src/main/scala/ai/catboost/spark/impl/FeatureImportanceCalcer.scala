package ai.catboost.spark.impl

import scala.reflect.classTag

import collection.JavaConverters._
import collection.mutable

import org.apache.commons.lang3.tuple.Pair

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder,RowEncoder}
import org.apache.spark.sql.types._

import ai.catboost.spark._

import ai.catboost.CatBoostError

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._


class FeatureImportanceCalcer extends Logging {
  private[spark] def collectLeavesWeightsFromDataset(model: TFullModel, data: Pool) : Array[Double] = {
    val dataForApplication = data.quantizeForModelApplicationImpl(model)

    val leafCount = model.GetLeafCount()
    val result = new Array[Double](leafCount)

    dataForApplication.mapQuantizedPartitions(
      selectedColumns=Seq("features", "weight", "groupWeight"),
      includeEstimatedFeatures=false,
      includePairsIfPresent=false,
      dstColumnNames=Array[String](),
      dstRowLength=0,
      (
        dataProvider: TDataProviderPtr,
        estimatedDataProvider: TDataProviderPtr,
        dstRows: mutable.ArrayBuffer[Array[Any]],
        localExecutor: TLocalExecutor
      ) => {
        val leavesWeights = native_impl.CollectLeavesStatisticsWrapper(
          dataProvider,
          model,
          localExecutor
        )
        Iterator[linalg.DenseVector](new linalg.DenseVector(leavesWeights.toPrimitiveArray))
      }
    )(ExpressionEncoder(): Encoder[linalg.DenseVector], classTag[linalg.DenseVector]).toLocalIterator.asScala.foreach(
      (partialResult : linalg.DenseVector) => {
        for (i <- 0 until result.length) {
          result(i) = result(i) + partialResult(i)
        }
      }
    )
    result
  }

  def prepareTrees(
    model: TFullModel,
    data: Pool, // can be null if not needed
    preCalcMode: EPreCalcShapValues,
    calcInternalValues: Boolean,
    calcType: ECalcTypeShapValues,
    calcShapValuesByLeaf: Boolean,
    localExecutor: TLocalExecutor,
    modelOutputType: EExplainableModelOutput=EExplainableModelOutput.Raw,
    referenceData: Pool=null
  ) : TShapPreparedTrees = {
    if ((calcType == ECalcTypeShapValues.Independent) || (referenceData != null)) {
      throw new CatBoostError("Independent Tree SHAP values are not supported yet")
    }
    val needSumModelAndDatasetWeights = native_impl.HasNonZeroApproxForZeroWeightLeaf(model)
    val leavesWeightsFromDataset = if (!model.HasLeafWeights() || needSumModelAndDatasetWeights) {
      if (data == null) {
        throw new CatBoostError(
          "To calculate SHAP values, either a model with leaf weights, or a dataset are required."
        )
      }
      collectLeavesWeightsFromDataset(model, data)
    } else {
      Array[Double]()
    }
    native_impl.PrepareTreesWithoutIndependent(
      model,
      if (data != null) { data.count } else { -1L },
      needSumModelAndDatasetWeights,
      leavesWeightsFromDataset,
      preCalcMode,
      calcInternalValues,
      calcType,
      calcShapValuesByLeaf,
      localExecutor
    )
  }

  def calcLossFunctionChange(model: TFullModel, data: Pool, calcType: ECalcTypeShapValues) : Array[Double] = {
    if (data == null) {
      throw new CatBoostError("LossFunctionChange feature importance requires dataset specified")
    }

    val maxObjectCount = native_impl.GetMaxObjectCountForFstrCalc(data.count, data.getFeatureCount)
    val dataForLossChangeCalculation = if (maxObjectCount < data.count) {
      val sampledData = data.sample(maxObjectCount.toDouble / data.count.toDouble)
      logInfo(s"Selected ${sampledData.count} samples from ${data.count} for LossFunctionChange calculation.")
      sampledData
    } else {
      data
    }.quantizeForModelApplicationImpl(model).ensurePartitionByGroupsIfPresent()

    val spark = data.data.sparkSession

    val threadCount = SparkHelpers.getThreadCountForDriver(spark)
    val localExecutor = new TLocalExecutor
    localExecutor.Init(threadCount)

    val preparedTrees = prepareTrees(
      model,
      dataForLossChangeCalculation,
      EPreCalcShapValues.Auto,
      calcInternalValues=true,
      calcType=calcType,
      calcShapValuesByLeaf=true,
      localExecutor=localExecutor
    )
    val combinationClassFeatures = native_impl.GetCombinationClassFeatures(model)
    val featuresCount = combinationClassFeatures.size().toInt

    var aggregatedStats : Array[Double] = null

    dataForLossChangeCalculation.mapQuantizedPartitions(
      selectedColumns=Seq("groupId", "label", "features", "weight", "groupWeight"),
      includeEstimatedFeatures=false,
      includePairsIfPresent=true,
      dstColumnNames=Array[String](),
      dstRowLength=0,
      (
        dataProvider: TDataProviderPtr,
        estimatedDataProvider: TDataProviderPtr,
        dstRows: mutable.ArrayBuffer[Array[Any]],
        localExecutor: TLocalExecutor
      ) => {
        val result = native_impl.CalcFeatureEffectLossChangeMetricStatsWrapper(
          model,
          featuresCount,
          preparedTrees,
          dataProvider,
          calcType,
          localExecutor
        )
        Iterator[linalg.Vector](new linalg.DenseVector(result.toPrimitiveArray))
      }
    )(ExpressionEncoder(): Encoder[linalg.Vector], classTag[linalg.Vector]).toLocalIterator.asScala.foreach(
      (partialResult : linalg.Vector) => {
        if (aggregatedStats == null) {
          aggregatedStats = partialResult.toArray
        } else {
          for (i <- 0 until aggregatedStats.length) {
            aggregatedStats(i) = aggregatedStats(i) + partialResult(i)
          }
        }
      }
    )

    native_impl.CalcFeatureEffectLossChangeFromScores(
      model,
      combinationClassFeatures,
      aggregatedStats
    ).toPrimitiveArray
  }

  def calcPredictionValuesChange(model: TFullModel, data: Pool) : Array[Double] = {
    val leavesWeightsFromDataset = if (data != null) {
      logInfo("Used dataset leave statistics for fstr calculation")
      collectLeavesWeightsFromDataset(model, data)
    } else {
      Array[Double]()
    }
    native_impl.CalcFeatureEffectAverageChangeWrapper(model, leavesWeightsFromDataset).toPrimitiveArray
  }

  def calcPredictionDiff(model: TFullModel, data: Pool) : Array[Double] = {
    if (data == null) {
      throw new CatBoostError("PredictionDiff feature importance requires dataset specified")
    }
    if (data.isQuantized) {
      throw new CatBoostError("PredictionDiff feature importance does not support quantized datasets")
    }

    val threadCount = SparkHelpers.getThreadCountForDriver(data.data.sparkSession)
    val localExecutor = new TLocalExecutor
    localExecutor.Init(threadCount)

    val (_, rawObjectsDataProvider) = DataHelpers.processDatasetWithRawFeatures(
      data.data.select(data.getFeaturesCol).toLocalIterator.asScala,
      featuresColumnIdx=0,
      featuresLayout=data.getFeaturesLayout,
      maxUniqCatFeatureValues=data.getCatFeaturesUniqValueCounts.max,
      keepRawFeaturesInDstRows=false,
      dstRowLength=0,
      localExecutor=localExecutor
    )
    native_impl.GetPredictionDiffWrapper(model, rawObjectsDataProvider, localExecutor).toPrimitiveArray
  }

  /**
   * Supported values of fstrType are FeatureImportance, PredictionValuesChange, LossFunctionChange, PredictionDiff
   * @param data
   *  if fstrType is PredictionDiff it is required and must contain 2 samples
   *  if fstrType is PredictionValuesChange this param is required in case if model was explicitly trained
   *   with flag to store no leaf weights.
   *  otherwise it can be null
   * @return array of feature importances (index corresponds to order of features in the model)
   */
  def calc(
    model: TFullModel,
    fstrType: EFstrType,
    data: Pool=null,
    calcType: ECalcTypeShapValues=ECalcTypeShapValues.Regular
  ) : Array[Double] = {
    val resolvedFstrType = if (fstrType == EFstrType.FeatureImportance) {
      native_impl.GetDefaultFstrType(model)
    } else {
      fstrType
    }

    resolvedFstrType match {
      case EFstrType.PredictionValuesChange => this.calcPredictionValuesChange(model, data)
      case EFstrType.LossFunctionChange => this.calcLossFunctionChange(model, data, calcType)
      case EFstrType.PredictionDiff => this.calcPredictionDiff(model, data)
      case _ => throw new CatBoostError(s"getFeatureImportance: unexpected fstrType: ${fstrType}")
    }
  }

  def calcShapValues(
    model: TFullModel,
    data: Pool,
    preCalcMode: EPreCalcShapValues,
    calcType: ECalcTypeShapValues,
    modelOutputType: EExplainableModelOutput,
    referenceData: Pool,
    outputColumns: Array[String]
  ) : DataFrame = {
    val dataForApplication = data.quantizeForModelApplicationImpl(model)

    val threadCount = SparkHelpers.getThreadCountForDriver(data.data.sparkSession)
    val localExecutor = new TLocalExecutor
    localExecutor.Init(threadCount)

    val preparedTrees = prepareTrees(
      model,
      dataForApplication,
      preCalcMode,
      calcInternalValues=false,
      calcType=calcType,
      calcShapValuesByLeaf=true,
      localExecutor=localExecutor,
      modelOutputType=modelOutputType,
      referenceData=referenceData
    )

    val modelDimensionsCount = model.GetDimensionsCount().toInt

    val dstSchema = StructType(
      DataHelpers.selectSchemaFields(dataForApplication.data.schema, outputColumns)
      :+ StructField(
          "shapValues",
          if (modelDimensionsCount > 1) {
            linalg.SQLDataTypes.MatrixType
          } else {
            linalg.SQLDataTypes.VectorType
          },
          false
        )
    )

    dataForApplication.mapQuantizedPartitions(
      selectedColumns=Seq("features"),
      includeEstimatedFeatures=false,
      includePairsIfPresent=true,
      dstColumnNames=outputColumns,
      dstRowLength=dstSchema.length - 1,
      (
        dataProvider: TDataProviderPtr,
        estimatedDataProvider: TDataProviderPtr,
        dstRows: mutable.ArrayBuffer[Array[Any]],
        localExecutor: TLocalExecutor
      ) => {
        val result = native_impl.CalcShapValuesWithPreparedTreesWrapper(
          model,
          dataProvider,
          preparedTrees,
          calcType,
          localExecutor
        )
        val objectCount = result.GetObjectCount
        val shapValuesCount = result.GetShapValuesCount
        (if (modelDimensionsCount > 1) {
          (0 until objectCount).map(
            objectIdx => {
              val shapValues = new linalg.DenseMatrix(
                modelDimensionsCount,
                shapValuesCount,
                result.Get(objectIdx).toPrimitiveArray,
                isTransposed=true
              )
              Row.fromSeq(dstRows(objectIdx).toSeq :+ shapValues)
            }
          )
        } else {
          (0 until objectCount).map(
            objectIdx => {
              val shapValues = new linalg.DenseVector(result.Get(objectIdx).toPrimitiveArray)
              Row.fromSeq(dstRows(objectIdx).toSeq :+ shapValues)
            }
          )
        }).toIterator
      }
    )(RowEncoderConstructor.construct(dstSchema), classTag[Row])
  }

  def calcShapInteractionValues(
    model: TFullModel,
    data: Pool,
    featureIndices: Pair[Int, Int], // can be null
    featureNames: Pair[String, String], // can be null
    preCalcMode: EPreCalcShapValues,
    calcType: ECalcTypeShapValues,
    outputColumns: Array[String]
  ) : DataFrame = {
    val dataForApplication = data.quantizeForModelApplicationImpl(model)

    val threadCount = SparkHelpers.getThreadCountForDriver(data.data.sparkSession)
    val localExecutor = new TLocalExecutor
    localExecutor.Init(threadCount)

    val preparedTrees = prepareTrees(
      model,
      dataForApplication,
      preCalcMode,
      calcInternalValues=true,
      calcType=calcType,
      calcShapValuesByLeaf=false,
      localExecutor=localExecutor
    )

    val modelDimensionsCount = model.GetDimensionsCount().toInt

    var dstSchemaFields = DataHelpers.selectSchemaFields(dataForApplication.data.schema, outputColumns)
    val outputColumnCount = dstSchemaFields.size
    if (modelDimensionsCount > 1) {
      dstSchemaFields = dstSchemaFields :+ StructField("classIdx", IntegerType, false)
    }
    dstSchemaFields = dstSchemaFields ++ Seq(
      StructField("featureIdx1", IntegerType, false),
      StructField("featureIdx2", IntegerType, false),
      StructField("shapInteractionValue", DoubleType, false)
    )

    val selectedFeatureIndices = new Array[Int](2)
    if (featureIndices != null) {
      if (featureNames != null) {
        throw new CatBoostError("only one of featureIndices of featureNames can be specified")
      }
      selectedFeatureIndices(0) = featureIndices.getLeft()
      selectedFeatureIndices(1) = featureIndices.getRight()
    } else if (featureNames != null) {
      native_impl.GetSelectedFeaturesIndices(
        model,
        featureNames.getLeft(),
        featureNames.getRight(),
        selectedFeatureIndices
      )
    } else {
      selectedFeatureIndices(0) = -1
      selectedFeatureIndices(1) = -1
    }

    dataForApplication.mapQuantizedPartitions(
      selectedColumns=Seq("features"),
      includeEstimatedFeatures=false,
      includePairsIfPresent=true,
      dstColumnNames=outputColumns,
      dstRowLength=outputColumnCount,
      (
        dataProvider: TDataProviderPtr,
        estimatedDataProvider: TDataProviderPtr,
        dstRows: mutable.ArrayBuffer[Array[Any]],
        localExecutor: TLocalExecutor
      ) => {
        val result = native_impl.CalcShapInteractionValuesWithPreparedTreesWrapper(
          model,
          dataProvider,
          selectedFeatureIndices,
          calcType,
          localExecutor,
          preparedTrees
        )
        val objectCount = result.GetObjectCount
        val shapInteractionValuesCount = result.GetShapInteractionValuesCount
        (if (modelDimensionsCount > 1) {
          val dstRow = new Array[Any](outputColumnCount + 4)
          (0 until objectCount).flatMap(
            objectIdx => {
              Array.copy(dstRows(objectIdx), 0, dstRow, 0, outputColumnCount)
              (0 until modelDimensionsCount).flatMap(
                dimension => {
                  dstRow(outputColumnCount) = dimension
                  val values = result.Get(objectIdx, dimension).toPrimitiveArray
                  (0 until shapInteractionValuesCount).flatMap(
                    idx1 => {
                      dstRow(outputColumnCount + 1) = idx1
                      (0 until shapInteractionValuesCount).map(
                        idx2 => {
                          dstRow(outputColumnCount + 2) = idx2
                          dstRow(outputColumnCount + 3) = values(idx1 * shapInteractionValuesCount  + idx2)
                          Row.fromSeq(dstRow.toSeq)
                        }
                      )
                    }
                  )
                }
              )
            }
          )
        } else {
          val dstRow = new Array[Any](outputColumnCount + 3)
          (0 until objectCount).flatMap(
            objectIdx => {
              Array.copy(dstRows(objectIdx), 0, dstRow, 0, outputColumnCount)
              val values = result.Get(objectIdx).toPrimitiveArray
              (0 until shapInteractionValuesCount).flatMap(
                idx1 => {
                  dstRow(outputColumnCount) = idx1
                  (0 until shapInteractionValuesCount).map(
                    idx2 => {
                      dstRow(outputColumnCount + 1) = idx2
                      dstRow(outputColumnCount + 2) = values(idx1 * shapInteractionValuesCount  + idx2)
                      Row.fromSeq(dstRow.toSeq)
                    }
                  )
                }
              )
            }
          )
        }).toIterator
      }
    )(RowEncoderConstructor.construct(StructType(dstSchemaFields)), classTag[Row])
  }

  def calcInteraction(model: TFullModel) : Array[FeatureInteractionScore] = {
    val firstIndicesVector = new TVector_i32
    val secondIndicesVector = new TVector_i32
    val scoresVector = new TVector_double

    native_impl.CalcInteraction(model, firstIndicesVector, secondIndicesVector, scoresVector)

    val firstIndices = firstIndicesVector.toPrimitiveArray
    val secondIndices = secondIndicesVector.toPrimitiveArray
    val scores = scoresVector.toPrimitiveArray

    val resultSize = scores.length

    val result = new Array[FeatureInteractionScore](resultSize)

    for (i <- 0 until resultSize) {
      result(i) = new FeatureInteractionScore(firstIndices(i), secondIndices(i), scores(i))
    }

    result
  }
}
