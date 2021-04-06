package ai.catboost.spark

//import org.apache.spark.ml.{Model,Predictor}

import collection.mutable
import collection.mutable.HashMap

import java.net._
import java.nio.file._
import java.util.concurrent.{ExecutorCompletionService,Executors}

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.{DataFrame,Dataset,Row}
import org.apache.spark.TaskContext
import org.apache.spark.storage.StorageLevel

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.spark.impl.{CtrsContext,CtrFeatures}


/**
 * Base trait with common functionality for both [[CatBoostClassifier]] and [[CatBoostRegressor]]
 */
trait CatBoostPredictorTrait[
  Learner <: org.apache.spark.ml.Predictor[Vector, Learner, Model],
  Model <: org.apache.spark.ml.PredictionModel[Vector, Model]]
    extends org.apache.spark.ml.Predictor[Vector, Learner, Model]
      with params.DatasetParamsTrait
      with DefaultParamsWritable
{
  this: params.TrainingParamsTrait =>

  /**
   *  @return (preprocessedTrainPool, preprocessedEvalPools, ctrsContext)
   */
  protected def addEstimatedCtrFeatures(
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool],
    catBoostJsonParams: JObject
  ) : (Pool, Array[Pool], CtrsContext) = {
    val catFeaturesMaxUniqValueCount = native_impl.CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(
      quantizedTrainPool.quantizedFeaturesInfo.__deref__()
    )

    val oneHotMaxSize = native_impl.GetOneHotMaxSize(
      catFeaturesMaxUniqValueCount, 
      quantizedTrainPool.isDefined(quantizedTrainPool.labelCol),
      compact(catBoostJsonParams)
    )
    if (catFeaturesMaxUniqValueCount > oneHotMaxSize) {
      CtrFeatures.addCtrsAsEstimated(
        quantizedTrainPool,
        quantizedEvalPools,
        this,
        oneHotMaxSize
      )
    } else {
      (quantizedTrainPool, quantizedEvalPools, null)
    }
  }
    

  /**
   *  override in descendants if necessary
   *
   *  @return (preprocessedTrainPool, preprocessedEvalPools, catBoostJsonParams, ctrsContext)
   */
  protected def preprocessBeforeTraining(
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool]
  ) : (Pool, Array[Pool], JObject, CtrsContext) = {
    val catBoostJsonParams = ai.catboost.spark.params.Helpers.sparkMlParamsToCatBoostJsonParams(this)
    val (preprocessedTrainPool, preprocessedEvalPools, ctrsContext) 
        = addEstimatedCtrFeatures(quantizedTrainPool, quantizedEvalPools, catBoostJsonParams)
    (
      preprocessedTrainPool, 
      preprocessedEvalPools, 
      catBoostJsonParams, 
      ctrsContext
    )
  }

  protected def createModel(fullModel: TFullModel) : Model;

  protected override def train(dataset: Dataset[_]): Model = {
    val pool = new Pool(dataset.asInstanceOf[DataFrame])
    this.copyValues(pool)
    fit(pool)
  }

  /**
   * Additional variant of `fit` method that accepts CatBoost's [[Pool]] s and allows to specify additional
   * datasets for computing evaluation metrics and overfitting detection similarily to CatBoost's other APIs.
   *
   * @param trainPool The input training dataset.
   * @param evalPools The validation datasets used for the following processes:
   *  - overfitting detector
   *  - best iteration selection
   *  - monitoring metrics' changes
   * @return trained model
   */
  def fit(trainPool: Pool, evalPools: Array[Pool] = Array[Pool]()): Model = {
    ai.catboost.spark.params.Helpers.checkParamsCompatibility(
      this.getClass.getName,
      this,
      "trainPool",
      trainPool
    )
    for (i <- 0 until evalPools.length) {
      ai.catboost.spark.params.Helpers.checkParamsCompatibility(
        this.getClass.getName,
        this,
        s"evalPool #$i",
        evalPools(i)
      )
    }

    val spark = trainPool.data.sparkSession

    val quantizedTrainPool = if (trainPool.isQuantized) {
      trainPool
    } else {
      val quantizationParams = new ai.catboost.spark.params.QuantizationParams
      this.copyValues(quantizationParams)
      this.logInfo(s"fit. schedule quantization for train dataset")
      trainPool.quantize(quantizationParams)
    }

    // TODO(akhropov): eval pools are not distributed for now, so they are not repartitioned
    var evalIdx = 0
    val quantizedEvalPools = evalPools.map {
      evalPool => {
        evalIdx = evalIdx + 1
        if (evalPool.isQuantized) {
          evalPool
        } else {
          this.logInfo(s"fit. schedule quantization for eval dataset #${evalIdx - 1}")
          evalPool.quantize(quantizedTrainPool.quantizedFeaturesInfo)
        }
      }
    }
    val (preprocessedTrainPool, preprocessedEvalPools, catBoostJsonParams, ctrsContext) 
      = preprocessBeforeTraining(
        quantizedTrainPool,
        quantizedEvalPools
      )

    this.logInfo("fit. persist preprocessedTrainPool: start")
    preprocessedTrainPool.persist(StorageLevel.MEMORY_ONLY)
    this.logInfo("fit. persist preprocessedTrainPool: finish")

    val partitionCount = get(sparkPartitionCount).getOrElse(SparkHelpers.getWorkerCount(spark))
    this.logInfo(s"fit. partitionCount=${partitionCount}")
    
    val precomputedOnlineCtrMetaDataAsJsonString = if (ctrsContext != null) {
      ctrsContext.precomputedOnlineCtrMetaDataAsJsonString
    } else {
      null
    }

    val master = impl.CatBoostMasterWrapper(
      preprocessedTrainPool,
      preprocessedEvalPools,
      compact(catBoostJsonParams),
      precomputedOnlineCtrMetaDataAsJsonString
    )

    val trainingDriver : TrainingDriver = new TrainingDriver(
      listeningPort = 0,
      workerCount = partitionCount,
      startMasterCallback = master.trainCallback,
      workerInitializationTimeout = getOrDefault(workerInitializationTimeout)
    )

    val listeningPort = trainingDriver.getListeningPort
    this.logInfo(s"fit. TrainingDriver listening port = ${listeningPort}")

    this.logInfo(s"fit. Training started")
    
    val ecs = new ExecutorCompletionService[Unit](Executors.newFixedThreadPool(2))

    val trainingDriverFuture = ecs.submit(trainingDriver, ())

    val workers = new impl.CatBoostWorkers(
      spark,
      partitionCount,
      listeningPort,
      preprocessedTrainPool,
      catBoostJsonParams,
      precomputedOnlineCtrMetaDataAsJsonString,
      master.savedPoolsFuture
    )

    val workersFuture = ecs.submit(workers, ())

    val firstCompletedFuture = ecs.take()

    if (firstCompletedFuture == workersFuture) {
      impl.Helpers.checkOneFutureAndWaitForOther(workersFuture, trainingDriverFuture, "workers")
    } else { // firstCompletedFuture == trainingDriverFuture
      impl.Helpers.checkOneFutureAndWaitForOther(trainingDriverFuture, workersFuture, "master")
    }
    
    this.logInfo(s"fit. Training finished")

    val resultModel = createModel(
      if (ctrsContext != null) {
        this.logInfo(s"fit. Add CtrProvider to model")
        CtrFeatures.addCtrProviderToModel(
          master.nativeModelResult,
          ctrsContext,
          preprocessedTrainPool,
          preprocessedEvalPools
        ) 
      } else {
        master.nativeModelResult 
      }
    )

    preprocessedTrainPool.unpersist()

    resultModel
  }
}
