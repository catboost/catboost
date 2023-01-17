package ai.catboost.spark

//import org.apache.spark.ml.{Model,Predictor}

import collection.mutable
import collection.mutable.HashMap
import util.control.Breaks._

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
import ai.catboost.CatBoostError


class CatBoostTrainingContext (
  val ctrsContext: CtrsContext,
  val catBoostJsonParams: JObject,
  val serializedLabelConverter: TVector_i8
)


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
    updatedCatBoostJsonParams: JObject,  // with set loss_function and class labels can be inferred
    classTargetPreprocessor: Option[TClassTargetPreprocessor] = None,
    serializedLabelConverter: TVector_i8 = new TVector_i8
  ) : (Pool, Array[Pool], CtrsContext) = {
    val catFeaturesMaxUniqValueCount = native_impl.CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(
      quantizedTrainPool.quantizedFeaturesInfo.__deref__()
    )

    val oneHotMaxSize = native_impl.GetOneHotMaxSize(
      catFeaturesMaxUniqValueCount, 
      quantizedTrainPool.isDefined(quantizedTrainPool.labelCol),
      compact(updatedCatBoostJsonParams)
    )
    if (catFeaturesMaxUniqValueCount > oneHotMaxSize) {
      CtrFeatures.addCtrsAsEstimated(
        quantizedTrainPool,
        quantizedEvalPools,
        updatedCatBoostJsonParams,
        oneHotMaxSize,
        classTargetPreprocessor,
        serializedLabelConverter
      )
    } else {
      (quantizedTrainPool, quantizedEvalPools, null)
    }
  }
    

  /**
   *  override in descendants if necessary
   *
   *  @return (preprocessedTrainPool, preprocessedEvalPools, catBoostTrainingContext)
   */
  protected def preprocessBeforeTraining(
    quantizedTrainPool: Pool,
    quantizedEvalPools: Array[Pool]
  ) : (Pool, Array[Pool], CatBoostTrainingContext) = {
    val catBoostJsonParams = ai.catboost.spark.params.Helpers.sparkMlParamsToCatBoostJsonParams(this)
    val (preprocessedTrainPool, preprocessedEvalPools, ctrsContext) 
        = addEstimatedCtrFeatures(quantizedTrainPool, quantizedEvalPools, catBoostJsonParams)
    (
      preprocessedTrainPool, 
      preprocessedEvalPools, 
      new CatBoostTrainingContext(
        ctrsContext,
        catBoostJsonParams,
        new TVector_i8
      )
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
    val (preprocessedTrainPool, preprocessedEvalPools, catBoostTrainingContext)
      = preprocessBeforeTraining(
        quantizedTrainPool,
        quantizedEvalPools
      )

    val partitionCount = get(sparkPartitionCount).getOrElse(SparkHelpers.getWorkerCount(spark))
    this.logInfo(s"fit. partitionCount=${partitionCount}")

    this.logInfo("fit. train.prepareDatasetForTraining: start")
    val preparedTrainDataset = DataHelpers.prepareDatasetForTraining(
      preprocessedTrainPool, 
      datasetIdx=0.toByte,
      workerCount=partitionCount
    )
    this.logInfo("fit. train.prepareDatasetForTraining: finish")

    val preparedEvalDatasets = preprocessedEvalPools.zipWithIndex.map {
      case (evalPool, evalIdx) => {
        this.logInfo(s"fit. eval #${evalIdx}.prepareDatasetForTraining: start")
        val preparedEvalDataset = DataHelpers.prepareDatasetForTraining(
          evalPool, 
          datasetIdx=(evalIdx + 1).toByte,
          workerCount=partitionCount
        )
        this.logInfo(s"fit. eval #${evalIdx}.prepareDatasetForTraining: finish")
        preparedEvalDataset
      }
    }

    val precomputedOnlineCtrMetaDataAsJsonString = if (catBoostTrainingContext.ctrsContext != null) {
      catBoostTrainingContext.ctrsContext.precomputedOnlineCtrMetaDataAsJsonString
    } else {
      null
    }

    val master = impl.CatBoostMasterWrapper(
      preparedTrainDataset,
      preparedEvalDatasets,
      compact(catBoostTrainingContext.catBoostJsonParams),
      precomputedOnlineCtrMetaDataAsJsonString
    )

    val connectTimeoutValue = getOrDefault(connectTimeout)
    val workerInitializationTimeoutValue = getOrDefault(workerInitializationTimeout)
    val workerMaxFailuresValue = getOrDefault(workerMaxFailures)
    val workerListeningPortValue = getOrDefault(workerListeningPort)

    val workers = impl.CatBoostWorkers(
      spark,
      partitionCount,
      connectTimeoutValue,
      workerInitializationTimeoutValue,
      workerListeningPortValue,
      preparedTrainDataset,
      preparedEvalDatasets,
      catBoostTrainingContext.catBoostJsonParams,
      catBoostTrainingContext.serializedLabelConverter,
      precomputedOnlineCtrMetaDataAsJsonString,
      master.savedPoolsFuture
    )

    breakable {
      // retry training if network connection issues were the reason of failure
      while (true) {
        val trainingDriver : TrainingDriver = new TrainingDriver(
          listeningPort = getOrDefault(trainingDriverListeningPort),
          workerCount = partitionCount,
          startMasterCallback = master.trainCallback,
          connectTimeout = connectTimeoutValue,
          workerInitializationTimeout = workerInitializationTimeoutValue
        )

        try {
          val listeningPort = trainingDriver.getListeningPort
          this.logInfo(s"fit. TrainingDriver listening port = ${listeningPort}")

          this.logInfo(s"fit. Training started")

          val ecs = new ExecutorCompletionService[Unit](Executors.newFixedThreadPool(2))

          val trainingDriverFuture = ecs.submit(trainingDriver, ())

          val workersFuture = ecs.submit(
            new Runnable {
              def run = {
                workers.run(listeningPort)
              }
            },
            ()
          )

          var catboostWorkersConnectionLost = false
          try {
            impl.Helpers.waitForTwoFutures(ecs, trainingDriverFuture, "master", workersFuture, "workers")
            break
          } catch {
            case e : java.util.concurrent.ExecutionException => {
              e.getCause match {
                case connectionLostException : CatBoostWorkersConnectionLostException => {
                  catboostWorkersConnectionLost = true
                }
                case _ => throw e
              }
            }
          }
          if (workers.workerFailureCount >= workerMaxFailuresValue) {
            throw new CatBoostError(s"CatBoost workers failed at least $workerMaxFailuresValue times")
          }
          if (catboostWorkersConnectionLost) {
            log.info(s"CatBoost master: communication with some of the workers has been lost. Retry training")
          } else {
            break
          }
        } finally {
          trainingDriver.close(tryToShutdownWorkers=true, waitToShutdownWorkers=false)
        }
      }
    }
    this.logInfo(s"fit. Training finished")

    val resultModel = createModel(
      if (catBoostTrainingContext.ctrsContext != null) {
        this.logInfo(s"fit. Add CtrProvider to model")
        CtrFeatures.addCtrProviderToModel(
          master.nativeModelResult,
          catBoostTrainingContext.ctrsContext,
          quantizedTrainPool,
          quantizedEvalPools
        ) 
      } else {
        master.nativeModelResult 
      }
    )

    preprocessedTrainPool.unpersist()

    resultModel
  }
}
