package ai.catboost.spark

import org.apache.spark.ml.linalg._

import ru.yandex.catboost.spark.catboost4j_spark.core.src._

import ai.catboost.CatBoostError


private[spark] trait CatBoostModelTrait[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]]
  extends org.apache.spark.ml.PredictionModel[Vector, Model]
{
  protected var nativeModel: native_impl.TFullModel
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
}
