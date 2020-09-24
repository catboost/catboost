package ai.catboost.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util.Identifiable

import org.apache.spark.ml.CatBoostRegressorBase // defined inside catboost4j-spark

import ai.catboost.spark.params._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl


class CatBoostRegressionModel (
  override val uid: String,
  var nativeModel : native_impl.TFullModel = null,
  var nativeDimension: Int
)
  extends RegressionModel[Vector, CatBoostRegressionModel]
    with CatBoostModelTrait[CatBoostRegressionModel]
{
  def this(nativeModel : native_impl.TFullModel) = this(
    Identifiable.randomUID("CatBoostRegressionModel"),
    nativeModel,
    nativeDimension = 1 // always for regression
  )

  override def copy(extra: ParamMap): CatBoostRegressionModel = {
    val newModel = defaultCopy[CatBoostRegressionModel](extra)
    newModel.nativeModel = this.nativeModel
    newModel.nativeDimension = this.nativeDimension
    newModel
  }

  /**
   * Prefer batch computations operating on datasets as a whole for efficiency
   */
  override def predict(features: Vector): Double = {
    predictRawImpl(features)(0)
  }
}


class CatBoostRegressor (override val uid: String)
  extends CatBoostRegressorBase[Vector, CatBoostRegressor, CatBoostRegressionModel]
    with CatBoostPredictorTrait[CatBoostRegressor, CatBoostRegressionModel, RegressorTrainingParamsTrait]
{
  def this() = this(Identifiable.randomUID("CatBoostRegressor"))

  override def copy(extra: ParamMap): CatBoostRegressor = defaultCopy(extra)

  protected override def createModel(nativeModel: native_impl.TFullModel): CatBoostRegressionModel = {
    new CatBoostRegressionModel(nativeModel)
  }
}

