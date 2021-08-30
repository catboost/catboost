package org.apache.spark.ml

import org.apache.spark.ml.regression.{RegressionModel, Regressor}

// make public internal Spark Regressor class
abstract class CatBoostRegressorBase[F, R <: Regressor[F, R, M], M <: RegressionModel[F, M]] extends Regressor[F, R, M]