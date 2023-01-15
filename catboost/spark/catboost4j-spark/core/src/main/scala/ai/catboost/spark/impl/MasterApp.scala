package ai.catboost.spark.impl

import ai.catboost.spark._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

/**
 * Separate app is needed to start Master as an independent process because it can call abort() on
 *  network errors and we don't want to abort() main Spark driver process.
 * On the other hand we don't want to pack separate CatBoost CLI apps for each platform in addition to
 * catboost4j-spark-impl library, so that's why there's a simple JVM-based CLI wrapper
 */
private[spark] object MasterApp {
  /**
   * Accepts single argument with JSON training params
   */
  def main(args: Array[String]) : Unit = {
    try {
      val returnValue = native_impl.ModeFitImpl(new TVector_TString(args));
      if (returnValue != 0) {
        sys.exit(returnValue)
      }
    } catch {
      case t : Throwable => {
        t.printStackTrace()
        sys.exit(1)
      }
    }
  }
}
