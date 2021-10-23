package ai.catboost.spark.impl

import ai.catboost.spark._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

/**
 * Separate app is needed to start Master as an independent process because it can call abort() on
 *  network errors and we don't want to abort() main Spark driver process.
 * On the other hand we don't want to pack separate CatBoost CLI apps for each platform in addition to
 * catboost4j-spark-impl library, so that's why there's a simple JVM-based CLI wrapper
 */
private[spark] object AppWrapper {
  /**
   * Accepts single argument with JSON training params
   */
  def apply(mainImpl: () => Int) : Unit = {
    try {
      val returnValue = mainImpl();
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


private[spark] object MasterApp {
  /**
   * Accepts single argument with JSON training params
   */
  def main(args: Array[String]) : Unit = {
    AppWrapper(() => native_impl.ModeFitImpl(new TVector_TString(args)))
  }
}

private[spark] object ShutdownWorkersApp {
  /**
   * Accepts single argument with path to hosts file
   */
  def main(args: Array[String]) : Unit = {
    if (args.length != 2) {
      throw new java.lang.RuntimeException(
        "ShutdownWorkersApp expects two arguments: <path_to_hosts_file> <timeout_in_sec>"
      )
    }
    AppWrapper(() => { native_impl.ShutdownWorkers(args(0), args(1).toInt); 0 })
  }
}
