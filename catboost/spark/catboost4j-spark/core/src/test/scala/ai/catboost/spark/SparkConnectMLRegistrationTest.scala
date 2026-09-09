package ai.catboost.spark;

import scala.io.Source

import collection.JavaConverters._

import org.apache.spark.ml.{Estimator,Transformer}

import org.junit.{Assert,Test}


/**
 * Check that CatBoost estimators and models are registered for discovery by Spark Connect ML
 *  (Spark 4.0+) and satisfy its reflective contract: a public no-arg constructor (required by
 *  ServiceLoader), a (uid: String) constructor for estimators and a static load(path: String)
 *  method for models.
 */
class SparkConnectMLRegistrationTest {
  private def registeredClassNames(serviceInterface: Class[_]) : Set[String] = {
    getClass.getClassLoader.getResources("META-INF/services/" + serviceInterface.getName).asScala.flatMap(
      url => Source.fromInputStream(url.openStream(), "UTF-8").getLines().map(_.trim).filter(
        line => line.nonEmpty && !line.startsWith("#")
      )
    ).toSet
  }

  @Test
  @throws(classOf[Exception])
  def testEstimatorsAreRegistered(): Unit = {
    val registered = registeredClassNames(classOf[Estimator[_]])
    for (cls <- Seq(classOf[CatBoostClassifier], classOf[CatBoostRegressor])) {
      Assert.assertTrue(registered.contains(cls.getName))
      Assert.assertTrue(classOf[Estimator[_]].isAssignableFrom(cls))
      cls.getConstructor()
      cls.getConstructor(classOf[String])
    }
  }

  @Test
  @throws(classOf[Exception])
  def testModelsAreRegistered(): Unit = {
    val registered = registeredClassNames(classOf[Transformer])
    for (cls <- Seq(classOf[CatBoostClassificationModel], classOf[CatBoostRegressionModel])) {
      Assert.assertTrue(registered.contains(cls.getName))
      Assert.assertTrue(classOf[Transformer].isAssignableFrom(cls))
      cls.getConstructor()
      cls.getMethod("load", classOf[String])
    }
  }
}
