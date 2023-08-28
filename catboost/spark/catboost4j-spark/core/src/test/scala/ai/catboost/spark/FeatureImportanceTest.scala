package ai.catboost.spark

import collection.mutable
import collection.JavaConverters._

import scala.io.Source
import scala.math.BigInt

import java.nio.file.{Path,Paths}

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.junit.{Assert,Test,Rule}
import org.junit.rules.TemporaryFolder

import org.apache.spark.ml.linalg.{Matrix,Vector}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._


class FeatureImportanceTest {
  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  def assertPrettifiedEquals(
    expectedFeatureImportancesPrettified: List[Map[String,Any]],
    featureImportancesPrettified: Array[FeatureImportance]
  ) = {
    Assert.assertEquals(expectedFeatureImportancesPrettified.length, featureImportancesPrettified.length)

    expectedFeatureImportancesPrettified.zip(featureImportancesPrettified).foreach{
      case (expected, actual) => {
        Assert.assertEquals(expected("featureName").asInstanceOf[String], actual.featureName)
        Assert.assertEquals(expected("importance").asInstanceOf[Double], actual.importance, 1.0e-6)
      }
    }
  }

  @Test
  @throws(classOf[Exception])
  def testPredictionValuesChange() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "higgs")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = regressor.fit(trainPool)

    val expectedFeatureImportancesFile = canonicalDataDir.resolve("feature_importance_prediction_values_change.json")
    val expectedFeatureImportancesJson = parse(
      Source.fromFile(expectedFeatureImportancesFile.toString).getLines.mkString
    ).asInstanceOf[JObject]

    for (calcType <- Seq(ECalcTypeShapValues.Regular, ECalcTypeShapValues.Approximate, ECalcTypeShapValues.Exact)) {
      val expectedFeatureImportances = expectedFeatureImportancesJson
        .values(s"calc_type_${calcType}")
        .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
        .toArray

      val featureImportancesPredictionValuesChange = model.getFeatureImportance(
        fstrType=EFstrType.PredictionValuesChange,
        calcType=calcType
      )
      Assert.assertArrayEquals(expectedFeatureImportances, featureImportancesPredictionValuesChange, 1.0e-6)

      val featureImportancesDefault = model.getFeatureImportance(calcType=calcType)
      Assert.assertArrayEquals(expectedFeatureImportances, featureImportancesDefault, 1.0e-6)


      val expectedFeatureImportancesPrettified = expectedFeatureImportancesJson
        .values(s"calc_type_${calcType}_prettified")
        .asInstanceOf[List[Map[String,Any]]]

      val featureImportancesPredictionValuesChangePrettified = model.getFeatureImportancePrettified(
        fstrType=EFstrType.PredictionValuesChange,
        calcType=calcType
      )
      assertPrettifiedEquals(
        expectedFeatureImportancesPrettified,
        featureImportancesPredictionValuesChangePrettified
      )

      val featureImportancesDefaultPrettified = model.getFeatureImportancePrettified(calcType=calcType)
      assertPrettifiedEquals(expectedFeatureImportancesPrettified, featureImportancesDefaultPrettified)
    }
  }

  @Test
  @throws(classOf[Exception])
  def testLossFunctionChange() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("QueryRMSE")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = regressor.fit(trainPool)

    val expectedFeatureImportancesFile = canonicalDataDir.resolve("feature_importance_loss_function_change.json")
    val expectedFeatureImportancesJson = parse(
      Source.fromFile(expectedFeatureImportancesFile.toString).getLines.mkString
    ).asInstanceOf[JObject]

    for (calcType <- Seq(ECalcTypeShapValues.Regular, ECalcTypeShapValues.Approximate, ECalcTypeShapValues.Exact)) {
      val expectedFeatureImportances = expectedFeatureImportancesJson
        .values(s"calc_type_${calcType}")
        .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
        .toArray

      val featureImportancesLossFunctionChange = model.getFeatureImportance(
        fstrType=EFstrType.LossFunctionChange,
        data=trainPool,
        calcType=calcType
      )
      Assert.assertArrayEquals(expectedFeatureImportances, featureImportancesLossFunctionChange, 1.0e-6)

      val featureImportancesDefault = model.getFeatureImportance(data=trainPool, calcType=calcType)
      Assert.assertArrayEquals(expectedFeatureImportances, featureImportancesDefault, 1.0e-6)


      val expectedFeatureImportancesPrettified = expectedFeatureImportancesJson
        .values(s"calc_type_${calcType}_prettified")
        .asInstanceOf[List[Map[String,Any]]]

      val featureImportancesLossFunctionChangePrettified = model.getFeatureImportancePrettified(
        fstrType=EFstrType.LossFunctionChange,
        data=trainPool,
        calcType=calcType
      )
      assertPrettifiedEquals(
        expectedFeatureImportancesPrettified,
        featureImportancesLossFunctionChangePrettified
      )

      val featureImportancesDefaultPrettified = model.getFeatureImportancePrettified(
        data=trainPool,
        calcType=calcType
      )
      assertPrettifiedEquals(expectedFeatureImportancesPrettified, featureImportancesDefaultPrettified)
    }
  }

  @Test
  @throws(classOf[Exception])
  def testInteraction() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("QueryRMSE")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = regressor.fit(trainPool)

    val expectedFeatureImportancesFile = canonicalDataDir.resolve("feature_importance_interaction.json")
    val expectedFeatureImportances = parse(
      Source.fromFile(expectedFeatureImportancesFile.toString).getLines.mkString
    ).values.asInstanceOf[List[Map[String,Any]]]

    val expectedFeatureImportancesMap = new mutable.HashMap[(Int, Int), Double]()
    for (item <- expectedFeatureImportances) {
      expectedFeatureImportancesMap.put(
        (item("firstFeatureIndex").asInstanceOf[BigInt].toInt
         ->
         item("secondFeatureIndex").asInstanceOf[BigInt].toInt
        ),
        item("score").asInstanceOf[Double]
      )
    }

    val featureImportancesInteraction = model.getFeatureImportanceInteraction()

    val featureImportancesMap = new mutable.HashMap[(Int, Int), Double]()
    for (item <- featureImportancesInteraction) {
      featureImportancesMap.put((item.firstFeatureIdx -> item.secondFeatureIdx), item.score)
    }

    Assert.assertEquals(expectedFeatureImportancesMap.size, featureImportancesMap.size)

    for (((firstFeatureIdx, secondFeatureIdx), expectedScore) <- expectedFeatureImportancesMap) {
      featureImportancesMap.get(firstFeatureIdx -> secondFeatureIdx) match {
        case Some(score) => Assert.assertEquals(expectedScore, score, 1.0e-6)
        case None => Assert.fail(
          s"Features pair ($firstFeatureIdx, $secondFeatureIdx) is absent in featureImportancesInteraction"
        )
      }
    }
  }


  def testShapValuesCase[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]](
    problemType: String,
    model : CatBoostModelTrait[Model],
    canonicalDataDir: Path,
    data: Pool
  ) {
    val expectedFeaturesImportancesZipFile = canonicalDataDir.resolve("feature_importance_shap_values.json.zip")
    val tmpDir = temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath
    TestHelpers.unzip(expectedFeaturesImportancesZipFile, Paths.get(tmpDir))
    val expectedFeatureImportancesFile = Paths.get(tmpDir, "feature_importance_shap_values.json")
    val expectedFeatureImportances = parse(
      Source.fromFile(expectedFeatureImportancesFile.toString).getLines.mkString
    ).values.asInstanceOf[Map[String,Any]]

    val shapModes = Seq(
      EPreCalcShapValues.Auto,
      EPreCalcShapValues.UsePreCalc,
      EPreCalcShapValues.NoPreCalc
    )
    val calcTypes = Seq(
      ECalcTypeShapValues.Regular,
      ECalcTypeShapValues.Approximate,
      ECalcTypeShapValues.Exact
    )
    for (shapMode <- shapModes) {
      for (calcType <- calcTypes) {
        val resultName = s"problem_type=$problemType,shap_mode=$shapMode,shap_calc_type=$calcType"
        val expectedShapValues = expectedFeatureImportances(resultName)
          .asInstanceOf[scala.collection.immutable.$colon$colon[_]]

        val shapValuesDf = model.getFeatureImportanceShapValues(
          data=data,
          preCalcMode=shapMode,
          calcType=calcType
        )
        val shapValuesColumnIdx = shapValuesDf.schema.fieldIndex("shapValues")
        if (problemType.equals("MultiClass")) {
          for ((row, objectIdx) <- shapValuesDf.toLocalIterator.asScala.zipWithIndex) {
            val shapValuesForObject = row.getAs[Matrix](shapValuesColumnIdx)
            val expectedShapValuesForObject = expectedShapValues(objectIdx)
              .asInstanceOf[scala.collection.immutable.$colon$colon[_]]
            val classCount = shapValuesForObject.numRows
            Assert.assertEquals(classCount, expectedShapValuesForObject.size)
            for ((shapValuesForClass, classIdx) <- shapValuesForObject.rowIter.zipWithIndex) {
              val expectedShapValuesForClass = expectedShapValuesForObject(classIdx)
                .asInstanceOf[scala.collection.immutable.$colon$colon[Double]].toArray
              Assert.assertArrayEquals(expectedShapValuesForClass, shapValuesForClass.toArray, 1.0e-5)
            }
          }
        } else {
          for ((row, objectIdx) <- shapValuesDf.toLocalIterator.asScala.zipWithIndex) {
            val shapValuesForObject = row.getAs[Vector](shapValuesColumnIdx).toArray
            val expectedShapValuesForObject = expectedShapValues(objectIdx)
              .asInstanceOf[scala.collection.immutable.$colon$colon[Double]].toArray
            Assert.assertArrayEquals(expectedShapValuesForObject, shapValuesForObject, 1.0e-5)
          }
        }
      }
    }
  }

  @Test
  @throws(classOf[Exception])
  def testShapValuesForBinClass() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "higgs")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("Logloss")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = classifier.fit(trainPool)
    testShapValuesCase("BinClass", model, canonicalDataDir, trainPool)
  }

  @Test
  @throws(classOf[Exception])
  def testShapValuesForMultiClass() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "cloudness_small")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train_float.cd")
    )

//    val classifier = new CatBoostClassifier()
//      .setIterations(20)
//      .setLossFunction("MultiClass")
//      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
//
//    val model = classifier.fit(trainPool)
    val model = CatBoostClassificationModel.loadNativeModel(
      canonicalDataDir.resolve(s"feature_importance_shap_values.problem_type=MultiClass.cbm").toString()
    )

    testShapValuesCase("MultiClass", model, canonicalDataDir, trainPool)
  }

  @Test
  @throws(classOf[Exception])
  def testShapValuesForRegression() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("QueryRMSE")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = regressor.fit(trainPool)
    testShapValuesCase("Regression", model, canonicalDataDir, trainPool)
  }

  @Test
  @throws(classOf[Exception])
  def testPredictionDiff() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "higgs")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val dataForPredictionDiff = new Pool(trainPool.data.limit(2))

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = regressor.fit(trainPool)

    val expectedFeatureImportancesFile = canonicalDataDir.resolve("feature_importance_prediction_diff.json")
    val expectedFeatureImportancesJson = parse(
      Source.fromFile(expectedFeatureImportancesFile.toString).getLines.mkString
    ).asInstanceOf[JObject]

    val expectedFeatureImportances = expectedFeatureImportancesJson
      .values("simple")
      .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
      .toArray

    val featureImportances = model.getFeatureImportance(
      fstrType=EFstrType.PredictionDiff,
      data=dataForPredictionDiff
    )
    Assert.assertArrayEquals(expectedFeatureImportances, featureImportances, 1.0e-6)


    val expectedFeatureImportancesPrettified = expectedFeatureImportancesJson
      .values("prettified")
      .asInstanceOf[List[Map[String,Any]]]

    val featureImportancesPrettified = model.getFeatureImportancePrettified(
      fstrType=EFstrType.PredictionDiff,
      data=dataForPredictionDiff
    )
    assertPrettifiedEquals(expectedFeatureImportancesPrettified, featureImportancesPrettified)
  }

  def testShapInteractionValuesCase[Model <: org.apache.spark.ml.PredictionModel[Vector, Model]](
    problemType: String,
    model : CatBoostModelTrait[Model],
    canonicalDataDir: Path,
    data: Pool
  ) {
    val dataForFeatureImportance = new Pool(TestHelpers.addIndexColumn(data.data.limit(5)))

    val expectedFeaturesImportancesZipFile = canonicalDataDir.resolve("feature_importance_shap_interaction_values.json.zip")
    val tmpDir = temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath
    TestHelpers.unzip(expectedFeaturesImportancesZipFile, Paths.get(tmpDir))
    val expectedFeatureImportancesFile = Paths.get(tmpDir, "feature_importance_shap_interaction_values.json")

    val expectedFeatureImportances = parse(
      Source.fromFile(expectedFeatureImportancesFile.toString).getLines.mkString
    ).values.asInstanceOf[Map[String,Any]]

    val shapModes = Seq(
      EPreCalcShapValues.Auto,
      EPreCalcShapValues.UsePreCalc,
      EPreCalcShapValues.NoPreCalc
    )
    val calcTypes = Seq(
      ECalcTypeShapValues.Regular
    )
    for (shapMode <- shapModes) {
      for (calcType <- calcTypes) {
        val resultName = s"problem_type=$problemType,shap_mode=$shapMode,shap_calc_type=$calcType"
        val expectedShapInteractionValues = expectedFeatureImportances(resultName)
          .asInstanceOf[scala.collection.immutable.$colon$colon[_]]

        // binclass, regression: 'objectIdx,featureIdx1,featureIdx2' -> score
        // multiclass: 'objectIdx,classIdx,featureIdx1,featureIdx2' -> score
        val expectedShapInteractionValuesMap = new mutable.HashMap[String,Double]()

        for ((expectedValuesForObject, objectIdx) <- expectedShapInteractionValues.zipWithIndex) {
          val typedValuesForObject = expectedValuesForObject
            .asInstanceOf[scala.collection.immutable.$colon$colon[_]]
          if (problemType.equals("MultiClass")) {
            for ((expectedValuesForClass, classIdx) <- typedValuesForObject.zipWithIndex) {
              val typedValuesForClass = expectedValuesForClass
                .asInstanceOf[scala.collection.immutable.$colon$colon[_]]
              for ((expectedValuesForIdx1, featureIdx1) <- typedValuesForClass.zipWithIndex) {
                val typedValuesForIdx1 = expectedValuesForIdx1
                  .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
                for ((expectedValueForIdx2, featureIdx2) <- typedValuesForIdx1.zipWithIndex) {
                  val key = s"$objectIdx,$classIdx,$featureIdx1,$featureIdx2"
                  expectedShapInteractionValuesMap.update(key, expectedValueForIdx2)
                }
              }
            }
          } else {
            for ((expectedValuesForIdx1, featureIdx1) <- typedValuesForObject.zipWithIndex) {
              val typedValuesForIdx1 = expectedValuesForIdx1
                .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
              for ((expectedValueForIdx2, featureIdx2) <- typedValuesForIdx1.zipWithIndex) {
                val key = s"$objectIdx,$featureIdx1,$featureIdx2"
                expectedShapInteractionValuesMap.update(key, expectedValueForIdx2)
              }
            }
          }
        }

        val shapInteractionValuesDf = model.getFeatureImportanceShapInteractionValues(
          data=dataForFeatureImportance,
          preCalcMode=shapMode,
          calcType=calcType
        )

        for (row <- shapInteractionValuesDf.toLocalIterator.asScala) {
          val objectIdx = row.getAs[Long]("index")
          val featureIdx1 = row.getAs[Int]("featureIdx1")
          val featureIdx2 = row.getAs[Int]("featureIdx2")
          val value = row.getAs[Double]("shapInteractionValue")
          val key = if (problemType.equals("MultiClass")) {
            val classIdx = row.getAs[Int]("classIdx")
            s"$objectIdx,$classIdx,$featureIdx1,$featureIdx2"
          } else {
            s"$objectIdx,$featureIdx1,$featureIdx2"
          }
          Assert.assertEquals(expectedShapInteractionValuesMap(key), value, 1.0e-5)
        }
      }
    }
  }

  @Test
  @throws(classOf[Exception])
  def testShapInteractionValuesForBinClass() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "higgs")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("Logloss")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = classifier.fit(trainPool)
    testShapInteractionValuesCase("BinClass", model, canonicalDataDir, trainPool)
  }

  @Test
  @throws(classOf[Exception])
  def testShapInteractionValuesForMultiClass() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "cloudness_small")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train_float.cd")
    )

//    val classifier = new CatBoostClassifier()
//      .setIterations(20)
//      .setLossFunction("MultiClass")
//      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
//
//    val model = classifier.fit(trainPool)
    val model = CatBoostClassificationModel.loadNativeModel(
      canonicalDataDir.resolve(s"feature_importance_shap_interaction_values.problem_type=MultiClass.cbm").toString()
    )

    testShapInteractionValuesCase("MultiClass", model, canonicalDataDir, trainPool)
  }

  @Test
  @throws(classOf[Exception])
  def testShapInteractionValuesForRegression() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "higgs")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train_small").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setLossFunction("RMSE")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = regressor.fit(trainPool)
    testShapInteractionValuesCase("Regression", model, canonicalDataDir, trainPool)
  }
}
