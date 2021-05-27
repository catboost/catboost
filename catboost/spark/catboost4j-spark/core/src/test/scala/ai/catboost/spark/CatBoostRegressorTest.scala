package ai.catboost.spark;

import java.nio.file.{Paths}

import scala.io.Source

import collection.JavaConverters._
import collection.mutable

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.sql._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.spark.params._

import org.junit.{Assert,Test,Ignore,Rule}
import org.junit.rules.TemporaryFolder


class CatBoostRegressorTest {
  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder
  
  
  @Test 
  @throws(classOf[Exception])
  def testSimple1() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.29847920816267565,
      0.1573237146313061,
      0.06839466622281495,
      0.01833385394747814,
      -0.000701746646667933,
      0.29408114885353864
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
    
    // check apply on quantized
    val quantizedPool = pool.quantize()
    val quantizedPredictions = model.transformPool(quantizedPool)
    
    TestHelpers.assertEqualsWithPrecision(
      expectedPredictions.drop("features"),
      quantizedPredictions.drop("features")
    )
  }

  @Test
  @throws(classOf[Exception])
  def testSimpleOnDataFrame() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    val srcDataSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", FloatType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true
    )
    
    val srcData = Seq(
      Row(Vectors.dense(0.1, 0.2, 0.11), 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), 1.1f),
      Row(Vectors.dense(0.13, 0.22, 0.23), 2.1f),
      Row(Vectors.dense(0.14, 0.18, 0.1), 0.0f),
      Row(Vectors.dense(0.9, 0.67, 0.17), -1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), 0.62f)
    )
    
    val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))
    
    val expectedPrediction = Seq(
      0.05222253481760597,
      0.9310698268032307,
      1.6885733461810206,
      0.017027222605587908,
      -0.7782598974535129,
      0.533743544402863
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", FloatType),
        ("prediction", DoubleType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = regressor.fit(df)
    
    val predictions = model.transform(df)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }

  @Test 
  @throws(classOf[Exception])
  def testFeaturesRenamed() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("f1", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight", "features" -> "f1")
    )
    
    val expectedPrediction = Seq(
      0.29847920816267565,
      0.1573237146313061,
      0.06839466622281495,
      0.01833385394747814,
      -0.000701746646667933,
      0.29408114885353864
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setFeaturesCol("f1")
    val model = regressor.fit(pool).setFeaturesCol("f1")
    val predictions = model.transform(pool.data)
    
    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test
  @throws(classOf[Exception])
  def testWithEvalSet() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcTrainData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f)
    )
    val srcTestData = Seq(
      Row(Vectors.dense(0.0, 0.33, 1.1), "0.22",  0x4AAFFF4567657575L, 0xD34BFBD7, 0.1f),
      Row(Vectors.dense(0.02, 0.0, 0.38), "0.11", 0x686726738873ABCDL, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.86, 0.54, 0.9), "0.48", 0x7652786FF37ABBEDL, 0x19CE5B0A, 0.17f)
    )
    
    val trainPool = PoolTestHelpers.createRawPool(
      TestHelpers.getCurrentMethodName,
      PoolTestHelpers.createSchema(
        srcSchemaData,
        featureNames,
        /*addFeatureNamesMetadata*/ true
      ),
      srcTrainData,
      Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    val testPool = PoolTestHelpers.createRawPool(
      TestHelpers.getCurrentMethodName,
      PoolTestHelpers.createSchema(
        srcSchemaData,
        featureNames,
        /*addFeatureNamesMetadata*/ true
      ),
      srcTestData,
      Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.1422439696582127,
      0.08192678811528119,
      0.025701323195275082
    )
    val expectedPredictionsData = mutable.Seq.concat(srcTestData)
    for (i <- 0 until srcTestData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )
    
    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = regressor.fit(trainPool, Array[Pool](testPool))
    val predictions = model.transform(testPool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test 
  @throws(classOf[Exception])
  def testWithEvalSets() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcTrainData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f)
    )
    val srcTestDataSeq = Seq(
      Seq(
        Row(Vectors.dense(0.0, 0.33, 1.1), "0.22",  0x4AAFFF4567657575L, 0xD34BFBD7, 0.1f),
        Row(Vectors.dense(0.02, 0.0, 0.38), "0.11", 0x686726738873ABCDL, 0x23D794E9, 1.0f),
        Row(Vectors.dense(0.86, 0.54, 0.9), "0.48", 0x7652786FF37ABBEDL, 0x19CE5B0A, 0.17f)
      ),
      Seq(
        Row(Vectors.dense(0.12, 0.28, 2.2), "0.1",  0x4AAFADDE37657575L, 0xD34BFBD7, 0.11f),
        Row(Vectors.dense(0.0, 0.0, 0.92), "0.9", 0x686726738873ABCDL, 0x23D794E9, 1.1f),
        Row(Vectors.dense(0.13, 2.1, 0.45), "0.88", 0x686726738873ABCDL, 0x56A96DFA, 1.2f),
        Row(Vectors.dense(0.17, 0.11, 0.0), "0.0", 0xADD57787677BBA22L, 0x19CE5B0A, 1.0f)
      )
    )
    
    val trainPool = PoolTestHelpers.createRawPool(
      TestHelpers.getCurrentMethodName,
      PoolTestHelpers.createSchema(
        srcSchemaData,
        featureNames,
        /*addFeatureNamesMetadata*/ true
      ),
      srcTrainData,
      Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    val testPools = srcTestDataSeq.map(
      srcTestData => {
        PoolTestHelpers.createRawPool(
          TestHelpers.getCurrentMethodName,
          PoolTestHelpers.createSchema(
            srcSchemaData,
            featureNames,
            /*addFeatureNamesMetadata*/ true
          ),
          srcTestData,
          Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
        )
      }
    )
    val expectedPredictionSeq = Seq(
      Seq(
        0.1422439696582127,
        0.08192678811528119,
        0.025701323195275082
      ),
      Seq(
        0.16990201041826378,
        0.08192678811528119,
        0.12087124558369691,
        0.18270988329272156
      )
    )
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictionDfs = (srcTestDataSeq zip expectedPredictionSeq).map{
      case (srcTestData, expectedPrediction) => {
        val expectedPredictionsData = mutable.Seq.concat(srcTestData)
        for (i <- 0 until srcTestData.length) {
          expectedPredictionsData(i) = TestHelpers.appendToRow(
            expectedPredictionsData(i),
            expectedPrediction(i)
          )
        }
        spark.createDataFrame(
          spark.sparkContext.parallelize(expectedPredictionsData),
          StructType(expectedPredictionsSchema)
        )
      }
    }

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = regressor.fit(trainPool, Array[Pool](testPools(0), testPools(1)))
    val predictionsSeq = testPools.map(testPool => model.transform(testPool.data))

    (predictionsSeq zip expectedPredictionDfs).map{
      case (predictions, expectedPredictionsDf) => {
        TestHelpers.assertEqualsWithPrecision(expectedPredictionsDf, predictions)
      }
    }
  }
  
  @Test 
  @throws(classOf[Exception])
  def testDurationParam() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.29847920816267565,
      0.1573237146313061,
      0.06839466622281495,
      0.01833385394747814,
      -0.000701746646667933,
      0.29408114885353864
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )
    
    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setSnapshotInterval(java.time.Duration.ofHours(1))
    
    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }

  @Test 
  @throws(classOf[Exception])
  def testOverfittingDetectorIncToDec() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd")
    )
    val evalPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("test").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val expectedPredictionsFile = canonicalDataDir.resolve("regression_overfitting_detector.json")

    val expectedPredictionsJson = parse(Source.fromFile(expectedPredictionsFile.toString).getLines.mkString)
    val expectedPrediction = expectedPredictionsJson
      .asInstanceOf[JObject].values("prediction_IncToDec")
      .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
      .toSeq
      
    val expectedPredictionsData = mutable.Seq.concat(evalPool.data.toLocalIterator.asScala.toTraversable)
    for (i <- 0 until expectedPredictionsData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      evalPool.data.schema.fields.map(f => (f.name, f.dataType)) :+ ("prediction", DoubleType),
      evalPool.getFeatureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ evalPool.data.schema.fieldNames :+ ("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    {
      val regressor = new CatBoostRegressor()
        .setIterations(200)
        .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
        .setOdPval(1.0e-2f)
      val model = regressor.fit(trainPool, Array[Pool](evalPool))
      val predictions = model.transform(evalPool.data)
  
      TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
    }
    {
      val regressor = new CatBoostRegressor()
        .setIterations(200)
        .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName + "2").getPath)
        .setOdType(EOverfittingDetectorType.IncToDec)
        .setOdPval(1.0e-2f)
      val model = regressor.fit(trainPool, Array[Pool](evalPool))
      val predictions = model.transform(evalPool.data)
  
      TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
    }
  }

  @Test 
  @throws(classOf[Exception])
  def testOverfittingDetectorIter() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd")
    )
    val evalPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("test").toString,
      columnDescription = dataDir.resolve("train.cd")
    )

    val expectedPredictionsFile = canonicalDataDir.resolve("regression_overfitting_detector.json")

    val expectedPredictionsJson = parse(Source.fromFile(expectedPredictionsFile.toString).getLines.mkString)
    val expectedPrediction = expectedPredictionsJson
      .asInstanceOf[JObject].values("prediction_Iter")
      .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
      .toSeq
      
    val expectedPredictionsData = mutable.Seq.concat(evalPool.data.toLocalIterator.asScala.toTraversable)
    for (i <- 0 until expectedPredictionsData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      evalPool.data.schema.fields.map(f => (f.name, f.dataType)) :+ ("prediction", DoubleType),
      evalPool.getFeatureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ evalPool.data.schema.fieldNames :+ ("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    {
      val regressor = new CatBoostRegressor()
        .setIterations(200)
        .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
        .setEarlyStoppingRounds(20)
      val model = regressor.fit(trainPool, Array[Pool](evalPool))
      val predictions = model.transform(evalPool.data)
  
      TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
    }
    {
      val regressor = new CatBoostRegressor()
        .setIterations(200)
        .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName + "2").getPath)
        .setOdType(EOverfittingDetectorType.Iter)
      val model = regressor.fit(trainPool, Array[Pool](evalPool))
      val predictions = model.transform(evalPool.data)
  
      TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
    }
  }

  @Test 
  @throws(classOf[Exception])
  def testParams() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.15170673766243029,
      0.11622931026015675,
      0.03726535148720679,
      0.010579656432736211,
      0.0037732481141320336,
      0.15237303004835703
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )
    
    val firstFeatureUsePenaltiesMap = new java.util.LinkedHashMap[String, Double]
    firstFeatureUsePenaltiesMap.put("f1", 0.0)
    firstFeatureUsePenaltiesMap.put("f2", 1.1)
    firstFeatureUsePenaltiesMap.put("f3", 2.0)
   
    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setLeafEstimationIterations(10)
      .setFirstFeatureUsePenaltiesMap(firstFeatureUsePenaltiesMap)
      .setFeatureWeightsList(Array[Double](1.0, 2.0, 3.0))
      
    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }


  @Test 
  @throws(classOf[Exception])
  def testOneHotCatFeatures() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("c1", "c2", "c3")
    val catFeaturesNumValues = Map("c1" -> 2, "c2" -> 4, "c3" -> 6)

    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0, 0, 0), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(1, 1, 0), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0, 2, 1), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(1, 2, 2), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0, 0, 3), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0, 0, 4), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f),
      Row(Vectors.dense(1, 3, 5), "0.1", 0xEFFAAEA875588873L, 0xD34BFBD7, 1.0f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true,
          catFeaturesNumValues = catFeaturesNumValues
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.3094933770071123,
      0.06869861198568002,
      0.16072009692696285,
      0.014205016537063388,
      0.006119254974129929,
      0.3221460373277655,
      0.08326180420157153
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction"),
      catFeaturesNumValues = catFeaturesNumValues
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setOneHotMaxSize(6)
      .setHasTime(true)
    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test 
  @throws(classOf[Exception])
  def testNumAndOneHotCatFeatures() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3", "f4", "c1", "c2", "c3")
    val catFeaturesNumValues = Map("c1" -> 2, "c2" -> 4, "c3" -> 6)

    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23, 0.72, 0, 0, 0), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11, -0.7, 1, 1, 0), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33, 0.18, 0, 2, 1), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17, 0.0, 1, 2, 2), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31, -0.12, 0, 0, 3), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1, 0.0, 0, 0, 4), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f),
      Row(Vectors.dense(1.0, 0.88, 0.21, 0.0, 1, 3, 5), "0.1", 0xEFFAAEA875588873L, 0xD34BFBD7, 1.0f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true,
          catFeaturesNumValues = catFeaturesNumValues
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.2997162131753886,
      0.0647881241157201,
      0.07863414627300913,
      0.014004785813835005,
      -0.00018176854774739688,
      0.2905242445716116,
      0.08442773926280388
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction"),
      catFeaturesNumValues = catFeaturesNumValues
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setOneHotMaxSize(6)
      .setHasTime(true)
    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test 
  @throws(classOf[Exception])
  def testNumAndOneHotCatFeaturesWithEvalSets() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3", "f4", "c1", "c2", "c3")
    val catFeaturesNumValues = Map("c1" -> 2, "c2" -> 4, "c3" -> 6)

    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcTrainData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23, 0.72, 0, 0, 0), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11, -0.7, 1, 1, 0), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33, 0.18, 0, 2, 1), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17, 0.0, 1, 2, 2), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31, -0.12, 0, 0, 3), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1, 0.0, 0, 0, 4), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f),
      Row(Vectors.dense(1.0, 0.88, 0.21, 0.0, 1, 3, 5), "0.1", 0xEFFAAEA875588873L, 0xD34BFBD7, 1.0f)
    )
    val srcTestDataSeq = Seq(
      Seq(
        Row(Vectors.dense(0.0, 0.33, 1.1, 0.01, 0, 1, 2), "0.22",  0x4AAFFF4567657575L, 0xD34BFBD7, 0.1f),
        Row(Vectors.dense(0.02, 0.0, 0.38, -0.3, 1, 2, 3), "0.11", 0x686726738873ABCDL, 0x23D794E9, 1.0f),
        Row(Vectors.dense(0.86, 0.54, 0.9, 0.0, 0, 2, 5), "0.48", 0x686726738873ABCDL, 0x19CE5B0A, 0.17f)
      ),
      Seq(
        Row(Vectors.dense(0.12, 0.28, 2.2, -0.12, 1, 3, 3), "0.1", 0x5628779FFABBAA67L, 0xD34BFBD7, 0.11f),
        Row(Vectors.dense(0.0, 0.0, 0.92, 0.0, 0, 3, 4), "0.9", 0x5628779FFABBAA67L, 0x23D794E9, 1.1f),
        Row(Vectors.dense(0.13, 2.1, 0.45, 1.0, 1, 2, 5), "0.88", 0x5628779FFABBAA67L, 0x56A96DFA, 1.2f),
        Row(Vectors.dense(0.17, 0.11, 0.0, 2.11, 1, 0, 2), "0.0", 0x90ABBD784AA812BAL, 0x19CE5B0A, 1.0f)
      )
    )

    val trainPool = PoolTestHelpers.createRawPool(
      TestHelpers.getCurrentMethodName,
      PoolTestHelpers.createSchema(
        srcSchemaData,
        featureNames,
        /*addFeatureNamesMetadata*/ true,
        catFeaturesNumValues = catFeaturesNumValues
      ),
      srcTrainData,
      Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    val testPools = srcTestDataSeq.map(
      srcTestData => {
        PoolTestHelpers.createRawPool(
          TestHelpers.getCurrentMethodName,
          PoolTestHelpers.createSchema(
            srcSchemaData,
            featureNames,
            /*addFeatureNamesMetadata*/ true,
            catFeaturesNumValues = catFeaturesNumValues
          ),
          srcTestData,
          Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
        )
      }
    )
    val expectedPredictionSeq = Seq(
      Seq(
        0.15662421960551953,
        0.07005287608735944,
        0.04151475093846452
      ),
      Seq(
        0.07077519846934306,
        0.15230223583519026,
        0.08603755562520628,
        0.11427156183786472
      )
    )
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction"),
      catFeaturesNumValues = catFeaturesNumValues
    )
    val expectedPredictionDfs = (srcTestDataSeq zip expectedPredictionSeq).map{
      case (srcTestData, expectedPrediction) => {
        val expectedPredictionsData = mutable.Seq.concat(srcTestData)
        for (i <- 0 until srcTestData.length) {
          expectedPredictionsData(i) = TestHelpers.appendToRow(
            expectedPredictionsData(i),
            expectedPrediction(i)
          )
        }
        spark.createDataFrame(
          spark.sparkContext.parallelize(expectedPredictionsData),
          StructType(expectedPredictionsSchema)
        )
      }
    }

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setOneHotMaxSize(6)
      .setHasTime(true)
    val model = regressor.fit(trainPool, Array[Pool](testPools(0), testPools(1)))
    val predictionsSeq = testPools.map(testPool => model.transform(testPool.data))

    (predictionsSeq zip expectedPredictionDfs).map{
      case (predictions, expectedPredictionsDf) => {
        TestHelpers.assertEqualsWithPrecision(expectedPredictionsDf, predictions)
      }
    }
  }
  
  @Test 
  @throws(classOf[Exception])
  def testOneHotAndCtrCatFeatures() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("c1", "c2", "c3")
    val catFeaturesNumValues = Map("c1" -> 2, "c2" -> 4, "c3" -> 6)

    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0, 0, 0), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(1, 1, 0), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0, 2, 1), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(1, 2, 2), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0, 0, 3), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0, 0, 4), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f),
      Row(Vectors.dense(1, 3, 5), "0.1", 0xEFFAAEA875588873L, 0xD34BFBD7, 1.0f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true,
          catFeaturesNumValues = catFeaturesNumValues
        ),
        srcData,
        Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    
    val expectedPrediction = Seq(
      0.3094933770071123,
      0.06869861198568002,
      0.16072009692696285,
      0.014205016537063388,
      0.006119254974129929,
      0.3221460373277655,
      0.08326180420157153
    )
    val expectedPredictionsData = mutable.Seq.concat(srcData)
    for (i <- 0 until srcData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction"),
      catFeaturesNumValues = catFeaturesNumValues
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setHasTime(true)
    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test 
  @throws(classOf[Exception])
  def testNumAndOneHotAndCtrCatFeaturesWithEvalSets() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3", "f4", "c1", "c2", "c3")
    val catFeaturesNumValues = Map("c1" -> 2, "c2" -> 4, "c3" -> 6)

    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcTrainData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23, 0.72, 0, 0, 0), "0.34", 0x86F1B93B695F9E61L, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11, -0.7, 1, 1, 0), "0.12", 0xB337C6FEFE2E2F73L, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33, 0.18, 0, 2, 1), "0.22", 0xB337C6FEFE2E2F73L, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17, 0.0, 1, 2, 2), "0.01", 0xD9DBDD3199D6518AL, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31, -0.12, 0, 0, 3), "0.0", 0xD9DBDD3199D6518AL, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1, 0.0, 0, 0, 4), "0.42", 0xD9DBDD3199D6518AL, 0x62772D1C, 0.45f),
      Row(Vectors.dense(1.0, 0.88, 0.21, 0.0, 1, 3, 5), "0.1", 0xEFFAAEA875588873L, 0xD34BFBD7, 1.0f)
    )
    val srcTestDataSeq = Seq(
      Seq(
        Row(Vectors.dense(0.0, 0.33, 1.1, 0.01, 0, 1, 2), "0.22",  0x4AAFFF4567657575L, 0xD34BFBD7, 0.1f),
        Row(Vectors.dense(0.02, 0.0, 0.38, -0.3, 1, 2, 3), "0.11", 0x686726738873ABCDL, 0x23D794E9, 1.0f),
        Row(Vectors.dense(0.86, 0.54, 0.9, 0.0, 0, 2, 5), "0.48", 0x686726738873ABCDL, 0x19CE5B0A, 0.17f)
      ),
      Seq(
        Row(Vectors.dense(0.12, 0.28, 2.2, -0.12, 1, 3, 3), "0.1", 0x5628779FFABBAA67L, 0xD34BFBD7, 0.11f),
        Row(Vectors.dense(0.0, 0.0, 0.92, 0.0, 0, 3, 4), "0.9", 0x5628779FFABBAA67L, 0x23D794E9, 1.1f),
        Row(Vectors.dense(0.13, 2.1, 0.45, 1.0, 1, 2, 5), "0.88", 0x5628779FFABBAA67L, 0x56A96DFA, 1.2f),
        Row(Vectors.dense(0.17, 0.11, 0.0, 2.11, 1, 0, 2), "0.0", 0x90ABBD784AA812BAL, 0x19CE5B0A, 1.0f)
      )
    )

    val trainPool = PoolTestHelpers.createRawPool(
      TestHelpers.getCurrentMethodName,
      PoolTestHelpers.createSchema(
        srcSchemaData,
        featureNames,
        /*addFeatureNamesMetadata*/ true,
        catFeaturesNumValues = catFeaturesNumValues
      ),
      srcTrainData,
      Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )
    val testPools = srcTestDataSeq.map(
      srcTestData => {
        PoolTestHelpers.createRawPool(
          TestHelpers.getCurrentMethodName,
          PoolTestHelpers.createSchema(
            srcSchemaData,
            featureNames,
            /*addFeatureNamesMetadata*/ true,
            catFeaturesNumValues = catFeaturesNumValues
          ),
          srcTestData,
          Map("groupId" -> "groupId", "subgroupId" -> "subgroupId", "weight" -> "weight")
        )
      }
    )
    val expectedPredictionSeq = Seq(
      Seq(
        0.15662421960551953,
        0.07005287608735944,
        0.04151475093846452
      ),
      Seq(
        0.07077519846934306,
        0.15230223583519026,
        0.08603755562520628,
        0.11427156183786472
      )
    )
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      srcSchemaData :+ ("prediction", DoubleType),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("prediction"),
      catFeaturesNumValues = catFeaturesNumValues
    )
    val expectedPredictionDfs = (srcTestDataSeq zip expectedPredictionSeq).map{
      case (srcTestData, expectedPrediction) => {
        val expectedPredictionsData = mutable.Seq.concat(srcTestData)
        for (i <- 0 until srcTestData.length) {
          expectedPredictionsData(i) = TestHelpers.appendToRow(
            expectedPredictionsData(i),
            expectedPrediction(i)
          )
        }
        spark.createDataFrame(
          spark.sparkContext.parallelize(expectedPredictionsData),
          StructType(expectedPredictionsSchema)
        )
      }
    }

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setHasTime(true)
    val model = regressor.fit(trainPool, Array[Pool](testPools(0), testPools(1)))
    val predictionsSeq = testPools.map(testPool => model.transform(testPool.data))
  
    (predictionsSeq zip expectedPredictionDfs).map{
      case (predictions, expectedPredictionsDf) => {
        TestHelpers.assertEqualsWithPrecision(expectedPredictionsDf, predictions)
      }
    }
  }


  @Test 
  @throws(classOf[Exception])
  def testWithPairs() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))
    
    val pool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd"),
      pairsDataPathWithScheme = "dsv-grouped://" + dataDir.resolve("train.grouped_pairs").toString
    )

    val expectedPredictionsFile = canonicalDataDir.resolve("regression_with_pairs.json")

    val expectedPredictionsJson = parse(Source.fromFile(expectedPredictionsFile.toString).getLines.mkString)
    val expectedPrediction = expectedPredictionsJson
      .asInstanceOf[JObject].values("prediction")
      .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
      .toSeq
      
    val expectedPredictionsData = mutable.Seq.concat(pool.data.toLocalIterator.asScala.toTraversable)
    for (i <- 0 until expectedPredictionsData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      pool.data.schema.fields.map(f => (f.name, f.dataType)) :+ ("prediction", DoubleType),
      pool.getFeatureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ pool.data.schema.fieldNames :+ ("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setLossFunction("PairLogit")
      .setHasTime(true)

    val model = regressor.fit(pool)
    val predictions = model.transform(pool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions, Seq("groupId", "sampleId"))
  }

  @Test
  @throws(classOf[Exception])
  def testWithPairsWithEvalSet() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val dataDir = Paths.get(System.getProperty("catboost.test.data.path"), "querywise")
    val canonicalDataDir = Paths.get(System.getProperty("canonical.data.path"))

    val trainPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("train").toString,
      columnDescription = dataDir.resolve("train.cd"),
      pairsDataPathWithScheme = "dsv-grouped://" + dataDir.resolve("train.grouped_pairs").toString
    )
    val evalPool = Pool.load(
      spark,
      dataPathWithScheme = dataDir.resolve("test").toString,
      columnDescription = dataDir.resolve("train.cd"),
      pairsDataPathWithScheme = "dsv-grouped://" + dataDir.resolve("test.grouped_pairs").toString
    )

    val expectedPredictionsFile = canonicalDataDir.resolve("regression_with_pairs_with_eval_set.json")

    val expectedPredictionsJson = parse(Source.fromFile(expectedPredictionsFile.toString).getLines.mkString)
    val expectedPrediction = expectedPredictionsJson
      .asInstanceOf[JObject].values("prediction")
      .asInstanceOf[scala.collection.immutable.$colon$colon[Double]]
      .toSeq
      
    val expectedPredictionsData = mutable.Seq.concat(evalPool.data.toLocalIterator.asScala.toTraversable)
    for (i <- 0 until expectedPredictionsData.length) {
      expectedPredictionsData(i) = TestHelpers.appendToRow(
        expectedPredictionsData(i),
        expectedPrediction(i)
      )
    }
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      evalPool.data.schema.fields.map(f => (f.name, f.dataType)) :+ ("prediction", DoubleType),
      evalPool.getFeatureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ evalPool.data.schema.fieldNames :+ ("prediction")
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    val regressor = new CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
      .setLossFunction("PairLogit")
      .setHasTime(true)

    val model = regressor.fit(trainPool, Array[Pool](evalPool))
    val predictions = model.transform(evalPool.data)

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions, Seq("groupId", "sampleId"))
  }
}
