package ai.catboost.spark;

import collection.JavaConverters._
import collection.mutable

import org.apache.spark.sql._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._

import ai.catboost.CatBoostError
import ai.catboost.spark.params._

import org.junit.{Assert,Test,Ignore,Rule}
import org.junit.rules.TemporaryFolder


class CatBoostClassifierTest {
  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder


  @Test
  @throws(classOf[Exception])
  def testBinaryClassificationSimpleOnDataFrame() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)

    val featureNames = Array[String]("f1", "f2", "f3")
    val srcDataSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", IntegerType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true
    )
    
    val srcData = Seq(
      Row(Vectors.dense(0.1, 0.2, 0.11), 1),
      Row(Vectors.dense(0.97, 0.82, 0.33), 2),
      Row(Vectors.dense(0.13, 0.22, 0.23), 2),
      Row(Vectors.dense(0.14, 0.18, 0.1), 1),
      Row(Vectors.dense(0.9, 0.67, 0.17), 2),
      Row(Vectors.dense(0.66, 0.1, 0.31), 1)
    )
    
    val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(df)
    val predictions = model.transform(df)
    
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", IntegerType),
        ("rawPrediction", SQLDataTypes.VectorType),
        ("probability", SQLDataTypes.VectorType),
        ("prediction", DoubleType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
    )
    
    val expectedPredictionsData = Seq(
      Row(
        Vectors.dense(0.1, 0.2, 0.11), 
        1, 
        Vectors.dense(0.08414989363659559, -0.08414989363659559),
        Vectors.dense(0.541975913549805, 0.458024086450195),
        0.0
      ),
      Row(
        Vectors.dense(0.97, 0.82, 0.33), 
        2,
        Vectors.dense(-0.07660239597875373,0.07660239597875373), 
        Vectors.dense(0.4617735427982884,0.5382264572017116),
        1.0
      ),
      Row(
        Vectors.dense(0.13, 0.22, 0.23), 
        2,
        Vectors.dense(-0.07657474373810148,0.07657474373810148), 
        Vectors.dense(0.4617872881333314,0.5382127118666686),
        1.0
      ),
      Row(
        Vectors.dense(0.14, 0.18, 0.1),
        1,
        Vectors.dense(0.09721485072232189,-0.09721485072232189), 
        Vectors.dense(0.5484548768406571,0.4515451231593429),
        0.0
      ),
      Row(
        Vectors.dense(0.9, 0.67, 0.17),
        2,
        Vectors.dense(-0.08702243949954704,0.08702243949954704), 
        Vectors.dense(0.4565982840017737,0.5434017159982263),
        1.0
      ),
      Row(
        Vectors.dense(0.66, 0.1, 0.31),
        1, 
        Vectors.dense(0.07883731282470079,-0.07883731282470079), 
        Vectors.dense(0.539337192390336,0.460662807609664),
        0.0
      )
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test   
  @throws(classOf[Exception])
  def testSimpleBinaryClassification() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("groupWeight", FloatType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.1, 0.2, 0.11), "0", 0xB337C6FEFE2E2F73L, 1.0f, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0", 0xB337C6FEFE2E2F73L, 1.0f, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.13, 0.22, 0.23), "1", 0x86F1B93B695F9E61L, 0.0f, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "1", 0xD9DBDD3199D6518AL, 0.5f, 0x62772D1C, 0.45f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0", 0xD9DBDD3199D6518AL, 0.5f, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "1", 0xD9DBDD3199D6518AL, 0.5f, 0x1FA606FD, 2.0f)
    )

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        srcData,
        Map("groupId" -> "groupId", "groupWeight" -> "groupWeight", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
    val quantizedPool = pool.quantize()
    
    val expectedRawPrediction = Seq(
      (
        0.024000747348794246,
        -0.024000747348794246
      ),
      (
        0.0723957751868566,
        -0.0723957751868566
      ),
      (
        -0.0021655237893338777,
        0.0021655237893338777
      ),
      (
        -0.08466020497053796,
        0.08466020497053796
      ),
      (
        0.1018104760949401,
        -0.1018104760949401
      ),
      (
        -0.16282620575881535,
        0.16282620575881535
      )
    )
    val expectedProbability = Seq(
      (
        0.5119980699899546,
        0.48800193001004544
      ),
      (
        0.5361347803932197,
        0.46386521960678023
      ),
      (
        0.49891723979786473,
        0.5010827602021353
      ),
      (
        0.4577707399729678,
        0.5422292600270322
      ),
      (
        0.5507300803143427,
        0.4492699196856574
      ),
      (
        0.41929883288779035,
        0.5807011671122096
      )
    )
    val expectedPrediction = Seq(0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    
    for (rawPrediction <- Seq(false, true)) {
      for (probability <- Seq(false, true)) {
        for (prediction <- Seq(false, true)) {
          model.setRawPredictionCol(if (rawPrediction) "rawPrediction" else "")
          model.setProbabilityCol(if (probability) "probability" else "")
          model.setPredictionCol(if (prediction) "prediction" else "")
          val predictions = model.transform(pool.data)
          
          val expectedPredictionsData = mutable.Seq.concat(srcData)
          var expectedPredictionsSchemaData = srcSchemaData
          if (rawPrediction) {
            expectedPredictionsSchemaData 
              = expectedPredictionsSchemaData :+ ("rawPrediction", SQLDataTypes.VectorType)
            for (i <- 0 until srcData.length) {
              expectedPredictionsData(i) = TestHelpers.appendToRow(
                expectedPredictionsData(i),
                Vectors.dense(expectedRawPrediction(i)._1, expectedRawPrediction(i)._2)
              )
            }
          }
          if (probability) {
            expectedPredictionsSchemaData 
              = expectedPredictionsSchemaData :+ ("probability", SQLDataTypes.VectorType)
            for (i <- 0 until srcData.length) {
              expectedPredictionsData(i) = TestHelpers.appendToRow(
                expectedPredictionsData(i),
                Vectors.dense(expectedProbability(i)._1, expectedProbability(i)._2)
              )
            }
          }
          if (prediction) {
            expectedPredictionsSchemaData 
              = expectedPredictionsSchemaData :+ ("prediction", DoubleType)
            for (i <- 0 until srcData.length) {
              expectedPredictionsData(i) = TestHelpers.appendToRow(
                expectedPredictionsData(i),
                expectedPrediction(i)
              )
            }
          }
          val expectedPredictionsSchema = PoolTestHelpers.createSchema(
            expectedPredictionsSchemaData,
            featureNames,
            /*addFeatureNamesMetadata*/ true,
            /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
          )
          val expectedPredictions = spark.createDataFrame(
            spark.sparkContext.parallelize(expectedPredictionsData),
            StructType(expectedPredictionsSchema)
          )
          
          TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
          
          // check apply on quantized
          val quantizedPredictions = model.transformPool(quantizedPool)
          
          TestHelpers.assertEqualsWithPrecision(
            expectedPredictions.drop("features"),
            quantizedPredictions.drop("features")
          )
        }
      }
    }
  }
  
  // Master: String target type is not currently supported
  @Test(expected = classOf[RuntimeException])
  @throws(classOf[Exception])
  def testBinaryClassificationWithClassNamesExtraction() {
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), "good"),
          Row(Vectors.dense(0.97, 0.82, 0.33), "bad"),
          Row(Vectors.dense(0.13, 0.22, 0.23), "good"),
          Row(Vectors.dense(0.14, 0.18, 0.1), "bad"),
          Row(Vectors.dense(0.9, 0.67, 0.17), "good"),
          Row(Vectors.dense(0.66, 0.1, 0.31), "bad")
        ),
        Map[String,String]()
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
    val predictions = model.transform(pool.data)
  }
  
  // Master: String target type is not currently supported
  @Test(expected = classOf[RuntimeException])
  @throws(classOf[Exception])
  def testBinaryClassificationWithClassNamesSet() {
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), "good"),
          Row(Vectors.dense(0.97, 0.82, 0.33), "bad"),
          Row(Vectors.dense(0.13, 0.22, 0.23), "good"),
          Row(Vectors.dense(0.14, 0.18, 0.1), "bad"),
          Row(Vectors.dense(0.9, 0.67, 0.17), "good"),
          Row(Vectors.dense(0.66, 0.1, 0.31), "bad")
        ),
        Map[String,String]()
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setClassNames(Array[String]("bad", "good"))
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
    val predictions = model.transform(pool.data)
  }
  
  @Test
  @throws(classOf[Exception])
  def testBinaryClassificationWithClassNamesAsIntSet() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), "1"),
          Row(Vectors.dense(0.97, 0.82, 0.33), "2"),
          Row(Vectors.dense(0.13, 0.22, 0.23), "2"),
          Row(Vectors.dense(0.14, 0.18, 0.1), "1"),
          Row(Vectors.dense(0.9, 0.67, 0.17), "2"),
          Row(Vectors.dense(0.66, 0.1, 0.31), "1")
        ),
        Map[String,String]()
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setClassNames(Array[String]("1", "2"))
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
    val predictions = model.transform(pool.data)
    
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", StringType),
        ("rawPrediction", SQLDataTypes.VectorType),
        ("probability", SQLDataTypes.VectorType),
        ("prediction", DoubleType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
    )
    
    val expectedPredictionsData = Seq(
      Row(
        Vectors.dense(0.1, 0.2, 0.11), 
        "1", 
        Vectors.dense(0.08414989363659559, -0.08414989363659559),
        Vectors.dense(0.541975913549805, 0.458024086450195),
        0.0
      ),
      Row(
        Vectors.dense(0.97, 0.82, 0.33), 
        "2",
        Vectors.dense(-0.07660239597875373,0.07660239597875373), 
        Vectors.dense(0.4617735427982884,0.5382264572017116),
        1.0
      ),
      Row(
        Vectors.dense(0.13, 0.22, 0.23), 
        "2",
        Vectors.dense(-0.07657474373810148,0.07657474373810148), 
        Vectors.dense(0.4617872881333314,0.5382127118666686),
        1.0
      ),
      Row(
        Vectors.dense(0.14, 0.18, 0.1),
        "1",
        Vectors.dense(0.09721485072232189,-0.09721485072232189), 
        Vectors.dense(0.5484548768406571,0.4515451231593429),
        0.0
      ),
      Row(
        Vectors.dense(0.9, 0.67, 0.17),
        "2",
        Vectors.dense(-0.08702243949954704,0.08702243949954704), 
        Vectors.dense(0.4565982840017737,0.5434017159982263),
        1.0
      ),
      Row(
        Vectors.dense(0.66, 0.1, 0.31),
        "1", 
        Vectors.dense(0.07883731282470079,-0.07883731282470079), 
        Vectors.dense(0.539337192390336,0.460662807609664),
        0.0
      )
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test   
  @throws(classOf[Exception])
  def testBinaryClassificationWithTargetBorder() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), 0.12),
          Row(Vectors.dense(0.97, 0.82, 0.33), 0.1),
          Row(Vectors.dense(0.13, 0.22, 0.23), 0.7),
          Row(Vectors.dense(0.14, 0.18, 0.1), 0.33),
          Row(Vectors.dense(0.9, 0.67, 0.17), 0.82),
          Row(Vectors.dense(0.66, 0.1, 0.31), 0.93)
        ),
        Map[String,String]()
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setTargetBorder(0.5f)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
    val predictions = model.transform(pool.data)
    
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", DoubleType),
        ("rawPrediction", SQLDataTypes.VectorType),
        ("probability", SQLDataTypes.VectorType),
        ("prediction", DoubleType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
    )
    
    val expectedPredictionsData = Seq(
      Row(
        Vectors.dense(0.1, 0.2, 0.11), 
        0.12, 
        Vectors.dense(0.08057222609664604,-0.08057222609664604),
        Vectors.dense(0.5401991612479529,0.45980083875204714),
        0.0
      ),
      Row(
        Vectors.dense(0.97, 0.82, 0.33), 
        0.1,
        Vectors.dense(0.04555562514432977,-0.04555562514432977), 
        Vectors.dense(0.5227620685962877,0.4772379314037123),
        0.0
      ),
      Row(
        Vectors.dense(0.13, 0.22, 0.23), 
        0.7,
        Vectors.dense(-0.0799281861623364,0.0799281861623364), 
        Vectors.dense(0.4601207937404179,0.5398792062595821),
        1.0
      ),
      Row(
        Vectors.dense(0.14, 0.18, 0.1),
        0.33,
        Vectors.dense(0.08057222609664604,-0.08057222609664604), 
        Vectors.dense(0.5401991612479529,0.45980083875204714),
        0.0
      ),
      Row(
        Vectors.dense(0.9, 0.67, 0.17),
        0.82,
        Vectors.dense(-0.07938095256503758,0.07938095256503758), 
        Vectors.dense(0.46039268179178616,0.5396073182082138),
        1.0
      ),
      Row(
        Vectors.dense(0.66, 0.1, 0.31),
        0.93, 
        Vectors.dense(-0.07118906575434053,0.07118906575434053), 
        Vectors.dense(0.4644654751240226,0.5355345248759774),
        1.0
      )
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test(expected = classOf[CatBoostError]) 
  @throws(classOf[Exception])
  def testBinaryClassificationWithRealTargetWithoutTargetBorder() {
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), 0.12),
          Row(Vectors.dense(0.97, 0.82, 0.33), 0.1),
          Row(Vectors.dense(0.13, 0.22, 0.23), 0.7),
          Row(Vectors.dense(0.14, 0.18, 0.1), 0.33),
          Row(Vectors.dense(0.9, 0.67, 0.17), 0.82),
          Row(Vectors.dense(0.66, 0.1, 0.31), 0.93)
        ),
        Map[String,String]()
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setLossFunction("Logloss")
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
  }
  
  @Test   
  @throws(classOf[Exception])
  def testBinaryClassificationWithClassWeightsMap() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", IntegerType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), 0),
          Row(Vectors.dense(0.97, 0.82, 0.33), 1),
          Row(Vectors.dense(0.13, 0.22, 0.23), 1),
          Row(Vectors.dense(0.14, 0.18, 0.1), 0),
          Row(Vectors.dense(0.9, 0.67, 0.17), 0),
          Row(Vectors.dense(0.66, 0.1, 0.31), 0)
        ),
        Map[String,String]()
    )
    
    val classWeightsMap = new java.util.LinkedHashMap[String, Double]
    classWeightsMap.put("0", 1.0)
    classWeightsMap.put("1", 2.0)

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setClassWeightsMap(classWeightsMap)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)
    val model = classifier.fit(pool)
    val predictions = model.transform(pool.data)
    
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", IntegerType),
        ("rawPrediction", SQLDataTypes.VectorType),
        ("probability", SQLDataTypes.VectorType),
        ("prediction", DoubleType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
    )
    
    val expectedPredictionsData = Seq(
      Row(
        Vectors.dense(0.1, 0.2, 0.11), 
        0, 
        Vectors.dense(0.061542387422523895, -0.061542387422523895),
        Vectors.dense(0.5307324041981032, 0.46926759580189686),
        0.0
      ),
      Row(
        Vectors.dense(0.97, 0.82, 0.33), 
        1,
        Vectors.dense(-0.10732143550400228,0.10732143550400228), 
        Vectors.dense(0.4465443569128503,0.5534556430871497),
        1.0
      ),
      Row(
        Vectors.dense(0.13, 0.22, 0.23), 
        1,
        Vectors.dense(-0.09010562508687871,0.09010562508687871), 
        Vectors.dense(0.45506872106197505,0.544931278938025),
        1.0
      ),
      Row(
        Vectors.dense(0.14, 0.18, 0.1),
        0,
        Vectors.dense(0.0660650934240398,-0.0660650934240398), 
        Vectors.dense(0.5329845725520714,0.46701542744792857),
        0.0
      ),
      Row(
        Vectors.dense(0.9, 0.67, 0.17),
        0,
        Vectors.dense(0.057555746416403084,-0.057555746416403084), 
        Vectors.dense(0.5287461381176124,0.4712538618823876),
        0.0
      ),
      Row(
        Vectors.dense(0.66, 0.1, 0.31),
        0, 
        Vectors.dense(0.03719023254887147,-0.03719023254887147), 
        Vectors.dense(0.5185865479633033,0.4814134520366967),
        0.0
      )
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  
  @Test   
  @throws(classOf[Exception])
  def testBinaryClassificationWithWeights() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")

    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", IntegerType),
            ("weight", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), 0, 1.0),
          Row(Vectors.dense(0.97, 0.82, 0.33), 1, 2.0),
          Row(Vectors.dense(0.13, 0.22, 0.23), 1, 2.0),
          Row(Vectors.dense(0.14, 0.18, 0.1), 0, 1.0),
          Row(Vectors.dense(0.9, 0.67, 0.17), 0, 1.0),
          Row(Vectors.dense(0.66, 0.1, 0.31), 0, 1.0)
        ),
        Map[String,String]("weight" -> "weight")
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      
    val model = classifier.fit(pool)
    val predictions = model.transform(pool.data)
    
    val expectedPredictionsSchema = PoolTestHelpers.createSchema(
      Seq(
        ("features", SQLDataTypes.VectorType),
        ("label", IntegerType),
        ("weight", DoubleType),
        ("rawPrediction", SQLDataTypes.VectorType),
        ("probability", SQLDataTypes.VectorType),
        ("prediction", DoubleType)
      ),
      featureNames,
      /*addFeatureNamesMetadata*/ true,
      /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
    )
    
    val expectedPredictionsData = Seq(
      Row(
        Vectors.dense(0.1, 0.2, 0.11), 
        0, 
        1.0,
        Vectors.dense(0.061542387422523895, -0.061542387422523895),
        Vectors.dense(0.5307324041981032, 0.46926759580189686),
        0.0
      ),
      Row(
        Vectors.dense(0.97, 0.82, 0.33), 
        1,
        2.0,
        Vectors.dense(-0.10732143550400228,0.10732143550400228), 
        Vectors.dense(0.4465443569128503,0.5534556430871497),
        1.0
      ),
      Row(
        Vectors.dense(0.13, 0.22, 0.23), 
        1,
        2.0,
        Vectors.dense(-0.09010562508687871,0.09010562508687871), 
        Vectors.dense(0.45506872106197505,0.544931278938025),
        1.0
      ),
      Row(
        Vectors.dense(0.14, 0.18, 0.1),
        0,
        1.0,
        Vectors.dense(0.0660650934240398,-0.0660650934240398), 
        Vectors.dense(0.5329845725520714,0.46701542744792857),
        0.0
      ),
      Row(
        Vectors.dense(0.9, 0.67, 0.17),
        0,
        1.0,
        Vectors.dense(0.057555746416403084,-0.057555746416403084), 
        Vectors.dense(0.5287461381176124,0.4712538618823876),
        0.0
      ),
      Row(
        Vectors.dense(0.66, 0.1, 0.31),
        0, 
        1.0,
        Vectors.dense(0.03719023254887147,-0.03719023254887147), 
        Vectors.dense(0.5185865479633033,0.4814134520366967),
        0.0
      )
    )
    val expectedPredictions = spark.createDataFrame(
      spark.sparkContext.parallelize(expectedPredictionsData),
      StructType(expectedPredictionsSchema)
    )

    TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
  }
  
  @Test  
  @throws(classOf[Exception])
  def testSimpleMultiClassification() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    
    val featureNames = Array[String]("f1", "f2", "f3")
    
    val srcSchemaData = Seq(
      ("features", SQLDataTypes.VectorType),
      ("label", StringType),
      ("groupId", LongType),
      ("groupWeight", FloatType),
      ("subgroupId", IntegerType),
      ("weight", FloatType)
    )
    val srcData = Seq(
      Row(Vectors.dense(0.13, 0.22, 0.23), "1", 0x86F1B93B695F9E61L, 0.0f, 0x23D794E9, 1.0f),
      Row(Vectors.dense(0.1, 0.2, 0.11), "2", 0xB337C6FEFE2E2F73L, 1.0f, 0xD34BFBD7, 0.12f),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0", 0xB337C6FEFE2E2F73L, 1.0f, 0x19CE5B0A, 0.18f),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0", 0xD9DBDD3199D6518AL, 0.5f, 0x19CE5B0A, 1.0f),
      Row(Vectors.dense(0.66, 0.1, 0.31), "2", 0xD9DBDD3199D6518AL, 0.5f, 0x1FA606FD, 2.0f),
      Row(Vectors.dense(0.14, 0.18, 0.1), "1", 0xD9DBDD3199D6518AL, 0.5f, 0x62772D1C, 0.45f)
    )
    
    val pool = PoolTestHelpers.createRawPool(
        TestHelpers.getCurrentMethodName,
        PoolTestHelpers.createSchema(
          srcSchemaData,
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        srcData,
        Map("groupId" -> "groupId", "groupWeight" -> "groupWeight", "subgroupId" -> "subgroupId", "weight" -> "weight")
    )

    val classifier = new CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath)

    val model = classifier.fit(pool)
    val quantizedPool = pool.quantize()
    
    val expectedRawPrediction = Seq(
      (
        -0.36816374672276697,
        -0.5304785373559223,
        0.8986422840786894
      ),
      (
        -0.5099844837333486,
        -0.34565252624116527,
        0.8556370099745135
      ),
      (
        1.21778120500423,
        -0.7056515515173459,
        -0.5121296534868843
      ),
      (
        1.6681179010023703,
        -0.810035055766326,
        -0.8580828452360446
      ),
      (
        -1.0408685223435348,
        -1.037886278376265,
        2.0787548007197993
      ),
      (
        -0.5160549080699031,
        0.8676427693974363,
        -0.35158786132753306
      )
    )
    val expectedProbability = Seq(
      (
        0.1851964688532495,
        0.15744913673282981,
        0.6573543944139207
      ),
      (
        0.1640214677785814,
        0.19331660356400265,
        0.642661928657416
      ),
      (
        0.7556267140614333,
        0.11040050777538488,
        0.13397277816318182
      ),
      (
        0.8592096589894418,
        0.07208601139564771,
        0.06870432961491034
      ),
      (
        0.040583036064342666,
        0.040704245226631595,
        0.9187127187090257
      ),
      (
        0.16211681859545163,
        0.6467855943516575,
        0.19109758705289087
      )
    )
    val expectedPrediction = Seq(2.0, 2.0, 0.0, 0.0, 2.0, 1.0)
    
    for (rawPrediction <- Seq(false, true)) {
      for (probability <- Seq(false, true)) {
        for (prediction <- Seq(false, true)) {
          model.setRawPredictionCol(if (rawPrediction) "rawPrediction" else "")
          model.setProbabilityCol(if (probability) "probability" else "")
          model.setPredictionCol(if (prediction) "prediction" else "")
          val predictions = model.transform(pool.data)
          
          val expectedPredictionsData = mutable.Seq.concat(srcData)
          var expectedPredictionsSchemaData = srcSchemaData
          if (rawPrediction) {
            expectedPredictionsSchemaData 
              = expectedPredictionsSchemaData :+ ("rawPrediction", SQLDataTypes.VectorType)
            for (i <- 0 until srcData.length) {
              expectedPredictionsData(i) = TestHelpers.appendToRow(
                expectedPredictionsData(i),
                Vectors.dense(expectedRawPrediction(i)._1, expectedRawPrediction(i)._2, expectedRawPrediction(i)._3)
              )
            }
          }
          if (probability) {
            expectedPredictionsSchemaData 
              = expectedPredictionsSchemaData :+ ("probability", SQLDataTypes.VectorType)
            for (i <- 0 until srcData.length) {
              expectedPredictionsData(i) = TestHelpers.appendToRow(
                expectedPredictionsData(i),
                Vectors.dense(expectedProbability(i)._1, expectedProbability(i)._2, expectedProbability(i)._3)
              )
            }
          }
          if (prediction) {
            expectedPredictionsSchemaData 
              = expectedPredictionsSchemaData :+ ("prediction", DoubleType)
            for (i <- 0 until srcData.length) {
              expectedPredictionsData(i) = TestHelpers.appendToRow(
                expectedPredictionsData(i),
                expectedPrediction(i)
              )
            }
          }
          val expectedPredictionsSchema = PoolTestHelpers.createSchema(
            expectedPredictionsSchemaData,
            featureNames,
            /*addFeatureNamesMetadata*/ true,
            /*nullableFields*/ Seq("rawPrediction", "probability", "prediction")
          )
          val expectedPredictions = spark.createDataFrame(
            spark.sparkContext.parallelize(expectedPredictionsData),
            StructType(expectedPredictionsSchema)
          )
          
          TestHelpers.assertEqualsWithPrecision(expectedPredictions, predictions)
          
          // check apply on quantized
          val quantizedPredictions = model.transformPool(quantizedPool)
          
          TestHelpers.assertEqualsWithPrecision(
            expectedPredictions.drop("features"),
            quantizedPredictions.drop("features")
          )
        }
      }
    }
  }

  @Test 
  @throws(classOf[Exception])
  def testSerialization() {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName);
    {
      val path = System.getProperty("user.home") + "/catboost/spark/debug/serialized_classifier_0"
      val classifier = new CatBoostClassifier
      classifier.write.overwrite.save(path)
      val loadedClassifier = CatBoostClassifier.load(path)
      //Assert.assertEquals(classifier, loadedClassifier)
    }
    {
      val path = System.getProperty("user.home") + "/catboost/spark/debug/serialized_classifier_1"
      val classifier = new CatBoostClassifier().setLossFunction("MultiClass").setIterations(2)
      classifier.write.overwrite.save(path)
      val loadedClassifier = CatBoostClassifier.load(path)
      
      // TODO - uids
      //Assert.assertEquals(classifier, loadedClassifier)
    }
  }
}
