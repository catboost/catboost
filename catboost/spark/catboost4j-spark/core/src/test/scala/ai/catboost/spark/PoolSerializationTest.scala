package ai.catboost.spark;

import java.io.{File,PrintWriter}

import org.apache.spark.sql.{SaveMode,SparkSession}

import org.junit.{Assert,Test,Ignore,Rule}
import org.junit.rules.TemporaryFolder


class PoolSerializationTest {
  val _temporaryFolder = new TemporaryFolder

  @Rule
  def temporaryFolder = _temporaryFolder

  def getSimpleTestPool(spark: SparkSession) : Pool = {
    PoolTestHelpers.createRandomPool(
      spark,
      floatFeatureCount = 10,
      catFeatureCount = 5,
      objectCount = 30,
      hasSampleId = true
    )
  }

  def testCase(
    tmpFolderPath : String,
    pool: Pool,
    dataFramesOptions: Option[Map[String, String]] = None,
    dataFramesFormat: Option[String] = None,
    saveMode: Option[SaveMode] = None,
    retryCreation: Boolean = false
  ) = {
    val savedPoolPath = tmpFolderPath + "/pool"

    val poolWriter = pool.write
    val poolReader = new PoolReader(pool.data.sparkSession)
    if (dataFramesFormat.nonEmpty) {
      poolWriter.dataFramesWriterFormat(dataFramesFormat.get)
      poolReader.dataFramesReaderFormat(dataFramesFormat.get)
    }
    if (dataFramesOptions.nonEmpty) {
      poolWriter.dataFramesWriterOptions(dataFramesOptions.get)
      poolReader.dataFramesReaderOptions(dataFramesOptions.get)
    }
    if (saveMode.nonEmpty) {
      poolWriter.mode(saveMode.get)
    }
    poolWriter.save(savedPoolPath)
    if (retryCreation) {
      poolWriter.save(savedPoolPath)
    }
    val loadedPool = poolReader.load(savedPoolPath)

    PoolTestHelpers.assertEqualsWithPrecision(
      pool,
      loadedPool,
      compareByIds = true,
      ignoreNullableInSchema = true
    )
  }

  @Test
  @throws(classOf[Exception])
  def testSupportedFormats() = {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    val pool = getSimpleTestPool(spark)

    for (format <- Seq("parquet", "default")) {
      val tmpFolderPath = temporaryFolder.newFolder(TestHelpers.getCurrentMethodName + "." + format).getPath
      testCase(
        tmpFolderPath,
        pool,
        dataFramesFormat=if (!format.equals("default")) { Some(format) } else None
      )
    }
  }

  @Test(expected = classOf[Exception])
  @throws(classOf[Exception])
  def testUnsupportedFormats() = {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    val pool = getSimpleTestPool(spark)

    for (format <- Seq("csv", "text", "json", "orc")) {
      val tmpFolderPath = temporaryFolder.newFolder(TestHelpers.getCurrentMethodName + "." + format).getPath
      testCase(
        tmpFolderPath,
        pool,
        dataFramesFormat = if (!format.equals("default")) { Some(format) } else None
      )
    }
  }

  @Test
  @throws(classOf[Exception])
  def testWithPairs() = {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    val pool = PoolTestHelpers.createRandomPool(
      spark,
      floatFeatureCount = 10,
      catFeatureCount = 5,
      hasSampleId = true,
      groupCount = 10,
      pairsFraction = 0.3
    )
    testCase(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath, pool)
  }

  @Test //(expected = NullPointerException.class)
  @throws(classOf[Exception])
  def testSuccessfulRetryCreation() = {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    val pool = getSimpleTestPool(spark)

    for (saveMode <- Seq(SaveMode.Ignore, SaveMode.Overwrite)) {
      val tmpFolderPath = temporaryFolder.newFolder(
            "testSuccessfulRetryCreation." + saveMode.toString
      ).getPath
      testCase(tmpFolderPath, pool, saveMode = Some(saveMode), retryCreation = true)
    }
  }

  @Test(expected = classOf[Exception])
  @throws(classOf[Exception])
  def testFailedRetryCreation() = {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    val pool = getSimpleTestPool(spark)
    testCase(temporaryFolder.newFolder(TestHelpers.getCurrentMethodName).getPath, pool, retryCreation = true)
  }

  @Test
  @throws(classOf[Exception])
  def testQuantized() = {
    val spark = TestHelpers.getOrCreateSparkSession(TestHelpers.getCurrentMethodName)
    val pool = getSimpleTestPool(spark)
    val quantizedPool = pool.quantize()

    for (format <- Seq("orc", "parquet", "default")) {
      val tmpFolderPath = temporaryFolder.newFolder(TestHelpers.getCurrentMethodName + "." + format).getPath
      testCase(
        tmpFolderPath,
        quantizedPool,
        dataFramesFormat = if (!format.equals("default")) { Some(format) } else None
      )
    }
  }
}
