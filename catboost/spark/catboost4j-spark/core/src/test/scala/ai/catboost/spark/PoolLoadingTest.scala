package ai.catboost.spark;

import org.apache.spark.ml.linalg._;
import org.apache.spark.sql._;
import org.apache.spark.sql.types._;

import org.junit.{Assert,Test,Ignore};

import ai.catboost.CatBoostError
import ai.catboost.spark.params._


class PoolLoadingTest {

    @Test
    @throws(classOf[Exception])
    def testLoadDSVSimple() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0\t0.1\t0.2\n" +
        "1\t0.97\t0.82\n" +
        "0\t0.13\t0.22\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVSimple")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams()
      )

      val featureNames = Array[String]("0", "1")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          nullableFields = Seq("features", "label")
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2), "0"),
          Row(Vectors.dense(0.97, 0.82), "1"),
          Row(Vectors.dense(0.13, 0.22), "0")
        ),
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadDSVWithHeader() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "Target\tFeat0\tFeat1\n" +
        "0\t0.1\t0.2\n" +
        "1\t0.97\t0.82\n" +
        "0\t0.13\t0.22\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVWithHeader")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams().setHasHeader(true)
      )

      val featureNames = Array[String]("Feat0", "Feat1")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          nullableFields = Seq("features", "label")
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2), "0"),
          Row(Vectors.dense(0.97, 0.82), "1"),
          Row(Vectors.dense(0.13, 0.22), "0")
        ),
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadDSVWithDelimiter() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "Target,Feat0,Feat1\n" +
        "0,0.1,0.2\n" +
        "1,0.97,0.82\n" +
        "0,0.13,0.22\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVWithHeader")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams().setHasHeader(true).setDelimiter(",")
      )

      val featureNames = Array[String]("Feat0", "Feat1")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          nullableFields = Seq("features", "label")
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2), "0"),
          Row(Vectors.dense(0.97, 0.82), "1"),
          Row(Vectors.dense(0.13, 0.22), "0")
        ),
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadDSVGroupData() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0.12\tquery0\tsite1\t0.12\t1.0\t0.1\t0.2\t0.11\n" +
        "0.22\tquery0\tsite22\t0.18\t1.0\t0.97\t0.82\t0.33\n" +
        "0.34\tquery1\tSite9\t1.0\t0.0\t0.13\t0.22\t0.23\n" +
        "0.42\tQuery 2\tsite12\t0.45\t0.5\t0.14\t0.18\t0.1\n" +
        "0.01\tQuery 2\tsite22\t1.0\t0.5\t0.9\t0.67\t0.17\n" +
        "0.0\tQuery 2\tSite45\t2.0\t0.5\t0.66\t0.1\t0.31\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget\n" +
        "1\tGroupId\n" +
        "2\tSubgroupId\n" +
        "3\tWeight\n" +
        "4\tGroupWeight\n" +
        "5\tNum\tf0\n" +
        "6\tNum\tf1\n" +
        "7\tNum\tf2\n"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVGroupData")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams()
      )

      val featureNames = Array[String]("f0", "f1", "f2")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType),
            ("groupId", LongType),
            ("groupWeight", FloatType),
            ("subgroupId", IntegerType),
            ("weight", FloatType)
          ),
          featureNames,
          nullableFields = Seq("features", "label", "groupId", "groupWeight", "subgroupId", "weight")
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 1.0f, 0xD34BFBD7, 0.12f),
          Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 1.0f, 0x19CE5B0A, 0.18f),
          Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0.0f, 0x23D794E9, 1.0f),
          Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0.5f, 0x62772D1C, 0.45f),
          Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0.5f, 0x19CE5B0A, 1.0f),
          Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0.5f, 0x1FA606FD, 2.0f)
        ),
        featureNames
      )
    }


    // TODO(akhropov): cat features are not supported yet
    @Test(expected = classOf[CatBoostError])
    @throws(classOf[Exception])
    def testLoadDSVFloatAndCatFeatures() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0.12\tquery0\t0.1\tMale\t0.2\tGermany\t0.11\n" +
        "0.22\tquery0\t0.97\tFemale\t0.82\tRussia\t0.33\n" +
        "0.34\tquery1\t0.13\tMale\t0.22\tUSA\t0.23\n" +
        "0.42\tQuery 2\t0.14\tMale\t0.18\tFinland\t0.1\n" +
        "0.01\tQuery 2\t0.9\tFemale\t0.67\tUSA\t0.17\n" +
        "0.0\tQuery 2\t0.66\tFemale\t0.1\tUK\t0.31\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget\n" +
        "1\tGroupId\n" +
        "2\tNum\tfloat0\n" +
        "3\tCateg\tGender1\n" +
        "4\tNum\tfloat2\n" +
        "5\tCateg\tCountry3\n" +
        "6\tNum\tfloat4\n"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVFloatAndCatFeatures")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams()
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadLibSVMSimple() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0 1:0.1 3:0.2\n" +
        "1 2:0.97 5:0.82 6:0.11 8:1.2\n" +
        "0 3:0.13 7:0.22 8:0.17\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadLibSVMSimple")
        .getOrCreate();

      val pool = Pool.load(
        spark,
        "libsvm://" + dataFile.toString
      )
        //columnDescription = cdFile,
        //params = new PoolLoadParams()
      //)

      val featureNames = Array[String]("0", "1", "2", "3", "4", "5", "6", "7")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", FloatType)
          ),
          featureNames,
          nullableFields = Seq("features", "label")
        ),
        Seq(
          Row(Vectors.sparse(8, Seq((0, 0.1), (2, 0.2))), 0.0f),
          Row(Vectors.sparse(8, Seq((1, 0.97), (4, 0.82), (5, 0.11), (7, 1.2))), 1.0f),
          Row(Vectors.sparse(8, Seq((2, 0.13), (6, 0.22), (7, 0.17))), 0.0f)
        ),
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadLibSVMWithColumnDescription() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0 1:0.1 3:0.2\n" +
        "1 2:0.97 5:0.82 6:0.11 8:1.2\n" +
        "0 3:0.13 7:0.22 8:0.17\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget\n" +
        "1\tNum\tF1\n" +
        "2\tNum\tF2\n" +
        "3\tNum\tF3\n" +
        "4\tNum\tF4\n" +
        "5\tNum\tF5\n" +
        "6\tNum\tF6\n" +
        "7\tNum\tF7\n" +
        "8\tNum\tF8\n"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadLibSVMSimple")
        .getOrCreate();

      val pool = Pool.load(
        spark,
        "libsvm://" + dataFile.toString,
        columnDescription = cdFile
      )
        //params = new PoolLoadParams()
      //)

      val featureNames = Array[String]("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", FloatType)
          ),
          featureNames,
          nullableFields = Seq("features", "label")
        ),
        Seq(
          Row(Vectors.sparse(8, Seq((0, 0.1), (2, 0.2))), 0.0f),
          Row(Vectors.sparse(8, Seq((1, 0.97), (4, 0.82), (5, 0.11), (7, 1.2))), 1.0f),
          Row(Vectors.sparse(8, Seq((2, 0.13), (6, 0.22), (7, 0.17))), 0.0f)
        ),
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadDSVWithPairs() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0.12\tquery0\tsite1\t0.12\t1.0\t0.1\t0.2\t0.11\n" +
        "0.22\tquery0\tsite22\t0.18\t1.0\t0.97\t0.82\t0.33\n" +
        "0.34\tquery1\tSite9\t1.0\t0.0\t0.13\t0.22\t0.23\n" +
        "0.42\tQuery 2\tsite12\t0.45\t0.5\t0.14\t0.18\t0.1\n" +
        "0.01\tQuery 2\tsite22\t1.0\t0.5\t0.9\t0.67\t0.17\n" +
        "0.0\tQuery 2\tSite45\t2.0\t0.5\t0.66\t0.1\t0.31\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget\n" +
        "1\tGroupId\n" +
        "2\tSubgroupId\n" +
        "3\tWeight\n" +
        "4\tGroupWeight\n" +
        "5\tNum\tf0\n" +
        "6\tNum\tf1\n" +
        "7\tNum\tf2\n"
      )
      val pairsFile = PoolTestHelpers.writeToTempFile(
        "query0\t0\t1\n" +
        "Query 2\t0\t2\n" +
        "Query 2\t1\t2\n"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVWithPairs")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams(),
        pairsDataPathWithScheme = "dsv-grouped://" + pairsFile.toString
      )

      val featureNames = Array[String]("f0", "f1", "f2")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType),
            ("groupId", LongType),
            ("groupWeight", FloatType),
            ("subgroupId", IntegerType),
            ("weight", FloatType),
            ("sampleId", LongType)
          ),
          featureNames,
          nullableFields = Seq("features", "label", "groupId", "groupWeight", "subgroupId", "weight", "sampleId")
        ),
        Seq(
          Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0.0f, 0x23D794E9, 1.0f, 0L),
          Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 1.0f, 0xD34BFBD7, 0.12f, 0L),
          Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 1.0f, 0x19CE5B0A, 0.18f, 1L),
          Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0.5f, 0x62772D1C, 0.45f, 0L),
          Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0.5f, 0x19CE5B0A, 1.0f, 1L),
          Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0.5f, 0x1FA606FD, 2.0f, 2L)
        ),
        featureNames,
        Some(
          Seq(
            Row(0xB337C6FEFE2E2F73L, 0L, 1L),
            Row(0xD9DBDD3199D6518AL, 0L, 2L),
            Row(0xD9DBDD3199D6518AL, 1L, 2L)
          )
        ),
        Some(
          Seq(
            StructField("groupId", LongType, false),
            StructField("winnerId", LongType, false),
            StructField("loserId", LongType, false)
          )
        ),
        compareByIds = true
      )
    }

    @Test
    @throws(classOf[Exception])
    def testLoadDSVWithPairsWithWeights() = {
      val dataFile = PoolTestHelpers.writeToTempFile(
        "0.12\tquery0\tsite1\t0.12\t1.0\t0.1\t0.2\t0.11\n" +
        "0.22\tquery0\tsite22\t0.18\t1.0\t0.97\t0.82\t0.33\n" +
        "0.34\tquery1\tSite9\t1.0\t0.0\t0.13\t0.22\t0.23\n" +
        "0.42\tQuery 2\tsite12\t0.45\t0.5\t0.14\t0.18\t0.1\n" +
        "0.01\tQuery 2\tsite22\t1.0\t0.5\t0.9\t0.67\t0.17\n" +
        "0.0\tQuery 2\tSite45\t2.0\t0.5\t0.66\t0.1\t0.31\n"
      )
      val cdFile = PoolTestHelpers.writeToTempFile(
        "0\tTarget\n" +
        "1\tGroupId\n" +
        "2\tSubgroupId\n" +
        "3\tWeight\n" +
        "4\tGroupWeight\n" +
        "5\tNum\tf0\n" +
        "6\tNum\tf1\n" +
        "7\tNum\tf2\n"
      )
      val pairsFile = PoolTestHelpers.writeToTempFile(
        "query0\t0\t1\t1.0\n" +
        "Query 2\t0\t2\t2.0\n" +
        "Query 2\t1\t2\t0.5\n"
      )

      val spark = SparkSession.builder()
        //.master("local[4]")
        .master("local[1]")
        .appName("testLoadDSVWithPairsWithWeights")
        .getOrCreate()

      val pool = Pool.load(
        spark,
        dataFile.toString,
        columnDescription = cdFile,
        params = new PoolLoadParams(),
        pairsDataPathWithScheme = "dsv-grouped://" + pairsFile.toString
      )

      val featureNames = Array[String]("f0", "f1", "f2")

      PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType),
            ("groupId", LongType),
            ("groupWeight", FloatType),
            ("subgroupId", IntegerType),
            ("weight", FloatType),
            ("sampleId", LongType)
          ),
          featureNames,
          nullableFields = Seq("features", "label", "groupId", "groupWeight", "subgroupId", "weight", "sampleId")
        ),
        Seq(
          Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E61L, 0.0f, 0x23D794E9, 1.0f, 0L),
          Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F73L, 1.0f, 0xD34BFBD7, 0.12f, 0L),
          Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F73L, 1.0f, 0x19CE5B0A, 0.18f, 1L),
          Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518AL, 0.5f, 0x62772D1C, 0.45f, 0L),
          Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518AL, 0.5f, 0x19CE5B0A, 1.0f, 1L),
          Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518AL, 0.5f, 0x1FA606FD, 2.0f, 2L)
        ),
        featureNames,
        Some(
          Seq(
            Row(0xB337C6FEFE2E2F73L, 0L, 1L, 1.0f),
            Row(0xD9DBDD3199D6518AL, 0L, 2L, 2.0f),
            Row(0xD9DBDD3199D6518AL, 1L, 2L, 0.5f)
          )
        ),
        Some(
          Seq(
            StructField("groupId", LongType, false),
            StructField("winnerId", LongType, false),
            StructField("loserId", LongType, false),
            StructField("weight", FloatType, false)
          )
        ),
        compareByIds = true
      )
    }
}
