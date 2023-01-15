import pool_test_helpers
import test_helpers

import pytest


# Good
def testLoadDSVSimple():
    dataFile = pool_test_helpers.writeToTempFile(
        "0\t0.1\t0.2\n" +
        "1\t0.97\t0.82\n" +
        "0\t0.13\t0.22\n"
    )
    cdFile = pool_test_helpers.writeToTempFile(
        "0\tTarget"
    )

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    pool = catboost_spark.Pool.load(spark, dataFile, columnDescription=cdFile)
    featureNames = ["_f0", "_f1"]

    pool.data.show(truncate=False)

    """
    PoolTestHelpers.comparePoolWithExpectedData(
        pool,
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", StringType)
          ),
          featureNames,
          nullableFields=Seq("features", "label")
        ),
        Seq(
          Row(Vectors.dense(0.1, 0.2), "0"),
          Row(Vectors.dense(0.97, 0.82), "1"),
          Row(Vectors.dense(0.13, 0.22), "0")
        ),
        featureNames
    )
    """


# Good
def testLoadDSVWithHeader():
    dataFile = pool_test_helpers.writeToTempFile(
        "Target\tFeat0\tFeat1\n" +
        "0\t0.1\t0.2\n" +
        "1\t0.97\t0.82\n" +
        "0\t0.13\t0.22\n"
    )
    cdFile = pool_test_helpers.writeToTempFile(
        "0\tTarget"
    )

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    pool = catboost_spark.Pool.load(
        spark,
        dataFile,
        columnDescription=cdFile,
        poolLoadParams=catboost_spark.PoolLoadParams(hasHeader=True)
    )
    featureNames = ["_f0", "_f1"]

    pool.data.show(truncate=False)


# Good
def testLoadDSVWithDelimiter():
    dataFile = pool_test_helpers.writeToTempFile(
        "Target,Feat0,Feat1\n" +
        "0,0.1,0.2\n" +
        "1,0.97,0.82\n" +
        "0,0.13,0.22\n"
    )
    cdFile = pool_test_helpers.writeToTempFile(
        "0\tTarget"
    )

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    pool = catboost_spark.Pool.load(
        spark,
        dataFile,
        columnDescription=cdFile,
        poolLoadParams=catboost_spark.PoolLoadParams(hasHeader=True, delimiter=',')
    )
    featureNames = ["_f0", "_f1"]

    pool_test_helpers.printPool(pool)


# Good
def testLoadLibSVMSimple():
    dataFile = pool_test_helpers.writeToTempFile(
        "0 1:0.1 3:0.2\n" +
        "1 2:0.97 5:0.82 6:0.11 8:1.2\n" +
        "0 3:0.13 7:0.22 8:0.17\n"
    )
    cdFile = pool_test_helpers.writeToTempFile(
        "0\tTarget"
    )

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    pool = catboost_spark.Pool.load(
        spark,
        "libsvm://" + dataFile
    )

    pool_test_helpers.printPool(pool)


# Good
def testLoadDSVWithPairs():
    dataFile = pool_test_helpers.writeToTempFile(
        "0.12\tquery0\tsite1\t0.12\t1.0\t0.1\t0.2\t0.11\n" +
        "0.22\tquery0\tsite22\t0.18\t1.0\t0.97\t0.82\t0.33\n" +
        "0.34\tquery1\tSite9\t1.0\t0.0\t0.13\t0.22\t0.23\n" +
        "0.42\tQuery 2\tsite12\t0.45\t0.5\t0.14\t0.18\t0.1\n" +
        "0.01\tQuery 2\tsite22\t1.0\t0.5\t0.9\t0.67\t0.17\n" +
        "0.0\tQuery 2\tSite45\t2.0\t0.5\t0.66\t0.1\t0.31\n"
    )
    cdFile = pool_test_helpers.writeToTempFile(
        "0\tTarget\n" +
        "1\tGroupId\n" +
        "2\tSubgroupId\n" +
        "3\tWeight\n" +
        "4\tGroupWeight\n" +
        "5\tNum\tf0\n" +
        "6\tNum\tf1\n" +
        "7\tNum\tf2\n"
    )
    pairsFile = pool_test_helpers.writeToTempFile(
        "query0\t0\t1\n" +
        "Query 2\t0\t2\n" +
        "Query 2\t1\t2\n"
    )

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    pool = catboost_spark.Pool.load(
        spark,
        dataFile,
        columnDescription=cdFile,
        pairsDataPathWithScheme="dsv-grouped://" + pairsFile
    )

    pool_test_helpers.printPool(pool)
