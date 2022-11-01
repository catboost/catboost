import tempfile

import pytest

import test_helpers
import pool_test_helpers

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
from pyspark.sql.types import *


def getSimpleTestPool(method_name):
    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("sampleId", IntegerType())
    ]

    srcData = [
        Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E),
        Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD),
        Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0),
        Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0),
        Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F),
        Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1)
    ]
    return pool_test_helpers.createRawPool(
        method_name,
        pool_test_helpers.createSchema(
            srcSchemaData,
            featureNames,
            addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "sampleId" : "sampleId"}
    )

def implTestCase(
    tmpFolderPath,
    pool,
    dataFramesOptions = None,  # Option[Map[String, String]] = None,
    dataFramesFormat = None,  # Option[String] = None,
    saveMode = None,  # Option[SaveMode] = None,
    retryCreation = False  # : Boolean = false
):
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    savedPoolPath = tmpFolderPath + "/pool"

    poolWriter = pool.write
    poolReader = catboost_spark.PoolReader(spark)
    if dataFramesFormat is not None:
        poolWriter.dataFramesWriterFormat(dataFramesFormat)
        poolReader.dataFramesReaderFormat(dataFramesFormat)

    if dataFramesOptions is not None:
        poolWriter.dataFramesWriterOptions(dataFramesOptions)
        poolReader.dataFramesReaderOptions(dataFramesOptions)

    if saveMode is not None:
        poolWriter.mode(saveMode)

    poolWriter.save(savedPoolPath)
    if retryCreation:
        poolWriter.save(savedPoolPath)

    loadedPool = poolReader.load(savedPoolPath)

    pool_test_helpers.printPool(pool, 'pool')
    pool_test_helpers.printPool(loadedPool, 'loadedPool')


def testSupportedFormats():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    pool = getSimpleTestPool(test_helpers.getCurrentMethodName())

    for format in ("parquet", "default"):
        tmpFolderPath = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName() + "." + format)

        implTestCase(
            tmpFolderPath,
            pool,
            dataFramesFormat = format if format != "default" else None
        )


def testUnsupportedFormats():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    pool = getSimpleTestPool(test_helpers.getCurrentMethodName())

    for format in ("csv", "text", "json", "orc"):
        tmpFolderPath = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName() + "." + format)

        with pytest.raises(Exception):
            implTestCase(
                tmpFolderPath,
                pool,
                dataFramesFormat = format if format != "default" else None
            )

def testWithPairs():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    pool = getSimpleTestPool(test_helpers.getCurrentMethodName())
    tmpFolderPath = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())

    implTestCase(tmpFolderPath, pool)

def testSuccessfulRetryCreation():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    pool = getSimpleTestPool(test_helpers.getCurrentMethodName())
    for saveMode in ("ignore", "overwrite"):
        tmpFolderPath = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName() + "." + saveMode)
        implTestCase(tmpFolderPath, pool, saveMode=saveMode, retryCreation=True)

def testFailedRetryCreation():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    pool = getSimpleTestPool(test_helpers.getCurrentMethodName())
    tmpFolderPath = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())

    with pytest.raises(Exception):
        implTestCase(tmpFolderPath, pool, retryCreation=True)

def testQuantized():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    pool = getSimpleTestPool(test_helpers.getCurrentMethodName())
    quantizedPool = pool.quantize()

    for format in ("orc", "parquet", "default"):
        tmpFolderPath = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName() + "." + format)
        implTestCase(
            tmpFolderPath,
            quantizedPool,
            dataFramesFormat = format if format != "default" else None
        )
