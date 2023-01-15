import test_helpers
import pool_test_helpers

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
from pyspark.sql.types import *


def implTestQuantizeCase(
    srcDataSchema,
    srcData,
    quantizationParams
):
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    print ("srcDataSchema=", srcDataSchema)

    df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    pool = catboost_spark.Pool(df)
    quantizedPool = pool.quantize(quantizationParams)

    pool_test_helpers.printPool(pool, 'raw')
    pool_test_helpers.printPool(quantizedPool, 'quantized')


# Good
def testQuantize():
    featureNames = ["f1", "f2", "f3"]

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    implTestQuantizeCase(
        pool_test_helpers.createSchema(
            [
                ("features", VectorUDT()),
                ("label", DoubleType())
            ],
            featureNames,
            addFeatureNamesMetadata=True
        ),
        srcData=[
            Row(Vectors.dense(0.0, 1.0, 0.2), 0.0),
            Row(Vectors.dense(0.1, 1.1, 2.1), 1.0),
            Row(Vectors.dense(0.2, 1.2, 2.2), 1.0),
            Row(Vectors.dense(0.0, 1.1, 3.2), 0.0)
        ],
        quantizationParams=catboost_spark.QuantizationParams()
    )


# Good
def testQuantizeWithNaNsAndBorderCount():
    featureNames = ["F1", "F2", "F3", "F4"]

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    implTestQuantizeCase(
        pool_test_helpers.createSchema(
            [
                ("features", VectorUDT()),
                ("label", DoubleType())
            ],
            featureNames,
            addFeatureNamesMetadata=True
        ),
        srcData=[
            Row(Vectors.dense(0.0, 1.0, 0.2, 100.11), 3.0),
            Row(Vectors.dense(float('nan'), 1.1, float('nan'), 20.2), 1.0),
            Row(Vectors.dense(0.2, 1.2, 2.2, 32.4), 11.0),
            Row(Vectors.dense(float('nan'), 0.0, 2.2, 71.1), 0.2),
            Row(Vectors.dense(float('nan'), 1.1, 0.4, 92.2), 6.1),
            Row(Vectors.dense(0.1, 0.0, 1.8, 111.0), 2.0),
            Row(Vectors.dense(0.28, 0.0, 8.3, 333.2), 0.0)
        ],
        quantizationParams=catboost_spark.QuantizationParams(borderCount=2, nanMode=catboost_spark.ENanMode.Max)
    )


# Good
def testQuantizeWithNaNsAndIgnoredFeatures():
    featureNames = ["F1", "F2", "F3", "F4"]

    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark
    implTestQuantizeCase(
        pool_test_helpers.createSchema(
            [
                ("features", VectorUDT()),
                ("label", DoubleType())
            ],
            featureNames,
            addFeatureNamesMetadata=True
        ),
        srcData=[
            Row(Vectors.dense(0.0, 1.0, 0.2, 100.11), 3.0),
            Row(Vectors.dense(float('nan'), 1.1, float('nan'), 20.2), 1.0),
            Row(Vectors.dense(0.2, 1.2, 2.2, 32.4), 11.0),
            Row(Vectors.dense(float('nan'), 0.0, 2.2, 71.1), 0.2),
            Row(Vectors.dense(float('nan'), 1.1, 0.4, 92.2), 6.1),
            Row(Vectors.dense(0.1, 0.0, 1.8, 111.0), 2.0),
            Row(Vectors.dense(0.28, 0.0, 8.3, 333.2), 0.0)
        ],
        quantizationParams=catboost_spark.QuantizationParams(
            borderCount=2,
            ignoredFeaturesIndices=[0, 2]
        )
    )
