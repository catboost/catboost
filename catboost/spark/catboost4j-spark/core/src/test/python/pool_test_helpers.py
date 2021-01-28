import tempfile

from pyspark.ml.linalg import *
from pyspark.ml.param import *
from pyspark.sql import *
from pyspark.sql.types import *

import test_helpers


def createSchema(
    schemaDesc, #Seq[(String,DataType)],
    featureNames, #Seq[String],
    addFeatureNamesMetadata = True, # Boolean = true,
    nullableFields = [], # Seq[String] = Seq(),
    catFeaturesNumValues = {}, # Map[String,Int] = Map[String,Int](),
    catFeaturesValues = {} #: Map[String,Seq[String]] = Map[String,Seq[String]]()
): #: Seq[StructField] = {
    result = []
    for name, dataType in schemaDesc:
        if (addFeatureNamesMetadata and ((name == "features") or (name == "f1"))):
            numericAttrs = []
            nominalAttrs = []

            for i, fname in enumerate(featureNames):
                if fname in catFeaturesNumValues:
                    nominalAttrs.append({"num_vals": catFeaturesNumValues[fname], "idx": i, "name": fname})
                if fname in catFeaturesValues:
                    nominalAttrs.append({"vals": catFeaturesValues[fname], "idx": i, "name": fname})
                else:
                    numericAttrs.append({"idx": i, "name": fname})

            attrs = {}
            if numericAttrs:
                attrs["numeric"] = numericAttrs
            if nominalAttrs:
                attrs["nominal"] = nominalAttrs

            metadata = {"ml_attr": {"attrs": attrs, "num_attrs": len(featureNames)}}

            result.append(
                StructField(name, dataType, name in nullableFields, metadata)
            )
        else:
            result.append(StructField(name, dataType, name in nullableFields))
    return result


def writeToTempFile(str):
    path = tempfile.mkstemp(prefix='PoolTest_')[1]
    with open(path, 'w') as f:
        f.write(str)
    return path

def comparePoolWithExpectedData(
    sparkSession,
    pool,
    expectedDataSchema,
    expectedData,
    expectedFeatureNames,
    expectedPairsData = None,
    expectedPairsDataSchema = None,

    # set to true if order of rows might change. Requires sampleId or (groupId, sampleId) in data
    compareByIds=False
):
    expectedDf = sparkSession.createDataFrame(
        sparkSession.sparkContext.parallelize(expectedData),
        StructType(expectedDataSchema)
    )

    assert all(pool.getFeatureNames == expectedFeatureNames)

    assert pool.data.schema == StructType(expectedDataSchema)

    if compareByIds:
        expectedDataToCompare = expectedDf.orderBy(
            pool.getOrDefault(pool.groupIdCol),
            pool.getOrDefault(pool.sampleIdCol)
        )
        poolDataToCompare = pool.data.orderBy(
            pool.getOrDefault(pool.groupIdCol),
            pool.getOrDefault(pool.sampleIdCol)
        )
    else:
        expectedDataToCompare = expectedDf
        poolDataToCompare = pool.data

    test_helpers.assertEqualsWithPrecision(expectedDataToCompare, poolDataToCompare)

    if expectedPairsData:
        poolPairsDataToCompare = pool.pairsData.orderBy("groupId", "winnerId", "loserId")
        expectedPairsDataToCompare = spark.createDataFrame(
            sparkSesion.sparkContext.parallelize(expectedPairsData),
            StructType(expectedPairsDataSchema)
        ).orderBy("groupId", "winnerId", "loserId")
        assert pool.pairsData
        test_helpers.assertEqualsWithPrecision(expectedPairsDataToCompare, poolPairsDataToCompare)
    else:
        assert not pool.pairsData


def printPool(pool, name="pool"):
    print ('----------------------------')
    print ("Pool '%s'" % name)
    attrs = [
        "isQuantized",
        "getFeatureCount",
        "getFeatureNames",
        "count",
        "pairsCount",
        "getBaselineCount"
    ]
    for attr in attrs:
        print ("%s = %s" % (attr, getattr(pool, attr)()))

    print("data=")
    pool.data.show(truncate=False)

    if pool.pairsData:
        print("pairsData=")
        pool.pairsData.show(truncate=False)

    print ('----------------------------')

def createRawPool(
    appName, #:  String,
    srcDataSchema, #: Seq[StructField],
    srcData, #: Seq[Row],
    columnNames,#: Map[String, String] // standard column name to name of column in the dataset
    srcPairsData = None # Seq[Row]
):
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    if srcPairsData:
        pairsDataSchema = StructType(
            [
                StructField("groupId", LongType(), False),
                StructField("winnerId", IntegerType(), False),
                StructField("loserId", IntegerType(), False)
            ]
        )
        pairsDf = spark.createDataFrame(spark.sparkContext.parallelize(srcPairsData), pairsDataSchema)
        pool = catboost_spark.Pool(df, pairsDf)
    else:
        pool = catboost_spark.Pool(df)

    if ("features" in columnNames):
        pool = pool.setFeaturesCol(columnNames["features"])
    if ("groupId" in columnNames):
        pool = pool.setGroupIdCol(columnNames["groupId"])
    if ("sampleId" in columnNames):
        pool = pool.setSampleIdCol(columnNames["sampleId"])
    if ("subgroupId" in columnNames):
        pool = pool.setSubgroupIdCol(columnNames["subgroupId"])
    if ("weight" in columnNames):
        pool = pool.setWeightCol(columnNames["weight"])
    if ("groupWeight" in columnNames):
        pool = pool.setGroupWeightCol(columnNames["groupWeight"])
    return pool

