import collections
import datetime
import os
import shutil
import tempfile

import test_helpers
import pool_test_helpers

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
from pyspark.sql.types import *


def testSimple():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcData = [
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(pool)
    predictions = model.transform(pool.data)

    print ("predictions")
    predictions.show(truncate=False)


def testSimpleOnDataFrame():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcDataSchema = pool_test_helpers.createSchema(
        [
            ("features", VectorUDT()),
            ("label", DoubleType())
        ],
        featureNames,
        addFeatureNamesMetadata=True
    )

    srcData = [
      Row(Vectors.dense(0.1, 0.2, 0.11), 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), 1.1),
      Row(Vectors.dense(0.13, 0.22, 0.23), 2.1),
      Row(Vectors.dense(0.14, 0.18, 0.1), 0.0),
      Row(Vectors.dense(0.9, 0.67, 0.17), -1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), 0.62)
    ]

    df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(df)
    predictions = model.transform(df)

    print ("predictions")
    predictions.show(truncate=False)


def testFeaturesRenamed():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("f1", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcData = [
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight", "features": "f1"}
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName()))
      .setFeaturesCol("f1")
    )
    model = regressor.fit(pool).setFeaturesCol("f1")
    predictions = model.transform(pool.data)

    print ("predictions")
    predictions.show(truncate=False)


def testWithEvalSet():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcTrainData = [
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    srcTestData = [
      Row(Vectors.dense(0.0, 0.33, 1.1), "0.22", 0x4AAFFF456765757, 0xD34BFBD, 0.1),
      Row(Vectors.dense(0.02, 0.0, 0.38), "0.11", 0x686726738873ABC, 0x23D794E, 1.0),
      Row(Vectors.dense(0.86, 0.54, 0.9), "0.48", 0x7652786FF37ABBE, 0x19CE5B0, 0.17)
    ]

    trainPool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcTrainData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )
    testPool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcTestData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName()))
    )
    model = regressor.fit(trainPool, evalDatasets=[testPool])
    predictions = model.transform(testPool.data)

    print ("predictions")
    predictions.show(truncate=False)

def testWithEvalSets():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcTrainData = [
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    srcTestDataList = [
        [
          Row(Vectors.dense(0.0, 0.33, 1.1), "0.22", 0x4AAFFF456765757, 0xD34BFBD, 0.1),
          Row(Vectors.dense(0.02, 0.0, 0.38), "0.11", 0x686726738873ABC, 0x23D794E, 1.0),
          Row(Vectors.dense(0.86, 0.54, 0.9), "0.48", 0x7652786FF37ABBE, 0x19CE5B0, 0.17)
        ],
        [
            Row(Vectors.dense(0.12, 0.28, 2.2), "0.1", 0x4AAFADDE3765757, 0xD34BFBD, 0.11),
            Row(Vectors.dense(0.0, 0.0, 0.92), "0.9", 0x686726738873ABC, 0x23D794E, 1.1),
            Row(Vectors.dense(0.13, 2.1, 0.45), "0.88", 0x686726738873ABC, 0x56A96DF, 1.2),
            Row(Vectors.dense(0.17, 0.11, 0.0), "0.0", 0xADD57787677BBA2, 0x19CE5B0, 1.0)
        ]
    ]

    trainPool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcTrainData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )
    testPools = [
        pool_test_helpers.createRawPool(
            test_helpers.getCurrentMethodName,
            pool_test_helpers.createSchema(
              srcSchemaData,
              featureNames,
              addFeatureNamesMetadata=True
            ),
            srcTestData,
            {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
        )
        for srcTestData in srcTestDataList
    ]

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName()))
    )
    model = regressor.fit(trainPool, evalDatasets=testPools)
    predictionsList = [ model.transform(testPool.data) for testPool in testPools ]

    for i in range(2):
        print ("predictions", i)
        predictionsList[i].show(truncate=False)

def testDurationParam():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcData = [
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName()))
      .setSnapshotInterval(datetime.timedelta(hours=10))
    )
    model = regressor.fit(pool)
    predictions = model.transform(pool.data)

    print ("predictions")
    predictions.show(truncate=False)

def testParams():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcData = [
      Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
      Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
      Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
      Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
      Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
      Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )

    firstFeatureUsePenaltiesMap = collections.OrderedDict([("f1", 0.0), ("f2", 1.1), ("f3", 2.0)])

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName()))
      .setLeafEstimationIterations(10)
      .setFirstFeatureUsePenaltiesMap(firstFeatureUsePenaltiesMap)
      .setFeatureWeightsList([1.0, 2.0, 3.0])
    )
    model = regressor.fit(pool)
    predictions = model.transform(pool.data)

    print ("predictions")
    predictions.show(truncate=False)


def testWithPairs():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("sampleId", LongType()),
        ("weight", FloatType())
    ]

    srcData = [
        Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
        Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
        Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
        Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
        Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
        Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    srcPairsData = [
        Row(0xB337C6FEFE2E2F7, 0xD34BFBD, 0x19CE5B0),
        Row(0xD9DBDD3199D6518, 0x19CE5B0, 0x62772D1),
        Row(0xD9DBDD3199D6518, 0x62772D1, 0x1FA606F)
    ]

    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "sampleId": "sampleId", "weight": "weight"},
        srcPairsData
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName()))
      .setLossFunction("PairLogit")
      .setHasTime(True)
    )

    model = regressor.fit(pool)
    predictions = model.transform(pool.data)

    print ("predictions")
    predictions.show(truncate=False)


def testModelSerialization():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcData = [
        Row(Vectors.dense(0.13, 0.22, 0.23), "0.34", 0x86F1B93B695F9E6, 0x23D794E, 1.0),
        Row(Vectors.dense(0.1, 0.2, 0.11), "0.12", 0xB337C6FEFE2E2F7, 0xD34BFBD, 0.12),
        Row(Vectors.dense(0.97, 0.82, 0.33), "0.22", 0xB337C6FEFE2E2F7, 0x19CE5B0, 0.18),
        Row(Vectors.dense(0.9, 0.67, 0.17), "0.01", 0xD9DBDD3199D6518, 0x19CE5B0, 1.0),
        Row(Vectors.dense(0.66, 0.1, 0.31), "0.0", 0xD9DBDD3199D6518, 0x1FA606F, 2.0),
        Row(Vectors.dense(0.14, 0.18, 0.1), "0.42", 0xD9DBDD3199D6518, 0x62772D1, 0.45)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "subgroupId": "subgroupId", "weight": "weight"}
    )

    regressor = (catboost_spark.CatBoostRegressor()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = regressor.fit(pool)
    predictions = model.transform(pool.data)

    print ("predictions")
    predictions.show(truncate=False)

    modelsDir = tempfile.mkdtemp(prefix="catboost_models_")

    nativeCatBoostModelPath = os.path.join(modelsDir, "regressor_model.cbm")
    model.saveNativeModel(nativeCatBoostModelPath)

    loadedCatBoostModel = catboost_spark.CatBoostRegressionModel.loadNativeModel(nativeCatBoostModelPath)
    predictionsLoadedCatBoost = loadedCatBoostModel.transform(pool.data)
    print ("predictionsLoadedCatBoost")
    predictionsLoadedCatBoost.show(truncate=False)

    nativeJsonModelPath = os.path.join(modelsDir, "regressor_model.json")
    model.saveNativeModel(nativeJsonModelPath, catboost_spark.EModelType.Json)

    nativeOnnxModelPath = os.path.join(modelsDir, "regressor_model.onnx")
    model.saveNativeModel(
        nativeOnnxModelPath,
        catboost_spark.EModelType.Onnx,
        {
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "test model for regression",
            "onnx_graph_name": "CatBoostModel_for_regression"
        }
    )

    loadedOnnxModel = catboost_spark.CatBoostRegressionModel.loadNativeModel(nativeOnnxModelPath, catboost_spark.EModelType.Onnx)
    predictionsLoadedOnnx = loadedOnnxModel.transform(pool.data)
    print ("predictionsLoadedOnnx")
    predictionsLoadedOnnx.show(truncate=False)

    sparkModelPath = os.path.join(modelsDir, "regressor_model")
    model.write().overwrite().save(sparkModelPath)
    loadedModel = catboost_spark.CatBoostRegressionModel.load(sparkModelPath)

    predictionsLoaded = loadedModel.transform(pool.data)
    print ("predictionsLoaded")
    predictionsLoaded.show(truncate=False)

    shutil.rmtree(modelsDir)


def testModelSerializationInPipeline():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    srcData = [
        Row(0.12, "query0", 0.1, "Male", 0.2, "Germany", 0.11),
        Row(0.22, "query0", 0.97, "Female", 0.82, "Russia", 0.33),
        Row(0.34, "query1", 0.13, "Male", 0.22, "USA", 0.23),
        Row(0.42, "Query 2", 0.14, "Male", 0.18, "Finland", 0.1),
        Row(0.01, "Query 2", 0.9, "Female", 0.67, "USA", 0.17),
        Row(0.0, "Query 2", 0.66, "Female", 0.1, "UK", 0.31)
    ]
    srcDataSchema = [
        StructField("Label", DoubleType()),
        StructField("GroupId", StringType()),
        StructField("float0", DoubleType()),
        StructField("Gender1", StringType()),
        StructField("float2", DoubleType()),
        StructField("Country3", StringType()),
        StructField("float4", DoubleType())
    ]

    df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    indexers = [
        StringIndexer(inputCol=catFeature, outputCol=catFeature + "Index")
        for catFeature in ["Gender1", "Country3"]
    ]
    assembler = VectorAssembler(
        inputCols=["float0", "Gender1Index", "float2", "Country3Index", "float4"],
        outputCol="features"
    )
    classifier = catboost_spark.CatBoostRegressor(labelCol="Label", iterations=20)

    pipeline = Pipeline(stages=indexers + [assembler, classifier])
    pipelineModel = pipeline.fit(df)

    serializationDir = tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())

    modelPath = os.path.join(serializationDir, "serialized_pipeline_model")

    pipelineModel.write().overwrite().save(modelPath)
    loadedPipelineModel = PipelineModel.load(modelPath)

    print ("predictions")
    pipelineModel.transform(df).show(truncate=False)

    print ("predictionsLoaded")
    loadedPipelineModel.transform(df).show(truncate=False)

    shutil.rmtree(serializationDir)
