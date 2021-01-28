import collections
import os
import shutil
import tempfile

import test_helpers
import pool_test_helpers

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row
from pyspark.sql.types import *


def testBinaryClassificationSimpleOnDataFrame():
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
        Row(Vectors.dense(0.1, 0.2, 0.11), 1.0),
        Row(Vectors.dense(0.97, 0.82, 0.33), 2.0),
        Row(Vectors.dense(0.13, 0.22, 0.23), 2.0),
        Row(Vectors.dense(0.14, 0.18, 0.1), 1.0),
        Row(Vectors.dense(0.9, 0.67, 0.17), 2.0),
        Row(Vectors.dense(0.66, 0.1, 0.31), 1.0)
    ]

    df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = classifier.fit(df)
    predictions = model.transform(df)

    print ("predictions")
    predictions.show(truncate=False)


def testSimpleBinaryClassification():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType()),
        ("groupId", LongType()),
        ("groupWeight", FloatType()),
        ("subgroupId", IntegerType()),
        ("weight", FloatType())
    ]

    srcData = [
        Row(Vectors.dense(0.1, 0.2, 0.11), "0", 0xB337C6FEFE2E2F7, 1.0, 0xD34BFBD, 0.12),
        Row(Vectors.dense(0.97, 0.82, 0.33), "0", 0xB337C6FEFE2E2F7, 1.0, 0x19CE5B0, 0.18),
        Row(Vectors.dense(0.13, 0.22, 0.23), "1", 0x86F1B93B695F9E6, 0.0, 0x23D794E, 1.0),
        Row(Vectors.dense(0.14, 0.18, 0.1), "1", 0xD9DBDD3199D6518, 0.5, 0x62772D1, 0.45),
        Row(Vectors.dense(0.9, 0.67, 0.17), "0", 0xD9DBDD3199D6518, 0.5, 0x19CE5B0, 1.0),
        Row(Vectors.dense(0.66, 0.1, 0.31), "1", 0xD9DBDD3199D6518, 0.5, 0x1FA606F, 2.0)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {"groupId": "groupId", "groupWeight": "groupWeight", "subgroupId": "subgroupId", "weight": "weight"}
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))

    model = classifier.fit(pool)
    predictions = model.transform(pool.data)

    for rawPrediction in [False, True]:
        for probability in [False, True]:
            for prediction in [False, True]:
                model.setRawPredictionCol("rawPrediction" if (rawPrediction) else "")
                model.setProbabilityCol("probability" if (probability) else "")
                model.setPredictionCol("prediction" if (prediction) else "")
                predictions = model.transform(pool.data)

                print('\nrawPrediction=%s, probability=%s, prediction=%s' % (rawPrediction, probability, prediction))
                predictions.show(truncate=False)


def testBinaryClassificationWithClassNamesAsIntSet():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", StringType())
    ]

    srcData = [
        Row(Vectors.dense(0.1, 0.2, 0.11), "1"),
        Row(Vectors.dense(0.97, 0.82, 0.33), "2"),
        Row(Vectors.dense(0.13, 0.22, 0.23), "2"),
        Row(Vectors.dense(0.14, 0.18, 0.1), "1"),
        Row(Vectors.dense(0.9, 0.67, 0.17), "2"),
        Row(Vectors.dense(0.66, 0.1, 0.31), "1")
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {}
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setClassNames(["1", "2"])
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))

    model = classifier.fit(pool)
    predictions = model.transform(pool.data)
    predictions.show(truncate=False)


def testBinaryClassificationWithTargetBorder():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", DoubleType())
    ]

    srcData = [
        Row(Vectors.dense(0.1, 0.2, 0.11), 0.12),
        Row(Vectors.dense(0.97, 0.82, 0.33), 0.1),
        Row(Vectors.dense(0.13, 0.22, 0.23), 0.7),
        Row(Vectors.dense(0.14, 0.18, 0.1), 0.33),
        Row(Vectors.dense(0.9, 0.67, 0.17), 0.82),
        Row(Vectors.dense(0.66, 0.1, 0.31), 0.93)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {}
    )

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setTargetBorder(0.5)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))

    model = classifier.fit(pool)
    predictions = model.transform(pool.data)
    predictions.show(truncate=False)


# Good
def testBinaryClassificationWithClassWeightsMap():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    featureNames = ["f1", "f2", "f3"]

    srcSchemaData = [
        ("features", VectorUDT()),
        ("label", IntegerType())
    ]

    srcData = [
          Row(Vectors.dense(0.1, 0.2, 0.11), 0),
          Row(Vectors.dense(0.97, 0.82, 0.33), 1),
          Row(Vectors.dense(0.13, 0.22, 0.23), 1),
          Row(Vectors.dense(0.14, 0.18, 0.1), 0),
          Row(Vectors.dense(0.9, 0.67, 0.17), 0),
          Row(Vectors.dense(0.66, 0.1, 0.31), 0)
    ]
    pool = pool_test_helpers.createRawPool(
        test_helpers.getCurrentMethodName,
        pool_test_helpers.createSchema(
          srcSchemaData,
          featureNames,
          addFeatureNamesMetadata=True
        ),
        srcData,
        {}
    )

    classWeightsMap = collections.OrderedDict([("0", 1.0), ("1", 2.0)])

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setClassWeightsMap(classWeightsMap)
      .setLoggingLevel(catboost_spark.ELoggingLevel.Debug)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))

    model = classifier.fit(pool)
    predictions = model.transform(pool.data)
    predictions.show(truncate=False)


def testClassifierSerialization():
    spark = test_helpers.getOrCreateSparkSession(test_helpers.getCurrentMethodName())
    import catboost_spark

    serializationDir = tempfile.mkdtemp(prefix="catboost_models_")

    path = os.path.join(serializationDir, "serialized_classifier_0")

    classifier = catboost_spark.CatBoostClassifier()
    classifier.write().overwrite().save(path)
    loadedClassifier = catboost_spark.CatBoostClassifier.load(path)

    path = os.path.join(serializationDir, "serialized_classifier_1")

    classifier = (catboost_spark.CatBoostClassifier().setLossFunction("MultiClass").setIterations(2))
    classifier.write().overwrite().save(path)
    loadedClassifier = catboost_spark.CatBoostClassifier.load(path)

    shutil.rmtree(serializationDir)


def testModelSerialization():
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
        Row(Vectors.dense(0.1, 0.2, 0.11), 1.0),
        Row(Vectors.dense(0.97, 0.82, 0.33), 2.0),
        Row(Vectors.dense(0.13, 0.22, 0.23), 2.0),
        Row(Vectors.dense(0.14, 0.18, 0.1), 1.0),
        Row(Vectors.dense(0.9, 0.67, 0.17), 2.0),
        Row(Vectors.dense(0.66, 0.1, 0.31), 1.0)
    ]

    df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema))

    classifier = (catboost_spark.CatBoostClassifier()
      .setIterations(20)
      .setTrainDir(tempfile.mkdtemp(prefix=test_helpers.getCurrentMethodName())))
    model = classifier.fit(df)
    predictions = model.transform(df)

    print ("predictions")
    predictions.show(truncate=False)

    modelsDir = tempfile.mkdtemp(prefix="catboost_models_")

    nativeCatBoostModelPath = os.path.join(modelsDir, "binclass_model_on_df.cbm")
    model.saveNativeModel(nativeCatBoostModelPath)

    loadedCatBoostModel = catboost_spark.CatBoostClassificationModel.loadNativeModel(nativeCatBoostModelPath)
    predictionsLoadedCatBoost = loadedCatBoostModel.transform(df)
    print ("predictionsLoadedCatBoost")
    predictionsLoadedCatBoost.show(truncate=False)

    nativeJsonModelPath =  os.path.join(modelsDir, "binclass_model_on_df.json")
    model.saveNativeModel(nativeJsonModelPath, catboost_spark.EModelType.Json)

    nativeOnnxModelPath =  os.path.join(modelsDir, "binclass_model_on_df.onnx")
    model.saveNativeModel(
      nativeOnnxModelPath,
      catboost_spark.EModelType.Onnx,
      {
        "onnx_domain": "ai.catboost",
        "onnx_model_version": 1,
        "onnx_doc_string": "test model for classification",
        "onnx_graph_name": "CatBoostModel_for_classification"
      }
    )

    loadedOnnxModel = catboost_spark.CatBoostClassificationModel.loadNativeModel(nativeOnnxModelPath, catboost_spark.EModelType.Onnx)
    predictionsLoadedOnnx = loadedOnnxModel.transform(df)
    print ("predictionsLoadedOnnx")
    predictionsLoadedOnnx.show(truncate=False)

    sparkModelPath = os.path.join(modelsDir, "binclass_model_on_df")
    model.write().overwrite().save(sparkModelPath)
    loadedModel = catboost_spark.CatBoostClassificationModel.load(sparkModelPath)

    predictionsLoaded = loadedModel.transform(df)
    print ("predictionsLoaded")
    predictionsLoaded.show(truncate=False)

    shutil.rmtree(modelsDir)
