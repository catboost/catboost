# Quick start for Scala

Perform the following steps:

1. [Install {{ catboost-spark }} package](spark-installation.md).
2. Make sure [Spark cluster is configured properly](spark-cluster-configuration.md).

Use one of the following examples:
- [Classification](#classification)
    - [Binary classification](#binary-classification)
    - [Multiclassification](#multi-classification)
- [Regression](#regression)

## Classification

### Binary classification {#binary-classification}

```scala
import org.apache.spark.sql.{Row,SparkSession}
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._

import ai.catboost.spark._

...

val spark = SparkSession.builder()
  .master("local[*]")
  .appName("ClassifierTest")
  .getOrCreate();

val srcDataSchema = Seq(
  StructField("features", SQLDataTypes.VectorType),
  StructField("label", StringType)
)

val trainData = Seq(
  Row(Vectors.dense(0.1, 0.2, 0.11), "0"),
  Row(Vectors.dense(0.97, 0.82, 0.33), "1"),
  Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
  Row(Vectors.dense(0.8, 0.62, 0.0), "0")
)

val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
val trainPool = new Pool(trainDf)

val evalData = Seq(
  Row(Vectors.dense(0.22, 0.33, 0.9), "1"),
  Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
  Row(Vectors.dense(0.77, 0.0, 0.0), "1")
)

val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
val evalPool = new Pool(evalDf)

val classifier = new CatBoostClassifier

// train a model
val model = classifier.fit(trainPool, Array[Pool](evalPool))

// apply the model
val predictions = model.transform(evalPool.data)
println("predictions")
predictions.show()

// save the model
val savedModelPath = "/my_models/binclass_model"
model.write.save(savedModelPath)

// save the model as a local file in CatBoost native format
val savedNativeModelPath = "./my_local_models/binclass_model.cbm"
model.saveNativeModel(savedNativeModelPath)

...

// load the model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
println("predictionsFromLoadedModel")
predictionsFromLoadedModel.show()


// load the model as a local file in CatBoost native format

val loadedNativeModel = CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
println("predictionsFromLoadedNativeModel")
predictionsFromLoadedNativeModel.show()
```

### Multiclassification {#multi-classification}

```scala
import org.apache.spark.sql.{Row,SparkSession}
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._

import ai.catboost.spark._

...

val spark = SparkSession.builder()
  .master("local[*]")
  .appName("ClassifierTest")
  .getOrCreate();

val srcDataSchema = Seq(
  StructField("features", SQLDataTypes.VectorType),
  StructField("label", StringType)
)

val trainData = Seq(
  Row(Vectors.dense(0.1, 0.2, 0.11), "1"),
  Row(Vectors.dense(0.97, 0.82, 0.33), "2"),
  Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
  Row(Vectors.dense(0.8, 0.62, 0.0), "0")
)

val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
val trainPool = new Pool(trainDf)

val evalData = Seq(
  Row(Vectors.dense(0.22, 0.33, 0.9), "2"),
  Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
  Row(Vectors.dense(0.77, 0.0, 0.0), "1")
)

val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
val evalPool = new Pool(evalDf)

val classifier = new CatBoostClassifier

// train a model
val model = classifier.fit(trainPool, Array[Pool](evalPool))

// apply the model
val predictions = model.transform(evalPool.data)
println("predictions")
predictions.show()

// save the model
val savedModelPath = "/my_models/multiclass_model"
model.write.save(savedModelPath)

// save the model as a local file in CatBoost native format
val savedNativeModelPath = "./my_local_models/multiclass_model.cbm"
model.saveNativeModel(savedNativeModelPath)

...

// load the model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
println("predictionsFromLoadedModel")
predictionsFromLoadedModel.show()

// load the model as a local file in CatBoost native format

val loadedNativeModel = CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
println("predictionsFromLoadedNativeModel")
predictionsFromLoadedNativeModel.show()
```

## Regression {#regression}

```scala
import org.apache.spark.sql.{Row,SparkSession}
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._

import ai.catboost.spark._

...

val spark = SparkSession.builder()
  .master("local[*]")
  .appName("RegressorTest")
  .getOrCreate();

val srcDataSchema = Seq(
  StructField("features", SQLDataTypes.VectorType),
  StructField("label", StringType)
)

val trainData = Seq(
  Row(Vectors.dense(0.1, 0.2, 0.11), "0.12"),
  Row(Vectors.dense(0.97, 0.82, 0.33), "0.22"),
  Row(Vectors.dense(0.13, 0.22, 0.23), "0.34"),
  Row(Vectors.dense(0.8, 0.62, 0.0), "0.1")
)

val trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
val trainPool = new Pool(trainDf)

val evalData = Seq(
  Row(Vectors.dense(0.22, 0.33, 0.9), "0.1"),
  Row(Vectors.dense(0.11, 0.1, 0.21), "0.9"),
  Row(Vectors.dense(0.77, 0.0, 0.0), "0.72")
)

val evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
val evalPool = new Pool(evalDf)

val regressor = new CatBoostRegressor

// train a model
val model = regressor.fit(trainPool, Array[Pool](evalPool))

// apply the model
val predictions = model.transform(evalPool.data)
println("predictions")
predictions.show()

// save the model
val savedModelPath = "/my_models/regression_model"
model.write.save(savedModelPath)

// save the model as a local file in CatBoost native format
val savedNativeModelPath = "./my_local_models/regression_model.cbm"
model.saveNativeModel(savedNativeModelPath)

...

// load the model (can be used in a different Spark session)

val loadedModel = CatBoostRegressionModel.load(savedModelPath)

val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
println("predictionsFromLoadedModel")
predictionsFromLoadedModel.show()

// load the model as a local file in CatBoost native format

val loadedNativeModel = CatBoostRegressionModel.loadNativeModel(savedNativeModelPath)

val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
println("predictionsFromLoadedNativeModel")
predictionsFromLoadedNativeModel.show()

```
