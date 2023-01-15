CatBoost Spark Package
======================

Installation
------------

####Requirements

* Linux or Mac OS X. Windows support in progress.
* Apache Spark 2.4+
* Scala 2.11 or 2.12
* Maven or SBT

Get the appropriate `catboost_spark_version` (you can look up available versions at [Maven central](https://search.maven.org/search?q=catboost-spark))

Add dependency with the appropriate `scala_compat_version` (`2.11` or `2.12`):

* Maven

Add to pom.xml of your project :

```
  <properties>
    ...
    <scala.compat.version>scala_compat_version</scala.compat.version>
    ...
  </properties>
  
  <dependencies>
    ...
    <dependency>
      <groupId>ai.catboost</groupId>
      <artifactId>catboost-spark_${scala.compat.version}</artifactId>
      <version>catboost_spark_version</version>
    </dependency>
   ...
```

* sbt

```
libraryDependencies ++= Seq(
  "ai.catboost" %% "catboost-spark" % "catboost_spark_version"
)
```


Examples
--------
Quick examples:

#### Classification:

##### Binary classification:

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

// train model
val model = classifier.fit(trainPool, Array[Pool](evalPool))

// apply model
val predictions = model.transform(evalPool.data)
predictions.show()

// save model
val savedModelPath = "/my_models/binclass_model"
model.write.save(savedModelPath)

...

// load model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictions2 = loadedModel.transform(evalPool.data)
predictions2.show()
```

##### Multiclassification:

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

// train model
val model = classifier.fit(trainPool, Array[Pool](evalPool))

// apply model
val predictions = model.transform(evalPool.data)
predictions.show()

// save model
val savedModelPath = "/my_models/multiclass_model"
model.write.save(savedModelPath)

...

// load model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictions2 = loadedModel.transform(evalPool.data)
predictions2.show()
```

#### Regression:

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

// train model
val model = regressor.fit(trainPool, Array[Pool](evalPool))

// apply model
val predictions = model.transform(evalPool.data)
predictions.show()

// save model
val savedModelPath = "/my_models/regression_model"
model.write.save(savedModelPath)

...

// load model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictions2 = loadedModel.transform(evalPool.data)
predictions2.show()

```

Refer to other usage examples in the scaladoc-generated documentation provided in the package. 


Documentation
-------------
- See scaladoc-generated documentation provided in the package.
- Training parameters are described in detail in [the documentation for python package](https://catboost.ai/docs/concepts/python-reference_parameters-list.html). Training parameters for Spark have the same meaning.

Known limitations
-----------------

* Windows is not supported. Work in progress.
* PySpark is not supported. Work in progress.
* GPU is not supported. Work in progress.
* Categorical features are not supported. Work in progress.
* Text features are not supported. Work in progress.
* Model analysis like feature importance and feature statistics with datasets on Spark is not supported. But it is possible to run such analysis with models exported to local files in usual CatBoost format.
* Model export in non-default formats is not supported. But it is possible to convert models exported to local files in usual CatBoost format using other CatBoost APIs.
* Generic string class labels are not supported. String class labels can be used only if these strings represent integer indices.
* ``boosting_type=Ordered`` is not supported.
* Training of models with non-symmetric trees is not supported. But such models can be loaded and applied on datasets in Spark.
* Monotone constraints are not supported.
* Multitarget training is not supported.
* Stochastic Gradient Langevin Boosting mode is not supported.
* Training with pairs is not supported.

Build from source
--------

####Dependencies and requirements

* Linux or Mac OS X. Windows support in progress.
* Python. 2.7 or 3.2+
* Maven 3.3.9+
* JDK 8. Newer versions of JDK are not supported yet.
* SWIG 4.0.2+

####Building steps

* Clone the repository:

```
git clone https://github.com/catboost/catboost.git
```

* Open the `catboost/catboost/spark/catboost4j-spark` directory from the local copy of the CatBoost repository.
* Use [usual maven build phases](https://maven.apache.org/guides/introduction/introduction-to-the-lifecycle.html)
