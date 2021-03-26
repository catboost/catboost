CatBoost for Apache Spark Package
======================

Table of contents:

* [Setup](#setup)
  * [Requirements](#requirements)
  * [Java or Scala](#java-or-scala)
  * [Python (PySpark)](#python-pyspark)
* [Spark Cluster Configuration](#spark-cluster-configuration)
* [Examples](#examples)
  * [Scala](#scala)
    * [Classification](#classification)
      * [Binary classification](#binary-classification)
      * [Multiclassification](#multiclassification)
    * [Regression](#regression)
  * [PySpark](#pyspark)
    * [Classification](#classification-1)
      * [Binary classification](#binary-classification-1)
      * [Multiclassification](#multiclassification-1)
    * [Regression](#regression-1)
* [Documentation](#documentation)
* [Known limitations](#known-limitations)
* [Build from source](#build-from-source)

Setup 
------------

#### Requirements

* Linux or Mac OS X. Windows support in progress.
* Apache Spark 2.3 - 3.0
* Scala 2.11 or 2.12

Get the appropriate `catboost_spark_version` (you can look up available versions at [Maven central](https://search.maven.org/search?q=catboost-spark))

Add dependency with the appropriate `spark_compat_version` (`2.3`, `2.4` or `3.0`) and `scala_compat_version` (`2.11` or `2.12`)

#### Java or Scala

Maven or sbt-based build system is supported.

* Maven

Add to pom.xml of your project :

```
  <properties>
    ...
    <spark.compat.version>spark_compat_version</spark.compat.version>
    <scala.compat.version>scala_compat_version</scala.compat.version>
    ...
  </properties>
  
  <dependencies>
    ...
    <dependency>
      <groupId>ai.catboost</groupId>
      <artifactId>catboost-spark_${spark.compat.version}_${scala.compat.version}</artifactId>
      <version>catboost_spark_version</version>
    </dependency>
   ...
```

* sbt

```
libraryDependencies ++= Seq(
  "ai.catboost" %% ("catboost-spark_" + sparkCompatVersion) +  % "catboost_spark_version"
)
```

#### Python (PySpark)

Just add `catboost-spark` Maven artifact with appropriate `spark_compat_version`, `scala_compat_version` and `catboost_spark_version` to `spark.jar.packages` Spark config parameter and import `catboost_spark` package:

```python
from pyspark.sql import SparkSession

sparkSession = (SparkSession.builder
	.master(...)
	.config("spark.jars.packages", "ai.catboost:catboost-spark_<spark_compat_version>_<scala_compat_version>:<catboost_spark_version>")
	.getOrCreate()
)

import catboost_spark

...

```

Spark Cluster Configuration
---------------------------

CatBoost for Apache Spark requires one training task per executor, so if you run training you have to set
`spark.task.cpus` parameter to be equal to the number of cores in executors (`spark.executor.cores`).
This limitation might be relaxed in the future ([the corresponding issue #1622](https://github.com/catboost/catboost/issues/1622)).

Model application or feature importance evaluation do not have this limitation.


Examples
--------

### Scala

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
println("predictions")
predictions.show()

// save model
val savedModelPath = "/my_models/binclass_model"
model.write.save(savedModelPath)

// save model as local file in CatBoost native format
val savedNativeModelPath = "./my_local_models/binclass_model.cbm"
model.saveNativeModel(savedNativeModelPath)

...

// load model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
println("predictionsFromLoadedModel")
predictionsFromLoadedModel.show()


// load model as local file in CatBoost native format

val loadedNativeModel = CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
println("predictionsFromLoadedNativeModel")
predictionsFromLoadedNativeModel.show()
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
println("predictions")
predictions.show()

// save model
val savedModelPath = "/my_models/multiclass_model"
model.write.save(savedModelPath)

// save model as local file in CatBoost native format
val savedNativeModelPath = "./my_local_models/multiclass_model.cbm"
model.saveNativeModel(savedNativeModelPath)

...

// load model (can be used in a different Spark session)

val loadedModel = CatBoostClassificationModel.load(savedModelPath)

val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
println("predictionsFromLoadedModel")
predictionsFromLoadedModel.show()

// load model as local file in CatBoost native format

val loadedNativeModel = CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
println("predictionsFromLoadedNativeModel")
predictionsFromLoadedNativeModel.show()
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
println("predictions")
predictions.show()

// save model
val savedModelPath = "/my_models/regression_model"
model.write.save(savedModelPath)

// save model as local file in CatBoost native format
val savedNativeModelPath = "./my_local_models/regression_model.cbm"
model.saveNativeModel(savedNativeModelPath)

...

// load model (can be used in a different Spark session)

val loadedModel = CatBoostRegressionModel.load(savedModelPath)

val predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
println("predictionsFromLoadedModel")
predictionsFromLoadedModel.show()

// load model as local file in CatBoost native format

val loadedNativeModel = CatBoostRegressionModel.loadNativeModel(savedNativeModelPath)

val predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
println("predictionsFromLoadedNativeModel")
predictionsFromLoadedNativeModel.show()

```

Refer to other usage examples in the scaladoc-generated documentation provided in the package.

### PySpark

#### Classification:

##### Binary classification:

```python
from pyspark.sql import Row,SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import *


spark = (SparkSession.builder
  .master("local[*]")
  .config("spark.jars.packages", "ai.catboost:catboost-spark_2.4_2.12:0.25")
  .appName("ClassifierTest")
  .getOrCreate()
)
import catboost_spark
    
srcDataSchema = [
    StructField("features", VectorUDT()),
    StructField("label", StringType())
]

trainData = [
    Row(Vectors.dense(0.1, 0.2, 0.11), "0"),
    Row(Vectors.dense(0.97, 0.82, 0.33), "1"),
    Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
    Row(Vectors.dense(0.8, 0.62, 0.0), "0")
]
    
trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
trainPool = catboost_spark.Pool(trainDf)
    
evalData = [
    Row(Vectors.dense(0.22, 0.33, 0.9), "1"),
    Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
    Row(Vectors.dense(0.77, 0.0, 0.0), "1")
]
    
evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
evalPool = catboost_spark.Pool(evalDf)
    
classifier = catboost_spark.CatBoostClassifier()
    
# train model
model = classifier.fit(trainPool, [evalPool])

# apply model
predictions = model.transform(evalPool.data)
predictions.show()

# save model
savedModelPath = "/my_models/binclass_model"
model.write().save(savedModelPath)

# save model as local file in CatBoost native format
savedNativeModelPath = './my_local_models/binclass_model.cbm'
model.saveNativeModel(savedNativeModelPath)


# load model (can be used in a different Spark session)

loadedModel = catboost_spark.CatBoostClassificationModel.load(savedModelPath)

predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
predictionsFromLoadedModel.show()


# load model as local file in CatBoost native format

loadedNativeModel = catboost_spark.CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
predictionsFromLoadedNativeModel.show()
```

##### Multiclassification:

```python
from pyspark.sql import Row,SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import *


spark = (SparkSession.builder
  .master("local[*]")
  .config("spark.jars.packages", "ai.catboost:catboost-spark_2.11:0.25-rc3")
  .appName("ClassifierTest")
  .getOrCreate()
)
import catboost_spark
    
srcDataSchema = [
    StructField("features", VectorUDT()),
    StructField("label", StringType())
]

trainData = [
    Row(Vectors.dense(0.1, 0.2, 0.11), "1"),
    Row(Vectors.dense(0.97, 0.82, 0.33), "2"),
    Row(Vectors.dense(0.13, 0.22, 0.23), "1"),
    Row(Vectors.dense(0.8, 0.62, 0.0), "0")
]
    
trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
trainPool = catboost_spark.Pool(trainDf)
    
evalData = [
    Row(Vectors.dense(0.22, 0.33, 0.9), "2"),
    Row(Vectors.dense(0.11, 0.1, 0.21), "0"),
    Row(Vectors.dense(0.77, 0.0, 0.0), "1")
]
    
evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
evalPool = catboost_spark.Pool(evalDf)
    
classifier = catboost_spark.CatBoostClassifier()
    
# train model
model = classifier.fit(trainPool, [evalPool])

# apply model
predictions = model.transform(evalPool.data)
predictions.show()

# save model
savedModelPath = "/my_models/multiclass_model"
model.write().save(savedModelPath)

# save model as local file in CatBoost native format
savedNativeModelPath = './my_local_models/multiclass_model.cbm'
model.saveNativeModel(savedNativeModelPath)


# load model (can be used in a different Spark session)

loadedModel = catboost_spark.CatBoostClassificationModel.load(savedModelPath)

predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
predictionsFromLoadedModel.show()


# load model as local file in CatBoost native format

loadedNativeModel = catboost_spark.CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
predictionsFromLoadedNativeModel.show()
```

#### Regression:

```python
from pyspark.sql import Row,SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import *


spark = (SparkSession.builder
  .master("local[*]")
  .config("spark.jars.packages", "ai.catboost:catboost-spark_2.4_2.12:0.25")
  .appName("RegressorTest")
  .getOrCreate()
)
import catboost_spark
    
srcDataSchema = [
    StructField("features", VectorUDT()),
    StructField("label", DoubleType())
]

trainData = [
    Row(Vectors.dense(0.1, 0.2, 0.11), 0.12),
    Row(Vectors.dense(0.97, 0.82, 0.33), 0.22),
    Row(Vectors.dense(0.13, 0.22, 0.23), 0.34),
    Row(Vectors.dense(0.8, 0.62, 0.0), 0.1)
]
    
trainDf = spark.createDataFrame(spark.sparkContext.parallelize(trainData), StructType(srcDataSchema))
trainPool = catboost_spark.Pool(trainDf)
    
evalData = [
    Row(Vectors.dense(0.22, 0.33, 0.9), 0.1),
    Row(Vectors.dense(0.11, 0.1, 0.21), 0.9),
    Row(Vectors.dense(0.77, 0.0, 0.0), 0.72)
]
    
evalDf = spark.createDataFrame(spark.sparkContext.parallelize(evalData), StructType(srcDataSchema))
evalPool = catboost_spark.Pool(evalDf)
    
regressor = catboost_spark.CatBoostRegressor()
    
# train model
model = regressor.fit(trainPool, [evalPool])

# apply model
predictions = model.transform(evalPool.data)
predictions.show()

# save model
savedModelPath = "/my_models/regression_model"
model.write().save(savedModelPath)

# save model as local file in CatBoost native format
savedNativeModelPath = './my_local_models/regression_model.cbm'
model.saveNativeModel(savedNativeModelPath)


# load model (can be used in a different Spark session)

loadedModel = catboost_spark.CatBoostRegressionModel.load(savedModelPath)

predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
predictionsFromLoadedModel.show()


# load model as local file in CatBoost native format

loadedNativeModel = catboost_spark.CatBoostRegressionModel.loadNativeModel(savedNativeModelPath)

predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
predictionsFromLoadedNativeModel.show()

```

Documentation
-------------
- See scaladoc-generated documentation provided in the package.
- Training parameters are described in detail in [the documentation for python package](https://catboost.ai/docs/concepts/python-reference_parameters-list.html). Training parameters for Spark have the same meaning.
- See tests in `core/src/test/scala` and `core/src/test/python` subdirectories for more usage examples.

Known limitations
-----------------

* Windows is not supported. Work in progress.
* GPU is not supported. Work in progress.
* Text and embeddings features are not supported. Work in progress.
* Feature distribution statistics (like `calc_feature_statistfzics`on CatBoost python package) with datasets on Spark is not supported. But it is possible to run such analysis with models exported to local files in usual CatBoost format.
* Generic string class labels are not supported. String class labels can be used only if these strings represent integer indices.
* ``boosting_type=Ordered`` is not supported.
* Training of models with non-symmetric trees is not supported. But such models can be loaded and applied on datasets in Spark.
* Monotone constraints are not supported.
* Multitarget training is not supported.
* Stochastic Gradient Langevin Boosting mode is not supported.

Build from source
--------

#### Dependencies and requirements

* Linux or Mac OS X. Windows support in progress.
* Python. 2.7 or 3.2+
* Maven 3.3.9+
* JDK 8. Newer versions of JDK are not supported yet.
* SWIG 4.0.2+

#### Building steps

* Clone the repository:

```
git clone https://github.com/catboost/catboost.git
```

* Open the `catboost/catboost/spark/catboost4j-spark` directory from the local copy of the CatBoost repository.
* Use [usual maven build phases](https://maven.apache.org/guides/introduction/introduction-to-the-lifecycle.html)
