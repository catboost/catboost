# Quick start for Python

1. [Choose the appropriate `catboost-spark` Maven artifact full name and version](../installation/spark-installation-pyspark.md)
2. Make sure [Spark cluster is configured properly](spark-cluster-configuration.md)

Use one of the following examples:
- [Classification](#classification)
    - [Binary classification](#binary-classification)
    - [Multiclassification](#multi-classification)
- [Regression](#regression)

## Classification {#classification}

### Binary classification {#binary-classification}

```python
from pyspark.sql import Row,SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import *


spark = (SparkSession.builder
  .master("local[*]")
  .config("spark.jars.packages", "ai.catboost:catboost-spark_3.1_2.12:1.0.6")
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
    
# train a model
model = classifier.fit(trainPool, eval_set=[evalPool])

# apply the model
predictions = model.transform(evalPool.data)
predictions.show()

# save the model
savedModelPath = "/my_models/binclass_model"
model.write().save(savedModelPath)

# save the model as a local file in CatBoost native format
savedNativeModelPath = './my_local_models/binclass_model.cbm'
model.saveNativeModel(savedNativeModelPath)


# load the model (can be used in a different Spark session)

loadedModel = catboost_spark.CatBoostClassificationModel.load(savedModelPath)

predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
predictionsFromLoadedModel.show()


# load the model as a local file in CatBoost native format

loadedNativeModel = catboost_spark.CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
predictionsFromLoadedNativeModel.show()
```

### Multiclassification {#multi-classification}

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
    
# train a model
model = classifier.fit(trainPool, eval_set=[evalPool])

# apply the model
predictions = model.transform(evalPool.data)
predictions.show()

# save the model
savedModelPath = "/my_models/multiclass_model"
model.write().save(savedModelPath)

# save the model as a local file in CatBoost native format
savedNativeModelPath = './my_local_models/multiclass_model.cbm'
model.saveNativeModel(savedNativeModelPath)


# load the model (can be used in a different Spark session)

loadedModel = catboost_spark.CatBoostClassificationModel.load(savedModelPath)

predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
predictionsFromLoadedModel.show()


# load the model as a local file in CatBoost native format

loadedNativeModel = catboost_spark.CatBoostClassificationModel.loadNativeModel(savedNativeModelPath)

predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
predictionsFromLoadedNativeModel.show()
```

## Regression {#regression}

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
    
# train a model
model = regressor.fit(trainPool, eval_set=[evalPool])

# apply the model
predictions = model.transform(evalPool.data)
predictions.show()

# save the model
savedModelPath = "/my_models/regression_model"
model.write().save(savedModelPath)

# save the model as a local file in CatBoost native format
savedNativeModelPath = './my_local_models/regression_model.cbm'
model.saveNativeModel(savedNativeModelPath)


# load the model (can be used in a different Spark session)

loadedModel = catboost_spark.CatBoostRegressionModel.load(savedModelPath)

predictionsFromLoadedModel = loadedModel.transform(evalPool.data)
predictionsFromLoadedModel.show()


# load the model as a local file in CatBoost native format

loadedNativeModel = catboost_spark.CatBoostRegressionModel.loadNativeModel(savedNativeModelPath)

predictionsFromLoadedNativeModel = loadedNativeModel.transform(evalPool.data)
predictionsFromLoadedNativeModel.show()

```

