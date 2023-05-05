# For PySpark

Get the appropriate `catboost_spark_version` (see available versions at [Maven central](https://search.maven.org/search?q=catboost-spark)).

Choose the appropriate `spark_compat_version` (`2.3`, `2.4`, `3.0`, `3.1`, `3.2`, `3.3` or `3.4`) and `scala_compat_version` (`2.11`, `2.12` or `2.13`, corresponding to versions supported by the particular Spark version).

Just add the `catboost-spark` Maven artifact with the appropriate `spark_compat_version`, `scala_compat_version` and `catboost_spark_version` to `spark.jar.packages` Spark config parameter and import the `catboost_spark` package:

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
