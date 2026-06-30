# For PySpark

{% include [installation-spark__dep-versions](../_includes/work_src/reusage-installation/spark__dep-versions.md) %}

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
