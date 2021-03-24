## Tests for catboost_spark for pyspark

Require available version of `catboost-spark_<spark_compat_version>_<scala_compat_version>` Maven artifact.

You can change the version of the artifact in `test_helpers.getOrCreateSparkSession` function.
You can also change other spark/pyspark parameters (change from `local` to proper Spark cluster for example).

Can be run using [pytest](https://pytest.org) like that:

```
pythonX.Y -m pytest
```
