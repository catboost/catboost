# {{ catboost-spark }} Documentation

Main features
-------------

* Support for both Numerical and Categorical (as both One-hot and CTRs) features.
* Reproducible training results.
* Model interoperability with local CatBoost implementations.
* Distributed feature evaluation (including SHAP values).
* Spark MLLib compatible APIs for JVM languages (Java, Scala, Kotlin etc.) and PySpark.
* Extended Apache Spark versions support: 3.0 to 3.5.

  {% cut "Previous versions" %}

  CatBoost versions before 1.2.8 supported Apache Spark versions 2.3 - 2.4 as well.

  {% endcut %}


### [{{ catboost-spark }} installation](spark-installation.md)
### Quick start for [Scala](spark-quickstart-scala.md) and [Python](spark-quickstart-python.md)
### [Spark cluster configuration](spark-cluster-configuration.md)
### [API documentation](spark-api-documentation.md)
### [Known limitations](spark-known-limitations.md)
