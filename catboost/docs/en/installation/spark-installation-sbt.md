# For sbt projects

Get the appropriate `catboost_spark_version` (see available versions at [Maven central](https://search.maven.org/search?q=catboost-spark))

Choose the appropriate `spark_compat_version` (`2.3`, `2.4`, `3.0`, `3.1`, `3.2`, `3.3` or `3.4`) and `scala_compat_version` (`2.11`, `2.12` or `2.13`, corresponding to versions supported by the particular Spark version).

Add dependency to `build.sbt` of your project:

```scala
libraryDependencies ++= Seq(
  "ai.catboost" %% ("catboost-spark_" + sparkCompatVersion) +  % "catboost_spark_version"
)
```
