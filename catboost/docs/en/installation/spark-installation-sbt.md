# For sbt projects

Get the appropriate `catboost_spark_version` (see available versions at [Maven central](https://search.maven.org/search?q=catboost-spark))

Choose the appropriate `spark_compat_version` (`2.3`, `2.4` or `3.0`) and `scala_compat_version` (`2.11` or `2.12`).

Add dependency to `build.sbt` of your project:

```scala
libraryDependencies ++= Seq(
  "ai.catboost" %% ("catboost-spark_" + sparkCompatVersion) +  % "catboost_spark_version"
)
```
