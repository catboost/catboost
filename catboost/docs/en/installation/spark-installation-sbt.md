# For sbt projects

{% include [installation-spark__dep-versions](../_includes/work_src/reusage-installation/spark__dep-versions.md) %}

Add dependency to `build.sbt` of your project:

```scala
libraryDependencies ++= Seq(
  "ai.catboost" %% ("catboost-spark_" + sparkCompatVersion) +  % "catboost_spark_version"
)
```
