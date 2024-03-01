# For Maven projects

Get the appropriate `catboost_spark_version` (see available versions at [Maven central](https://search.maven.org/search?q=catboost-spark)).

Choose the appropriate `spark_compat_version` (`2.3`, `2.4`, `3.0`, `3.1`, `3.2`, `3.3`, `3.4` or `3.5`) and `scala_compat_version` (`2.11`, `2.12` or `2.13`, corresponding to versions supported by the particular Spark version).

Add to the following to `pom.xml` of your project :

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
