# For Maven projects

{% include [installation-spark__dep-versions](../_includes/work_src/reusage-installation/spark__dep-versions.md) %}

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
