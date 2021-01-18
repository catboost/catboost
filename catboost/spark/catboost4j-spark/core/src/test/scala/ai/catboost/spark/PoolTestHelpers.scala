package ai.catboost.spark;

import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg._;
import org.apache.spark.sql._;
import org.apache.spark.sql.types._;

import org.junit.{Assert};

object PoolTestHelpers {
    def createSchema(
      schemaDesc: Seq[(String,DataType)],
      featureNames: Seq[String],
      addFeatureNamesMetadata: Boolean = true,
      nullableFields: Seq[String] = Seq(),
      catFeaturesNumValues: Map[String,Int] = Map[String,Int](),
      catFeaturesValues: Map[String,Seq[String]] = Map[String,Seq[String]]()
    ) : Seq[StructField] = {
      schemaDesc.map {
        case (name, dataType) if (addFeatureNamesMetadata && ((name == "features") || (name == "f1"))) => {
          val attrs = featureNames.map(
            name => {
              if (catFeaturesNumValues.contains(name)) {
                NominalAttribute.defaultAttr.withName(name).withNumValues(catFeaturesNumValues(name))
              } else if (catFeaturesValues.contains(name)) {
                NominalAttribute.defaultAttr.withName(name).withValues(catFeaturesValues(name).toArray)
              } else {
                NumericAttribute.defaultAttr.withName(name)
              }
            }
          ).toArray
          val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

          StructField(name, dataType, nullableFields.contains(name), attrGroup.toMetadata)
        }
        case (name, dataType) => StructField(name, dataType, nullableFields.contains(name))
      }
    }

    def writeToTempFile(content: String) : Path = {
      val tempFile = Files.createTempFile("PoolTest", "")
      tempFile.toFile.deleteOnExit
      Files.write(tempFile, content.getBytes(StandardCharsets.UTF_8))
      tempFile
    }

    @throws(classOf[java.lang.AssertionError])
    def comparePoolWithExpectedData(
        pool: Pool,
        expectedDataSchema: Seq[StructField],
        expectedData: Seq[Row],
        expectedFeatureNames: Array[String],
        expectedPairsData: Option[Seq[Row]] = None,
        expectedPairsDataSchema: Option[Seq[StructField]] = None,
        
         // set to true if order of rows might change. Requires sampleId or (groupId, sampleId) in data
        compareByIds: Boolean = false
    ) = {
      val spark = pool.data.sparkSession

      val expectedDf = spark.createDataFrame(
        spark.sparkContext.parallelize(expectedData),
        StructType(expectedDataSchema)
      )
      
      Assert.assertTrue(pool.getFeatureNames.sameElements(expectedFeatureNames))
      
      Assert.assertEquals(pool.data.schema, StructType(expectedDataSchema))
      
      val (expectedDataToCompare, poolDataToCompare) = if (compareByIds) {
        if (pool.isDefined(pool.groupIdCol)) {
          (
            expectedDf.orderBy(pool.getOrDefault(pool.groupIdCol), pool.getOrDefault(pool.sampleIdCol)),
            pool.data.orderBy(pool.getOrDefault(pool.groupIdCol), pool.getOrDefault(pool.sampleIdCol))
          )
        } else {
          (
            expectedDf.orderBy(pool.getOrDefault(pool.sampleIdCol)),
            pool.data.orderBy(pool.getOrDefault(pool.sampleIdCol))
          )
        }
      } else {
        (expectedDf, pool.data)
      }

      TestHelpers.assertEqualsWithPrecision(expectedDataToCompare, poolDataToCompare)
      
      
      expectedPairsData match {
        case Some(expectedPairsData) => {
          val poolPairsDataToCompare = pool.pairsData.orderBy("groupId", "winnerId", "loserId")
          
          val expectedPairsDataToCompare = spark.createDataFrame(
            spark.sparkContext.parallelize(expectedPairsData),
            StructType(expectedPairsDataSchema.get)
          ).orderBy("groupId", "winnerId", "loserId")
          
          Assert.assertTrue(pool.pairsData != null)
          TestHelpers.assertEqualsWithPrecision(expectedPairsDataToCompare, poolPairsDataToCompare)
        }
        case None => Assert.assertTrue(pool.pairsData == null)
      }
    }

    def createRawPool(
      appName: String,
      srcDataSchema : Seq[StructField],
      srcData: Seq[Row], 
      columnNames: Map[String, String] // standard column name to name of column in the dataset
    ) : Pool = {
      val spark = TestHelpers.getOrCreateSparkSession(appName)

      val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema));

      var pool = new Pool(df)
      
      if (columnNames.contains("features")) {
        pool = pool.setFeaturesCol(columnNames("features"))
      }
      if (columnNames.contains("groupId")) {
        pool = pool.setGroupIdCol(columnNames("groupId"))
      }
      if (columnNames.contains("subgroupId")) {
        pool = pool.setSubgroupIdCol(columnNames("subgroupId"))
      }
      if (columnNames.contains("weight")) {
        pool = pool.setWeightCol(columnNames("weight"))
      }
      if (columnNames.contains("groupWeight")) {
        pool = pool.setGroupWeightCol(columnNames("groupWeight"))
      }
      pool
    }
}
