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
      nullableFields: Seq[String] = Seq()
    ) : Seq[StructField] = {
      schemaDesc.map {
        case (name, dataType) if (addFeatureNamesMetadata && (name == "features")) => {
          val defaultAttr = NumericAttribute.defaultAttr
          val attrs = featureNames.map(defaultAttr.withName).toArray
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
          val poolPairsDataToCompare = pool.pairsData.orderBy("groupId", "winnerIdInGroup", "loserIdInGroup")
          
          val expectedPairsDataToCompare = spark.createDataFrame(
            spark.sparkContext.parallelize(expectedPairsData),
            StructType(expectedPairsDataSchema.get)
          ).orderBy("groupId", "winnerIdInGroup", "loserIdInGroup")
          
          Assert.assertTrue(pool.pairsData != null)
          TestHelpers.assertEqualsWithPrecision(expectedPairsDataToCompare, poolPairsDataToCompare)
        }
        case None => Assert.assertTrue(pool.pairsData == null)
      }
    }

}
