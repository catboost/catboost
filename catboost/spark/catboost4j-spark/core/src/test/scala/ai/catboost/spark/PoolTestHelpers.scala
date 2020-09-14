package ai.catboost.spark;

import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg._;
import org.apache.spark.sql._;
import org.apache.spark.sql.types._;

import org.junit.{Assert};

object PoolTestHelpers {
    ensureNativeLibLoaded

    def createSchema(
      schemaDesc: Seq[(String,DataType)],
      featureNames: Seq[String],
      addFeatureNamesMetadata: Boolean = true,
      nullable: Boolean = false
    ) : Seq[StructField] = {
      schemaDesc.map {
        case (name, dataType) if (addFeatureNamesMetadata && (name == "features")) => {
          val defaultAttr = NumericAttribute.defaultAttr
          val attrs = featureNames.map(defaultAttr.withName).toArray
          val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

          StructField(name, dataType, nullable, attrGroup.toMetadata)
        }
        case (name, dataType) => StructField(name, dataType, nullable)
      }
    }

    def writeToTempFile(content: String) : Path = {
      val tempFile = Files.createTempFile("PoolTest", "")
      tempFile.toFile.deleteOnExit
      Files.write(tempFile, content.getBytes(StandardCharsets.UTF_8))
      tempFile
    }

    def assertEqualsWithPrecision(lhs: DataFrame, rhs: DataFrame) = {
      Assert.assertEquals(lhs.count, rhs.count)

      Assert.assertEquals(lhs.schema, rhs.schema)

      val schema = lhs.schema
      val rowSize = schema.size

      val lhsRows = lhs.collect()
      val rhsRows = rhs.collect()

      for (rowIdx <- 0 until lhsRows.size) {
        val lRow = lhsRows(rowIdx)
        val rRow = rhsRows(rowIdx)

        Assert.assertEquals(lRow.size, rowSize)
        Assert.assertEquals(rRow.size, rowSize)

        for (fieldIdx <- 0 until rowSize) {
          schema(fieldIdx).dataType match {
            case FloatType =>
              Assert.assertEquals(lRow.getAs[Float](fieldIdx), rRow.getAs[Float](fieldIdx), 1e-5f)
            case DoubleType =>
              Assert.assertEquals(lRow.getAs[Double](fieldIdx), rRow.getAs[Double](fieldIdx), 1e-6)
            case SQLDataTypes.VectorType =>
              Assert.assertArrayEquals(
                lRow.getAs[Vector](fieldIdx).toArray,
                rRow.getAs[Vector](fieldIdx).toArray,
                1e-6
              )
            case _ =>
              Assert.assertEquals(lRow(fieldIdx), rRow(fieldIdx))
          }
        }
      }
    }


    @throws(classOf[java.lang.AssertionError])
    def comparePoolWithExpectedData(
        pool: Pool,
        expectedDataSchema: Seq[StructField],
        expectedData: Seq[Row],
        expectedFeatureNames: Array[String]
    ) = {
      val spark = pool.data.sparkSession

      assertEqualsWithPrecision(
        pool.data,
        spark.createDataFrame(
          spark.sparkContext.parallelize(expectedData),
          StructType(expectedDataSchema)
        )
      )

      Assert.assertTrue(pool.getFeatureNames.sameElements(expectedFeatureNames))
    }
}
