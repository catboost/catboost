package ai.catboost.spark;

import java.io.{File,PrintWriter}

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

    def assertEqualsWithPrecision(
      expected: Pool,
      actual: Pool,

      // set to true if order of rows might change. Requires sampleId or (groupId, sampleId) in data
      compareByIds: Boolean = false,
      ignoreNullableInSchema: Boolean = false
    ) = {
      val dataSortByFields = if (compareByIds) {
        if (expected.isDefined(expected.groupIdCol)) {
          Seq(expected.getOrDefault(expected.groupIdCol), actual.getOrDefault(actual.sampleIdCol))
        } else {
          Seq(expected.getOrDefault(expected.sampleIdCol))
        }
      } else {
        Seq()
      }

      TestHelpers.assertEqualsWithPrecision(
        expected.data,
        actual.data,
        dataSortByFields,
        ignoreNullableInSchema
      )

      if (expected.quantizedFeaturesInfo == null) {
        Assert.assertNull(actual.quantizedFeaturesInfo)
      } else {
        Assert.assertNotNull(actual.quantizedFeaturesInfo)
        Assert.assertTrue(actual.quantizedFeaturesInfo.EqualWithoutOptionsTo(expected.quantizedFeaturesInfo.__deref__))
      }

      TestHelpers.assertEqualsWithPrecision(
        expected.pairsData,
        actual.pairsData,
        Seq("groupId", "winnerId", "loserId"),
        ignoreNullableInSchema
      )

      Assert.assertEquals(expected.partitionedByGroups, actual.partitionedByGroups)
    }

    def createRawPool(
      appName: String,
      srcDataSchema : Seq[StructField],
      srcData: Seq[Row],
      columnNames: Map[String, String], // standard column name to name of column in the dataset
      srcPairsData: Option[Seq[Row]] = None,
      pairsHaveWeight: Boolean=false
    ) : Pool = {
      val spark = TestHelpers.getOrCreateSparkSession(appName)

      val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema));
      var pool = srcPairsData match {
        case Some(srcPairsData) => {
          var pairsSchema = Seq(
            StructField("groupId", LongType, false),
            StructField("winnerId", LongType, false),
            StructField("loserId", LongType, false)
          )
          if (pairsHaveWeight) {
            pairsSchema = pairsSchema :+ StructField("weight", FloatType, false)
          }
          val pairsDf = spark.createDataFrame(
            spark.sparkContext.parallelize(srcPairsData),
            StructType(pairsSchema)
          )
          new Pool(df, pairsDf)
        }
        case None => new Pool(df)
      }

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

    /**
     * specify either objectCount or groupCount
     */
    def createRandomPool(
      spark: SparkSession,
      floatFeatureCount: Int,
      catFeatureCount: Int = 0,
      objectCount: Long = 0,
      groupCount: Long = 0,
      expectedGroupSize: Int = 20,
      hasWeight: Boolean = false,
      pairsFraction: Double = 0.0, // fraction from all n^2/2 possible pairs for each group
      pairsHaveWeight: Boolean = false,
      hasSampleId: Boolean = false
    ) : Pool = {
      if ((groupCount != 0) && (objectCount != 0)) {
        throw new RuntimeException("both objectCount and groupCount specified")
      }

      var schemaFields = Seq[StructField]()
      if (groupCount != 0) {
        schemaFields = schemaFields :+ StructField("groupId", LongType, false)
      }
      schemaFields = schemaFields :+ StructField("features", SQLDataTypes.VectorType, false)
      schemaFields = schemaFields :+ StructField("label", DoubleType, false)
      if (hasWeight) {
        schemaFields = schemaFields :+ StructField("weight", DoubleType, false)
      }

      val hasSampleIdUpdated = hasSampleId || (pairsFraction != 0)
      if (hasSampleIdUpdated) {
        schemaFields = schemaFields :+ StructField("sampleId", LongType, false)
      }

      val random = new scala.util.Random(0)
      var sampleId : Long = 0
      var groupId : Long = 0

      val createRow = () => Row {
        var fields = Seq[Any]()

        if (groupCount != 0) {
          fields = fields :+ groupId
        }

        val features = new Array[Double](floatFeatureCount + catFeatureCount)
        for (floatFeatureIdx <- 0 until floatFeatureCount) {
          features(floatFeatureIdx) = random.nextDouble()
        }
        for (catFeatureIdx <- 0 until catFeatureCount) {
          features(floatFeatureCount + catFeatureIdx) = random.nextInt(30)
        }
        fields = fields :+ Vectors.dense(features)

        // label
        fields = fields :+ random.nextDouble()

        if (hasWeight) {
          fields = fields :+ random.nextDouble()
        }

        if (hasSampleIdUpdated) {
          // sampleId
          fields = fields :+ sampleId
        }
        Row.fromSeq(fields)
      }

      var rows = Seq[Row]()
      var pairsRows = Seq[Row]()
      if (groupCount != 0) {
        for (groupIdx <- 0L until groupCount) {
          groupId = groupIdx
          val groupSize = random.nextInt(2 * expectedGroupSize) + 1
          var sampleIds = Seq[Long]()
          for (idxInGroup <- 0 until groupSize) {
            rows = rows :+ createRow()(0).asInstanceOf[Row]

            sampleIds = sampleIds :+ sampleId
            sampleId = sampleId + 1
          }
          if (pairsFraction != 0.0) {
            val shuffledSampleIds = random.shuffle(sampleIds)
            for (firstIdx <- 0 until (groupSize - 1)) {
              for (secondIdx <- (firstIdx + 1) until groupSize) {
                if (random.nextDouble() <= pairsFraction) {
                  var pairFields = Seq[Any](groupId, shuffledSampleIds(firstIdx), shuffledSampleIds(secondIdx))
                  if (pairsHaveWeight) {
                    pairFields = pairFields :+ random.nextFloat()
                  }
                  pairsRows = pairsRows :+ Row.fromSeq(pairFields)
                }
              }
            }
          }
        }
      } else {
        for (i <- 0L until objectCount) {
          val row = createRow()(0).asInstanceOf[Row]
          rows = rows :+ row
          sampleId = sampleId + 1
        }
      }

      val mainDf = spark.createDataFrame(spark.sparkContext.parallelize(rows), StructType(schemaFields))

      var pool: Pool = null
      if (pairsFraction != 0.0) {
        var pairsSchemaFields = Seq(
          StructField("groupId", LongType, false),
          StructField("winnerId", LongType, false),
          StructField("loserId", LongType, false)
        )
        if (pairsHaveWeight) {
          pairsSchemaFields = pairsSchemaFields :+ StructField("weight", FloatType, false)
        }
        val pairsDf = spark.createDataFrame(
          spark.sparkContext.parallelize(pairsRows),
          StructType(pairsSchemaFields)
        )
        pool = (new Pool(mainDf, pairsDf)).setGroupIdCol("groupId")
      } else {
        pool = new Pool(mainDf)
        if (groupCount != 0) {
          pool = pool.setGroupIdCol("groupId")
        }
      }
      if (hasWeight) {
        pool = pool.setWeightCol("weight")
      }
      if (hasSampleIdUpdated) {
        pool = pool.setSampleIdCol("sampleId")
      }
      pool
    }

    def printDataFrame(df: DataFrame, file: File) = {
      val dfWriter = new PrintWriter(file)
      for (row <- df.collect()) {
        val rowFields = row.toSeq.map { cell =>
          cell match {
            case null => "null"
            case binary: Array[Byte] => binary.map("%02X".format(_)).mkString("[", " ", "]")
            case _ => cell.toString
          }
        }
        dfWriter.println(rowFields.mkString(" , "))
      }
      dfWriter.close()
    }
}
