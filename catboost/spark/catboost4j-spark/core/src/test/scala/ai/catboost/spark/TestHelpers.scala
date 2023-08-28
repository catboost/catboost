package ai.catboost.spark

import scala.collection.JavaConverters._

import java.io.{File,PrintWriter}
import java.nio.file.{Files, Path}
import java.util.zip.ZipFile

import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import org.junit.Assert


object TestHelpers {
  def getCurrentMethodName:String = Thread.currentThread.getStackTrace()(2).getMethodName

  def appendToRow(src: Row, value: Any) : Row = {
    Row.fromSeq(src.toSeq :+ value)
  }

  def getOrCreateSparkSession(appName: String) : SparkSession = {
    SparkSession.builder()
        .master("local[4]")
        .appName(appName)
        .getOrCreate()
  }

  def getDataForComparison(data: DataFrame, sortByFields: Seq[String]) : Array[Row] = {
    (if (sortByFields.isEmpty) {
      data
    } else {
      data.sort(sortByFields.head, sortByFields.tail:_*)
    }).collect()
  }

  def assertEqualsWithPrecision(
    expected: DataFrame,
    actual: DataFrame,
    sortByFields: Seq[String] = Seq(),
    ignoreNullableInSchema: Boolean = false
  ): Unit = {
    if (expected == null) {
      Assert.assertTrue(actual == null)
      return ()
    }
    Assert.assertTrue(actual != null)

    Assert.assertEquals(expected.count, actual.count)

    Assert.assertEquals(expected.schema.size, actual.schema.size)
    for (i <- 0 until expected.schema.length) {
      val expectedField = expected.schema(i)
      val actualField = actual.schema(i)
      Assert.assertEquals(expectedField.name, actualField.name)
      Assert.assertEquals(expectedField.dataType, actualField.dataType)
      if (!ignoreNullableInSchema) {
        Assert.assertEquals(expectedField.nullable, actualField.nullable)
      }
      Assert.assertEquals(expectedField.metadata, actualField.metadata)
    }

    val schema = expected.schema
    val rowSize = schema.size

    val expectedRows = getDataForComparison(expected, sortByFields)
    val actualRows = getDataForComparison(actual, sortByFields)

    for (rowIdx <- 0 until expectedRows.size) {
      val expectedRow = expectedRows(rowIdx)
      val actualRow = actualRows(rowIdx)

      Assert.assertEquals(rowSize, expectedRow.size)
      Assert.assertEquals(rowSize, actualRow.size)

      for (fieldIdx <- 0 until rowSize) {
        schema(fieldIdx).dataType match {
          case FloatType =>
            Assert.assertEquals(expectedRow.getAs[Float](fieldIdx), actualRow.getAs[Float](fieldIdx), 1e-5f)
          case DoubleType =>
            Assert.assertEquals(expectedRow.getAs[Double](fieldIdx), actualRow.getAs[Double](fieldIdx), 1e-6)
          case SQLDataTypes.VectorType =>
            Assert.assertArrayEquals(
              expectedRow.getAs[Vector](fieldIdx).toArray,
              actualRow.getAs[Vector](fieldIdx).toArray,
              1e-6
            )
          case BinaryType =>
            java.util.Arrays.equals(
              expectedRow.getAs[Array[Byte]](fieldIdx),
              actualRow.getAs[Array[Byte]](fieldIdx)
            )
          case _ =>
            Assert.assertEquals(expectedRow(fieldIdx), actualRow(fieldIdx))
        }
      }
    }
    ()
  }

  def addIndexColumn(df: DataFrame) : DataFrame = {
    df.sparkSession.createDataFrame(
      df.rdd.zipWithIndex.map {
        case (row, index) => Row.fromSeq(row.toSeq :+ index)
      },
      StructType(df.schema.fields :+ StructField("index", LongType, false))
    )
  }

  def unzip(zipPath: Path, outputPath: Path): Unit = {
    val zipFile = new ZipFile(zipPath.toFile)
    for (entry <- zipFile.entries.asScala) {
      val path = outputPath.resolve(entry.getName)
      if (entry.isDirectory) {
        Files.createDirectories(path)
      } else {
        Files.createDirectories(path.getParent)
        Files.copy(zipFile.getInputStream(entry), path)
      }
    }
  }

}
