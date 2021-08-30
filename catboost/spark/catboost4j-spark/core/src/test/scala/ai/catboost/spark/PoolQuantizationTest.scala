package ai.catboost.spark;

import org.apache.spark.ml.linalg._;
import org.apache.spark.sql._;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types._;

import org.junit.{Assert,Test};

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._;
import ai.catboost.spark.params._

class PoolQuantizationTest {
    def testQuantizeCase(
        srcDataSchema : Seq[StructField],
        srcData: Seq[Row],
        quantizationParams : QuantizationParams,
        expectedQuantizedDataSchema: Seq[StructField],
        expectedQuantizedData: Seq[Row],
        expectedQuantizedFeaturesInfo: QuantizedFeaturesInfoPtr,
        expectedFeatureNames: Array[String]
    ) {
        val spark = SparkSession.builder()
            .master("local[4]")
            .appName("TestQuantizeCase")
            .getOrCreate();

        val df = spark.createDataFrame(spark.sparkContext.parallelize(srcData), StructType(srcDataSchema));

        val pool = new Pool(df)
        val quantizedPool = pool.quantize(quantizationParams)

        Assert.assertEquals(quantizedPool.data.schema, StructType(expectedQuantizedDataSchema))

        TestHelpers.assertEqualsWithPrecision(
          quantizedPool.data,
          spark.createDataFrame(
            spark.sparkContext.parallelize(expectedQuantizedData),
            StructType(expectedQuantizedDataSchema)
          )
        )

        Assert.assertTrue(quantizedPool.quantizedFeaturesInfo.EqualWithoutOptionsTo(expectedQuantizedFeaturesInfo.__deref__))

        Assert.assertTrue(quantizedPool.getFeatureNames.sameElements(expectedFeatureNames))

        spark.stop();
    }

    @Test
    @throws(classOf[Exception])
    def testQuantize() {
      val featureNames = Array[String]("f1", "f2", "f3")

      testQuantizeCase(
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.0, 1.0, 0.2), 0.0),
          Row(Vectors.dense(0.1, 1.1, 2.1), 1.0),
          Row(Vectors.dense(0.2, 1.2, 2.2), 1.0),
          Row(Vectors.dense(0.0, 1.1, 3.2), 0.0)
        ),
        new QuantizationParams(),

        // expected
        PoolTestHelpers.createSchema(
          Seq(
            ("features", BinaryType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ false
        ),
        Seq(
          Row(Array[Byte](0x00, 0x00, 0x00), 0.0),
          Row(Array[Byte](0x01, 0x01, 0x01), 1.0),
          Row(Array[Byte](0x02, 0x02, 0x02), 1.0),
          Row(Array[Byte](0x00, 0x01, 0x03), 0.0)
        ),
        {
          val metadata = new TVector_TFeatureMetaInfo()
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "f1"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "f2"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "f3"))

          val featuresLayout = native_impl.MakeFeaturesLayout(metadata)
          val quantizedFeaturesInfoPtr = native_impl.MakeQuantizedFeaturesInfo(featuresLayout)
          quantizedFeaturesInfoPtr.SetQuantization(0, new TVector_float(Array[Float](0.05f, 0.15f)))
          quantizedFeaturesInfoPtr.SetQuantization(1, new TVector_float(Array[Float](1.05f, 1.15f)))
          quantizedFeaturesInfoPtr.SetQuantization(2, new TVector_float(Array[Float](1.15f, 2.15f, 2.7f)))

          quantizedFeaturesInfoPtr
        },
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testQuantizeWithNaNs() {
      val featureNames = Array[String]("F1", "F2", "F3")

      testQuantizeCase(
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.0,        1.0, 0.2       ), 3.0),
          Row(Vectors.dense(Double.NaN, 1.1, Double.NaN), 1.0),
          Row(Vectors.dense(0.2,        1.2, 2.2       ), 11.0),
          Row(Vectors.dense(Double.NaN, 0.0, 2.2       ), 0.2),
          Row(Vectors.dense(Double.NaN, 1.1, 0.4       ), 6.1)
        ),
        new QuantizationParams(),

        // expected
        PoolTestHelpers.createSchema(
          Seq(
            ("features", BinaryType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ false
        ),
        Seq(
          Row(Array[Byte](0x01, 0x01, 0x01), 3.0),
          Row(Array[Byte](0x00, 0x02, 0x00), 1.0),
          Row(Array[Byte](0x02, 0x03, 0x03), 11.0),
          Row(Array[Byte](0x00, 0x00, 0x03), 0.2),
          Row(Array[Byte](0x00, 0x02, 0x02), 6.1)
        ),
        {
          val metadata = new TVector_TFeatureMetaInfo()
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F1"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F2"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F3"))

          val featuresLayout = native_impl.MakeFeaturesLayout(metadata)
          val quantizedFeaturesInfoPtr = native_impl.MakeQuantizedFeaturesInfo(featuresLayout)
          quantizedFeaturesInfoPtr.SetNanMode(0, ENanMode.Min)
          quantizedFeaturesInfoPtr.SetQuantization(0, new TVector_float(Array[Float](Float.MinValue, 0.1f)))
          quantizedFeaturesInfoPtr.SetQuantization(1, new TVector_float(Array[Float](0.5f, 1.05f, 1.15f)))
          quantizedFeaturesInfoPtr.SetNanMode(2, ENanMode.Min)
          quantizedFeaturesInfoPtr.SetQuantization(2, new TVector_float(Array[Float](Float.MinValue, 0.3f, 1.3f)))

          quantizedFeaturesInfoPtr
        },
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testQuantizeWithNaNsAndBorderCount() {
      val featureNames = Array[String]("F1", "F2", "F3", "F4")

      testQuantizeCase(
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.0,        1.0, 0.2       , 100.11), 3.0),
          Row(Vectors.dense(Double.NaN, 1.1, Double.NaN, 20.2), 1.0),
          Row(Vectors.dense(0.2,        1.2, 2.2       , 32.4), 11.0),
          Row(Vectors.dense(Double.NaN, 0.0, 2.2       , 71.1), 0.2),
          Row(Vectors.dense(Double.NaN, 1.1, 0.4       , 92.2), 6.1),
          Row(Vectors.dense(0.1,        0.0, 1.8       , 111.0), 2.0),
          Row(Vectors.dense(0.28,       0.0, 8.3       , 333.2), 0.0)
        ),
        new QuantizationParams().setBorderCount(2).setNanMode(ENanMode.Max),

        // expected
        PoolTestHelpers.createSchema(
          Seq(
            ("features", BinaryType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ false
        ),
        Seq(
          Row(Array[Byte](0x00, 0x01, 0x00, 0x01), 3.0),
          Row(Array[Byte](0x03, 0x02, 0x03, 0x00), 1.0),
          Row(Array[Byte](0x02, 0x02, 0x02, 0x00), 11.0),
          Row(Array[Byte](0x03, 0x00, 0x02, 0x00), 0.2),
          Row(Array[Byte](0x03, 0x02, 0x01, 0x01), 6.1),
          Row(Array[Byte](0x01, 0x00, 0x01, 0x02), 2.0),
          Row(Array[Byte](0x02, 0x00, 0x02, 0x02), 0.0)
        ),
        {
          val metadata = new TVector_TFeatureMetaInfo()
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F1"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F2"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F3"))
          metadata.add(native_impl.MakeFeatureMetaInfo(EFeatureType.Float, "F4"))

          val featuresLayout = native_impl.MakeFeaturesLayout(metadata)
          val quantizedFeaturesInfoPtr = native_impl.MakeQuantizedFeaturesInfo(featuresLayout)
          quantizedFeaturesInfoPtr.SetNanMode(0, ENanMode.Max)
          quantizedFeaturesInfoPtr.SetQuantization(0, new TVector_float(Array[Float](0.05f, 0.15f, Float.MaxValue)))
          quantizedFeaturesInfoPtr.SetQuantization(1, new TVector_float(Array[Float](0.5f, 1.05f)))
          quantizedFeaturesInfoPtr.SetNanMode(2, ENanMode.Max)
          quantizedFeaturesInfoPtr.SetQuantization(2, new TVector_float(Array[Float](0.3f, 2.0f, Float.MaxValue)))
          quantizedFeaturesInfoPtr.SetQuantization(3, new TVector_float(Array[Float](81.6499938f, 105.555f)))

          quantizedFeaturesInfoPtr
        },
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testQuantizeWithNaNsAndConstantFeatures() {
      val featureNames = Array[String]("F1", "F2", "F3", "F4")

      testQuantizeCase(
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.0,        1.0, Double.NaN, 100.11), 3.0),
          Row(Vectors.dense(Double.NaN, 1.0, Double.NaN, 20.2), 1.0),
          Row(Vectors.dense(0.2,        1.0, Double.NaN, 32.4), 11.0),
          Row(Vectors.dense(Double.NaN, 1.0, Double.NaN, 71.1), 0.2),
          Row(Vectors.dense(Double.NaN, 1.0, Double.NaN, 92.2), 6.1),
          Row(Vectors.dense(0.1,        1.0, Double.NaN, 111.0), 2.0),
          Row(Vectors.dense(0.28,       1.0, Double.NaN, 333.2), 0.0)
        ),
        new QuantizationParams().setBorderCount(2).setNanMode(ENanMode.Max),

        // expected
        PoolTestHelpers.createSchema(
          Seq(
            ("features", BinaryType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ false
        ),
        Seq(
          Row(Array[Byte](0x00, 0x01), 3.0),
          Row(Array[Byte](0x03, 0x00), 1.0),
          Row(Array[Byte](0x02, 0x00), 11.0),
          Row(Array[Byte](0x03, 0x00), 0.2),
          Row(Array[Byte](0x03, 0x01), 6.1),
          Row(Array[Byte](0x01, 0x02), 2.0),
          Row(Array[Byte](0x02, 0x02), 0.0)
        ),
        {
          val featuresLayout = native_impl.MakeFeaturesLayout(
              4,
              new TVector_TString(featureNames),
              /*ignoredFeatures*/ new TVector_i32(Array[Int](1, 2))
          )
          val quantizedFeaturesInfoPtr = native_impl.MakeQuantizedFeaturesInfo(featuresLayout)
          quantizedFeaturesInfoPtr.SetNanMode(0, ENanMode.Max)
          quantizedFeaturesInfoPtr.SetQuantization(0, new TVector_float(Array[Float](0.05f, 0.15f, Float.MaxValue)))
          quantizedFeaturesInfoPtr.SetQuantization(3, new TVector_float(Array[Float](81.6499938f, 105.555f)))

          quantizedFeaturesInfoPtr
        },
        featureNames
      )
    }

    @Test
    @throws(classOf[Exception])
    def testQuantizeWithNaNsAndIgnoredFeatures() {
      val featureNames = Array[String]("F1", "F2", "F3", "F4")

      testQuantizeCase(
        PoolTestHelpers.createSchema(
          Seq(
            ("features", SQLDataTypes.VectorType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ true
        ),
        Seq(
          Row(Vectors.dense(0.0,        1.0, 0.2       , 100.11), 3.0),
          Row(Vectors.dense(Double.NaN, 1.1, Double.NaN, 20.2), 1.0),
          Row(Vectors.dense(0.2,        1.2, 2.2       , 32.4), 11.0),
          Row(Vectors.dense(Double.NaN, 0.0, 2.2       , 71.1), 0.2),
          Row(Vectors.dense(Double.NaN, 1.1, 0.4       , 92.2), 6.1),
          Row(Vectors.dense(0.1,        0.0, 1.8       , 111.0), 2.0),
          Row(Vectors.dense(0.28,       0.0, 8.3       , 333.2), 0.0)
        ),
        new QuantizationParams().setBorderCount(2).setIgnoredFeaturesIndices(Array[Int](0, 2)),

        // expected
        PoolTestHelpers.createSchema(
          Seq(
            ("features", BinaryType),
            ("label", DoubleType)
          ),
          featureNames,
          /*addFeatureNamesMetadata*/ false
        ),
        Seq(
          Row(Array[Byte](0x01, 0x01), 3.0),
          Row(Array[Byte](0x02, 0x00), 1.0),
          Row(Array[Byte](0x02, 0x00), 11.0),
          Row(Array[Byte](0x00, 0x00), 0.2),
          Row(Array[Byte](0x02, 0x01), 6.1),
          Row(Array[Byte](0x00, 0x02), 2.0),
          Row(Array[Byte](0x00, 0x02), 0.0)
        ),
        {
          val featuresLayout = native_impl.MakeFeaturesLayout(
              4,
              new TVector_TString(featureNames),
              /*ignoredFeatures*/ new TVector_i32(Array[Int](0, 2))
          )
          val quantizedFeaturesInfoPtr = native_impl.MakeQuantizedFeaturesInfo(featuresLayout)
          //quantizedFeaturesInfoPtr.SetQuantization(0, new TVector_float())
          quantizedFeaturesInfoPtr.SetQuantization(1, new TVector_float(Array[Float](0.5f, 1.05f)))
          //quantizedFeaturesInfoPtr.SetQuantization(2, new TVector_float())
          quantizedFeaturesInfoPtr.SetQuantization(3, new TVector_float(Array[Float](81.6499938f, 105.555f)))

          quantizedFeaturesInfoPtr
        },
        featureNames
      )
    }
}
