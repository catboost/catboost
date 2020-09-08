package ai.catboost.spark;

import collection.JavaConverters._
import scala.reflect.ClassTag

import org.junit.{Assert,Test};

import org.apache.commons.lang3.SerializationUtils

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl._;

class native_implTest {
  ensureNativeLibLoaded
  
  @Test
  @throws(classOf[Exception])
  def testTFeatureMetaInfoSerialization() {
    {
      val v = new TFeatureMetaInfo()
      v.setType(EFeatureType.Float)
      Assert.assertEquals(v, SerializationUtils.roundtrip(v))
    }
    {
      val v = new TFeatureMetaInfo()
      v.setType(EFeatureType.Categorical)
      v.setName("c1")
      v.setIsSparse(true)
      Assert.assertEquals(v, SerializationUtils.roundtrip(v))
    }
    {
      val v = new TFeatureMetaInfo()
      v.setType(EFeatureType.Text)
      v.setName("t1")
      v.setIsIgnored(true)
      v.setIsAvailable(false)
      Assert.assertEquals(v, SerializationUtils.roundtrip(v))
    }
  }
  
  @Test
  @throws(classOf[Exception])
  def testTFeaturesLayoutSerialization() {
    val data = new TVector_TFeatureMetaInfo();
    {
      val v = new TFeatureMetaInfo();
      v.setType(EFeatureType.Float);
      data.add(v);
    }
    {
      val v = new TFeatureMetaInfo();
      v.setType(EFeatureType.Categorical);
      v.setName("c1");
      v.setIsSparse(true);
      data.add(v);
    }
    {
      val v = new TFeatureMetaInfo();
      v.setType(EFeatureType.Text);
      v.setName("t1");
      v.setIsIgnored(true);
      v.setIsAvailable(false);
      data.add(v);
    }
    
    val featuresLayout = new TFeaturesLayout();
    featuresLayout.Init(data);
    Assert.assertEquals(featuresLayout, SerializationUtils.roundtrip(featuresLayout));
  }
  
  @Test
  @throws(classOf[Exception])
  def testQuantizedFeaturesInfo() {
    def generateFeaturesLayout : TFeaturesLayout = {
      val data = new TVector_TFeatureMetaInfo();
      {
        val v = new TFeatureMetaInfo();
        v.setType(EFeatureType.Float);
        v.setName("f1");
        data.add(v);
      }
      {
        val v = new TFeatureMetaInfo();
        v.setType(EFeatureType.Float);
        v.setName("f2");
        v.setIsSparse(true);
        data.add(v);
      }
      {
        val v = new TFeatureMetaInfo();
        v.setType(EFeatureType.Text);
        v.setName("f3");
        v.setIsIgnored(true);
        v.setIsAvailable(false);
        data.add(v);
      }
      val featuresLayout = new TFeaturesLayout();
      featuresLayout.Init(data);
      featuresLayout
    }
    
    val quantizedFeaturesInfo = new TQuantizedFeaturesInfo();
    {
      quantizedFeaturesInfo.Init(generateFeaturesLayout);
      quantizedFeaturesInfo.SetNanMode(0, ENanMode.Max);
      quantizedFeaturesInfo.SetQuantization(
        0,
        new TVector_float(Array[Float](0.1f, 0.3f, 0.9f)),
        /*defaultQuantizedBin*/ null
      );
      
      quantizedFeaturesInfo.SetNanMode(1, ENanMode.Min);
      
      val defaultQuantizedBin = new TDefaultQuantizedBin();
      defaultQuantizedBin.setIdx(1.toLong);
      defaultQuantizedBin.setFraction(0.7f);
      
      quantizedFeaturesInfo.SetQuantization(
        1,
        new TVector_float(Array[Float](0.0f, 0.5f, 1.0f, 2.3f)),
        defaultQuantizedBin
      );
    }
    
    Assert.assertEquals(quantizedFeaturesInfo.GetFeaturesLayout.__deref__, generateFeaturesLayout);
    {
      val borders = new TVector_float;
      val hasDefaultQuantizedBin = new boolp();
      val defaultQuantizedBin = new TDefaultQuantizedBin();

      Assert.assertEquals(quantizedFeaturesInfo.GetNanMode(0), ENanMode.Max);

      quantizedFeaturesInfo.GetQuantization(0, borders, hasDefaultQuantizedBin.cast, defaultQuantizedBin);
      Assert.assertEquals(borders, new TVector_float(Array[Float](0.1f, 0.3f, 0.9f)));
      Assert.assertEquals(hasDefaultQuantizedBin.value, false);

      Assert.assertEquals(quantizedFeaturesInfo.GetNanMode(1), ENanMode.Min);

      quantizedFeaturesInfo.GetQuantization(1, borders, hasDefaultQuantizedBin.cast, defaultQuantizedBin);
      Assert.assertEquals(borders, new TVector_float(Array[Float](0.0f, 0.5f, 1.0f, 2.3f)));
      Assert.assertEquals(hasDefaultQuantizedBin.value, true);
      Assert.assertEquals(defaultQuantizedBin.getIdx, 1.toLong);
      Assert.assertEquals(defaultQuantizedBin.getFraction, 0.7f, 1e-13f);
    }
    
    Assert.assertEquals(quantizedFeaturesInfo, SerializationUtils.roundtrip(quantizedFeaturesInfo));
  }
    
  
  @Test
  @throws(classOf[Exception])
  def testTVector_float() {
    val testSequences = Seq(
      Seq[Float](),    
      Seq(0.1f),
      Seq(0.2f, 0.4f, 1.2f)
    )
    
    testSequences.map {
      testSequence => {
        val srcArray = testSequence.toArray
        val v = new TVector_float(srcArray)
        val fromVector = v.toPrimitiveArray
        Assert.assertArrayEquals(srcArray, fromVector, 1e-13f)
        
        val v2 = new TVector_float(srcArray)
        Assert.assertEquals(v, v2)
        
        v2.add(0.88f)
        Assert.assertNotEquals(v, v2)
        
        Assert.assertEquals(v, SerializationUtils.roundtrip(v))
      }
    }
  }

  @Test
  @throws(classOf[Exception])
  def testTVector_i32() {
    val testSequences = Seq(
      Seq[Int](),    
      Seq(1),
      Seq(2, 4, 6)
    )
    
    testSequences.map {
      testSequence => {
        val srcArray = testSequence.toArray
        val v = new TVector_i32(srcArray)
        val fromVector = v.toPrimitiveArray
        Assert.assertArrayEquals(srcArray, fromVector)
        
        val v2 = new TVector_i32(srcArray)
        Assert.assertEquals(v, v2)
        
        v2.add(4)
        Assert.assertNotEquals(v, v2)
        
        Assert.assertEquals(v, SerializationUtils.roundtrip(v))
      }
    }
  }
}
