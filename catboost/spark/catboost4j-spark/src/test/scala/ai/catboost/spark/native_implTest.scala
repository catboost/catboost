package ai.catboost.spark;

import collection.JavaConverters._
import scala.reflect.ClassTag

import org.junit.{Assert,Test};

import ru.yandex.catboost.spark.catboost4j_spark.src.native_impl._;

class native_implTest {
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
      }
    }
  }
}
