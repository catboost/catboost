package ai.catboost.spark;

import org.junit.{Assert,Test,Ignore};

class SparkHelpersTest {

  @Test @Ignore
  @throws(classOf[Exception])
  def test_parseMemoryOverHeadOption() {
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("1"), 1L << 20L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("10"), 10L << 20L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("5k"), 5L << 10L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("5K"), 5L << 10L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("10m"), 10L << 20L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("16M"), 16L << 20L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("102g"), 102L << 30L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("11G"), 11L << 30L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("4t"), 4L << 40L)
    Assert.assertEquals(SparkHelpers.parseMemoryOverHeadOption("88T"), 88L << 40L)
  }
}
