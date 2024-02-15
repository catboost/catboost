package ai.catboost.spark

import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.Row

import org.junit.Assert


object TestMetrics {
  def assertMeanAveragePrecisionIsEqual(
    expectedMAPtopK : Double,
    dataset: Pool,
    ranker: CatBoostRegressionModel,
    topKinMAP: Int = 3
  ) = {
    val predictions = ranker.transform(dataset.data)

    val forRankingMetrics = predictions.select("groupId", "prediction", "label").rdd
      .groupBy(row => row.getLong(0)).flatMap{
        case (groupId, groupData : Iterable[Row]) => {
          val groupDataSeq : Seq[Row]= groupData.toSeq
          val positiveObjectIds
            = (0 until groupDataSeq.length).filter(groupDataSeq(_).getString(2).toDouble >= 0.5)
          val predictedObjectIds
            = (0 until groupDataSeq.length).sortBy(-groupDataSeq(_).getDouble(1)).take(topKinMAP)
          Some((predictedObjectIds.toArray, positiveObjectIds.toArray))
        }
      }

      val actualMAPtopK = new RankingMetrics[Int](forRankingMetrics).meanAveragePrecision

      Assert.assertEquals(expectedMAPtopK, actualMAPtopK, 1e-5)
  }
}
