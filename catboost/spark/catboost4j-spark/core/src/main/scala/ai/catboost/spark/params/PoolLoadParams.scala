package ai.catboost.spark.params;

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable

import ai.catboost.spark.params.macros.ParamGetterSetter


class PoolLoadParams (override val uid: String) extends Params
{
  def this() = this(
    Identifiable.randomUID("catboostPoolLoadParams")
  )

  override def copy(extra: ParamMap): PoolLoadParams = defaultCopy(extra)

  @ParamGetterSetter
  final val delimiter: Param[String] = new Param[String](
    this,
    "delimiter",
    "The delimiter character used to separate the data in the dataset description input file."
  )

  setDefault(delimiter, "\t")

  @ParamGetterSetter
  final val hasHeader: BooleanParam = new BooleanParam(
    this,
    "hasHeader",
    "Read the column names from the first line of the dataset description file if this parameter is set."
  )
}
