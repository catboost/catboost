package ai.catboost.spark.params;

import scala.reflect._

import collection.mutable

import com.google.common.base.CaseFormat
import com.google.common.base.Predicates.alwaysTrue

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._
import org.apache.spark.ml.param._;
import org.apache.spark.ml.util.Identifiable


// copied from org.apache.spark.ml.param because it's private there
object ParamValidators {
  def alwaysTrue[T]: T => Boolean = (_: T) => true
}

// A separate class is needed because default Param[E] won't jsonEncode/Decode properly
class EnumParam[E <: java.lang.Enum[E] : ClassTag](
  parent: String,
  name: String,
  doc: String,
  isValid: E => Boolean
) extends Param[E](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) = {
    this(parent, name, doc, ParamValidators.alwaysTrue[E])
  }

  def this(parent: Identifiable, name: String, doc: String, isValid: E => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: E): ParamPair[E] = super.w(value)

  override def jsonEncode(value: E): String = {
    compact(render(JString(value.toString)))
  }

  override def jsonDecode(json: String): E = {
    implicit val formats = DefaultFormats
    java.lang.Enum.valueOf(
        implicitly[ClassTag[E]].runtimeClass.asInstanceOf[Class[E]], parse(json).extract[String]
    )
  }
}

// A separate class is needed because default Param[E] won't jsonEncode/Decode properly
class DurationParam(
  parent: String,
  name: String,
  doc: String,
  isValid: java.time.Duration => Boolean
) extends Param[java.time.Duration](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) = {
    this(parent, name, doc, ParamValidators.alwaysTrue[java.time.Duration])
  }

  def this(parent: Identifiable, name: String, doc: String, isValid: java.time.Duration => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: java.time.Duration): ParamPair[java.time.Duration] = super.w(value)

  override def jsonEncode(value: java.time.Duration): String = {
    compact(render(JString(value.toString)))
  }

  override def jsonDecode(json: String): java.time.Duration = {
    implicit val formats = DefaultFormats
    java.time.Duration.parse(parse(json).extract[String])
  }
}


object Helpers {
  val namesMap = Map(
      "ignored_features_indices" -> "ignored_features",
      "ignored_features_names" -> "ignored_features",

      "worker_initialization_timeout" -> null,

      // defined in Predictor
      "features_col" -> null,
      "prediction_col" -> null,
      "label_col" -> null,

      // defined in ProbabilisticClassifier
      "raw_prediction_col" -> null,
      "probability_col" -> null,
      "thresholds" -> null
  )

  // result can be null, ignore such params
  def getMappedParamName(mllibParamName: String) : String = {
    val name = CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, mllibParamName)
    namesMap.getOrElse(name, name)
  }

  def sparkMlParamsToCatBoostJsonParams(params: Params) : JObject = {
    JObject(
      params.extractParamMap.toSeq.collect {
        case ParamPair(param, value) if getMappedParamName(param.name) != null
          => (getMappedParamName(param.name) -> parse(param.jsonEncode(value)))
      }.toList
    )
  }

  def sparkMlParamsToCatBoostJsonParamsString(params: Params) : String = {
    compact(sparkMlParamsToCatBoostJsonParams(params))
  }
}
