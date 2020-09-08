package ai.catboost.spark.params;

import scala.reflect._

import com.google.common.base.CaseFormat
import com.google.common.base.Predicates.alwaysTrue

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._
import org.apache.spark.ml.param._;
import org.apache.spark.ml.util.Identifiable


// copied from org.apache.spark.ml.param because it's private there
object EnumParamValidators {
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
    this(parent, name, doc, EnumParamValidators.alwaysTrue[E])
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


object Helpers {
  val namesMap = Map(
      "ignored_features_indices" -> "ignored_features",
      "ignored_features_names" -> "ignored_features"
  )
  
  def sparkMlParamsToCatBoostJsonParams(params: Params) : JValue = {
    JObject(
      params.extractParamMap.toSeq.map {
        case ParamPair(param, value) => {
          val name = CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, param.name)
          namesMap.getOrElse(name, name) -> parse(param.jsonEncode(value))
        }
      }.toList
    )
  }
  
  def sparkMlParamsToCatBoostJsonParamsString(params: Params) : String = {
    compact(sparkMlParamsToCatBoostJsonParams(params))
  }
}
