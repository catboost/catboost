package ai.catboost.spark.params;

import scala.reflect._

import collection.mutable
import scala.jdk.CollectionConverters._

import com.google.common.base.CaseFormat
import com.google.common.base.Predicates.alwaysTrue

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._

import org.apache.spark.ml.param._;
import org.apache.spark.ml.util.Identifiable
import ai.catboost.CatBoostError

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl.EOverfittingDetectorType


// copied from org.apache.spark.ml.param because it's private there
private[spark] object ParamValidators {
  def alwaysTrue[T]: T => Boolean = (_: T) => true
}


/** A separate class is needed because default Param[E] won't jsonEncode/Decode properly */
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

/** supported V types are String, Long, Double or Boolean */
class OrderedStringMapParam[V](
  parent: String,
  name: String,
  doc: String,
  isValid: java.util.LinkedHashMap[String, V] => Boolean
) extends Param[java.util.LinkedHashMap[String, V]](parent, name, doc, isValid) {

  def this(parent: String, name: String, doc: String) = {
    this(parent, name, doc, ParamValidators.alwaysTrue[java.util.LinkedHashMap[String, V]])
  }

  def this(parent: Identifiable, name: String, doc: String, isValid: java.util.LinkedHashMap[String, V] => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  /** Creates a param pair with the given value (for Java). */
  override def w(value: java.util.LinkedHashMap[String, V]): ParamPair[java.util.LinkedHashMap[String, V]] = super.w(value)

  override def jsonEncode(value: java.util.LinkedHashMap[String, V]): String = {
    implicit val formats = DefaultFormats
    compact(
      render(
        value.asScala.foldLeft(JObject())(
          (acc, kv) => {
            val jValue = kv._2 match {
              case s : String => JString(s)
              case num: Double => JDouble(num)
              case num: Long => JInt(BigInt(num))
              case value: Boolean => JBool(value)
              case _ => throw new RuntimeException("Unsupported map value type")
            }
            acc ~ (kv._1 -> jValue)
          }
        )
      )
    )
  }

  override def jsonDecode(json: String):  java.util.LinkedHashMap[String, V] = {
    val jObject = parse(json).asInstanceOf[JObject]
    val result = new java.util.LinkedHashMap[String, V]
    for ((key, jValue) <- jObject.obj) {
      jValue match {
        case JString(s) =>  result.put(key, s.asInstanceOf[V])
        case JDouble(num) =>  result.put(key, num.asInstanceOf[V])
        case JInt(num) =>  result.put(key, num.longValue.asInstanceOf[V])
        case JBool(value) =>  result.put(key, value.asInstanceOf[V])
        case _ => throw new RuntimeException("Unexpected JSON object value type for map")
      }
    }
    result
  }
}


private[spark] object Helpers {
  // difference with Params.getOrDefault is that if default value is undefined this function returns None
  def getOrDefault(params: Params, param: Param[_]) : Option[_] = {
    if (params.isSet(param)) {
      params.get(param)
    } else {
      params.getDefault(param)
    }
  }

  /** If param is set in both rParams and lParams check that values are equal */
  def checkParamsCompatibility(lParamsName: String, lParams: Params, rParamsName: String, rParams: Params) = {
    for (lParam <- lParams.params) {
      if (rParams.hasParam(lParam.name)) {
        (getOrDefault(lParams, lParam), getOrDefault(rParams, rParams.getParam(lParam.name))) match {
          case (Some(lValue), Some(rValue)) if (!lValue.equals(rValue)) =>
            throw new CatBoostError(
              s"Both $lParamsName and $rParamsName have parameter ${lParam.name} specified "
              + s"but with different values: $lValue and $rValue respectively."
            )
          case _ => ()
        }
      }
    }
  }

  def checkIncompatibleParams(params: mutable.HashMap[String, Any]) = {
    if (params.contains("ignoredFeaturesIndices") && params.contains("ignoredFeaturesNames")) {
      throw new CatBoostError("params cannot contain both ignoredFeaturesIndices and ignoredFeaturesNames")
    }
    if (params.contains("monotoneConstraintsMap") && params.contains("monotoneConstraintList")) {
      throw new CatBoostError("params cannot contain both monotoneConstraintMap and monotoneConstraintList")
    }
    if (params.contains("featureWeightsMap") && params.contains("featureWeightsList")) {
      throw new CatBoostError("params cannot contain both featureWeightsMap and featureWeightsList")
    }
    if (params.contains("firstFeatureUsePenaltiesMap") && params.contains("firstFeatureUsePenaltiesList")) {
      throw new CatBoostError(
        "params cannot contain both firstFeatureUsePenaltiesMap and firstFeatureUsePenaltiesList"
      )
    }
    if (params.contains("perObjectFeaturePenaltiesMap") && params.contains("perObjectFeaturePenaltiesList")) {
      throw new CatBoostError(
        "params cannot contain both perObjectFeaturePenaltiesMap and perObjectFeaturePenaltiesList"
      )
    }
    if (params.contains("classWeightsMap") && params.contains("classWeightsList")) {
      throw new CatBoostError("params cannot contain both classWeightsMap and classWeightsList")
    }
    var classWeightsSize : Option[Int] = None
    if (params.contains("classWeightsMap")) {
      classWeightsSize = Some(params("classWeightsMap").asInstanceOf[java.util.LinkedHashMap[String, Double]].size)
    }
    if (params.contains("classWeightsList")) {
      classWeightsSize = Some(params("classWeightsList").asInstanceOf[Array[Double]].length)
    }
    if (params.contains("classNames") && classWeightsSize.isDefined) {
      if (params("classNames").asInstanceOf[Array[String]].length != classWeightsSize.get) {
        throw new CatBoostError("classWeights and classNames params contain different number of classes")
      }
    }
    if (classWeightsSize.isDefined && params.contains("autoClassWeights")) {
      throw new CatBoostError("params cannot contain both classWeights and autoClassWeights")
    }
    if (classWeightsSize.isDefined && params.contains("scalePosWeight")) {
      throw new CatBoostError("params cannot contain both classWeights and scalePosWeight")
    }
    if (params.contains("autoClassWeights") && params.contains("scalePosWeight")) {
      throw new CatBoostError("params cannot contain both autoClassWeights and scalePosWeight")
    }
    if (params.contains("classNames") && params.contains("classesCount")) {
      if (params("classNames").asInstanceOf[Array[String]].length != params("classesCount").asInstanceOf[Int]) {
        throw new CatBoostError("classNames and classesCount params contain different number of classes")
      }
    }
  }

  def classNamesAreKnown(params: Params) : Boolean = {
    params.isSet(params.getParam("targetBorder")) || params.isSet(params.getParam("classesCount")) || params.isSet(params.getParam("classNames")) || params.isSet(params.getParam("classWeightsMap"))|| params.isSet(params.getParam("classWeightsList"))
  }


  def processClassWeightsParams(
    params: mutable.HashMap[String, Any],
    classNamesFromLabelData: Option[Array[String]]
  ) : JObject = {
    if (params.contains("classWeightsMap")) {
      val classWeightsMap = params("classWeightsMap").asInstanceOf[java.util.LinkedHashMap[String, Double]]
      val classWeightsList = new Array[Double](classWeightsMap.size)
      var result = JObject()

      val maybeClassNames = if (classNamesFromLabelData.isDefined) {
        classNamesFromLabelData
      } else if (params.contains("classNames")) {
        Some(params("classNames").asInstanceOf[Array[String]])
      } else {
        None
      }

      if (maybeClassNames.isDefined) {
        val classNames = maybeClassNames.get
        for (i <- 0 until classWeightsList.size) {
          val className = classNames(i)
          if (!classWeightsMap.containsValue(className)) {
            throw new CatBoostError(
              s"Class '$className' is present in classNames but is not present in classWeightsMap"
            )
          }
          classWeightsList(i) = classWeightsMap.get(className).toDouble
        }
      } else {
        val classNames = new Array[String](classWeightsMap.size)
        var i = 0
        for ((className, classWeight) <- classWeightsMap.asScala) {
          classNames(i) = className
          classWeightsList(i) = classWeight.toDouble
          i = i + 1
        }
        result = result ~ ("class_names" -> classNames.toSeq)
      }
      result = result ~ ("class_weights" -> classWeightsList.toSeq)
      result
    } else if (params.contains("classWeightsList")) {
      JObject() ~ ("class_weights" -> params("classWeightsList").asInstanceOf[Array[Double]].toSeq)
    } else if (params.contains("scalePosWeight")) {
      JObject() ~ ("class_weights" -> Seq(1.0, params("scalePosWeight").asInstanceOf[Float].toDouble))
    } else {
      JObject()
    }
  }

  def processClassNamesFromLabelData(classNamesFromLabelData: Option[Array[String]]) : JObject = {
    classNamesFromLabelData match {
      case Some(classNames) => ("class_names" -> classNames.toSeq)
      case None => JObject()
    }
  }

  def processOverfittingDetectorParams(params: mutable.HashMap[String, Any]) : JObject = {
    var maybeOdWait = params.get("odWait")
    if (params.contains("earlyStoppingRounds")) {
      if (maybeOdWait.isDefined) {
        throw new CatBoostError("only one of the parameters earlyStoppingRounds, odWait should be initialized")
      }
      maybeOdWait = params.get("earlyStoppingRounds")
    }
    maybeOdWait match {
      case Some(odWait) => {
        if (params.contains("odType")) {
          if (params("odType").asInstanceOf[EOverfittingDetectorType] != EOverfittingDetectorType.Iter) {
            throw new CatBoostError(
              "odWait or earlyStoppingRounds parameter specified with odType != EOverfittingDetectorType.Iter"
            )
          }
        }
        JObject(
          "od_type" -> "Iter",
          "od_wait" -> JInt(BigInt(odWait.asInstanceOf[Int]))
        )
      }
      case None => {
        params.get("odPval") match {
          case Some(odPval) => {
            if (params.contains("odType")) {
              if (params("odType").asInstanceOf[EOverfittingDetectorType] != EOverfittingDetectorType.IncToDec) {
                throw new CatBoostError(
                  "odPval parameter specified with odType != EOverfittingDetectorType.IncToDec"
                )
              }
            }
            ("od_pval" -> JDouble(odPval.asInstanceOf[Float]))
          }
          case None => JObject()
        }
      }
    }
  }

  def processSnapshotIntervalParam(params: mutable.HashMap[String, Any]) : JObject = {
    if (params.contains("snapshotInterval")) {
      JObject() ~ (
        "snapshot_interval" -> JInt(BigInt(params("snapshotInterval").asInstanceOf[java.time.Duration].getSeconds))
      )
    } else {
      JObject()
    }
  }


  val namesMap = Map(
    "ignored_features_indices" -> "ignored_features",
    "ignored_features_names" -> "ignored_features",
    "feature_weights_map" -> "feature_weights",
    "feature_weights_list" -> "feature_weights",
    "first_feature_use_penalties_map" -> "first_feature_use_penalties",
    "first_feature_use_penalties_list" -> "first_feature_use_penalties",
    "per_object_feature_penalties_map" -> "per_object_feature_penalties",
    "per_object_feature_penalties_list" -> "per_object_feature_penalties",

    "spark_partition_count" -> null,
    "training_driver_listening_port" -> null,
    "worker_initialization_timeout" -> null,
    "worker_max_failures" -> null,
    "worker_listening_port" -> null,
    "connect_timeout" -> null,

    // processed in separate functions
    "class_weights_map" -> null,
    "class_weights_list" -> null,
    "scale_pos_weight" -> null,
    "early_stopping_rounds" -> null,
    "od_pval" -> null,
    "od_type" -> null,
    "od_wait" -> null,
    "snapshot_interval" -> null,

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

  def processWithSimpleNameMapping(paramsSeq: Seq[ParamPair[_]]) : JObject = {
    JObject(
      paramsSeq.collect {
        case ParamPair(param, value) if getMappedParamName(param.name) != null
          => (getMappedParamName(param.name) -> parse(param.jsonEncode(value)))
      }.toList
    )
  }

  def sparkMlParamsToCatBoostJsonParams(
    params: Params,
    classNamesFromLabelData: Option[Array[String]] = None
  ) : JObject = {
    val paramsSeq = params.extractParamMap.toSeq
    val paramsHashMap = new mutable.HashMap[String, Any]
    for (paramPair <- paramsSeq) {
      paramsHashMap += (paramPair.param.name -> paramPair.value)
    }
    checkIncompatibleParams(paramsHashMap)


    JObject()
      .merge(processClassWeightsParams(paramsHashMap, classNamesFromLabelData))
      .merge(processClassNamesFromLabelData(classNamesFromLabelData))
      .merge(processOverfittingDetectorParams(paramsHashMap))
      .merge(processSnapshotIntervalParam(paramsHashMap))
      .merge(processWithSimpleNameMapping(paramsSeq))
  }

  def sparkMlParamsToCatBoostJsonParamsString(params: Params) : String = {
    compact(sparkMlParamsToCatBoostJsonParams(params))
  }
}
