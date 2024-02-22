package ai.catboost.spark.impl.pyspark_wrapper_generator

import java.io.{File,PrintWriter}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe
import scala.reflect.runtime.universe._

import com.google.common.base.CaseFormat

import org.apache.spark.ml._
import org.apache.spark.ml.param._

import ai.catboost.spark._
import ai.catboost.CatBoostError

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl


object Generator {
  def generateCorePyPrologue(sparkCompatVersion: String, out: PrintWriter) = {
    out.println(
      s"""
import collections
import datetime
from enum import Enum
from typing import TYPE_CHECKING

from py4j.java_gateway import JavaObject

from pyspark import keyword_only, SparkContext
"""
    )

    sparkCompatVersion match {
      case "3.0" => out.println(s"""
from pyspark.ml.classification import JavaProbabilisticClassificationModel
from pyspark.ml.regression import JavaRegressionModel
"""
        )
      case "3.1" | "3.2" | "3.3" | "3.4" | "3.5" => out.println(s"""
from pyspark.ml.classification import _JavaProbabilisticClassificationModel
from pyspark.ml.regression import _JavaRegressionModel
"""
        )
      case _ => out.println(s"""
from pyspark.ml.classification import JavaClassificationModel
from pyspark.ml.util import JavaPredictionModel
from pyspark.ml.wrapper import JavaModel
"""
      )
    }

    out.println(
      s"""
import pyspark.ml.common
from pyspark.ml.common import inherit_doc
from pyspark.ml.param import Param, Params
from pyspark.ml.util import JavaMLReader, JavaMLWriter, JavaMLWritable, MLReadable
"""
    )

    out.println(
      s"""
import pyspark.ml.wrapper
from pyspark.ml.wrapper import JavaParams, JavaEstimator, JavaWrapper
from pyspark.sql import DataFrame, SparkSession


if TYPE_CHECKING:
    from pyspark.sql._typing import OptionalPrimitiveType

"\""
    original JavaParams._from_java has to be replaced because of hardcoded class names transformation
"\""

@staticmethod
def _from_java_patched_for_catboost(java_stage):
    "\""
    Given a Java object, create and return a Python wrapper of it.
    Used for ML persistence.

    Meta-algorithms such as Pipeline should override this method as a classmethod.
    "\""
    def __get_class(clazz):
        "\""
        Loads Python class from its name.
        "\""
        parts = clazz.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
    stage_name = (
        java_stage.getClass().getName()
            .replace("org.apache.spark", "pyspark")
            .replace("ai.catboost.spark", "catboost_spark")
    )
    # Generate a default new instance from the stage_name class.
    py_type = __get_class(stage_name)
    if issubclass(py_type, JavaParams):
        # Load information from java_stage to the instance.
        py_stage = py_type()
        py_stage._java_obj = java_stage
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()
    elif hasattr(py_type, "_from_java"):
        py_stage = py_type._from_java(java_stage)
    else:
        raise NotImplementedError("This Java stage cannot be loaded into Python currently: %r"
                                  % stage_name)
    return py_stage

JavaParams._from_java = _from_java_patched_for_catboost


"\""
    Adapt _py2java and _java2py for additional types present in CatBoost Params
"\""

_standard_py2java = pyspark.ml.common._py2java
_standard_java2py = pyspark.ml.common._java2py

def _py2java(sc, obj):
    "\"" Convert Python object into Java "\""
    if isinstance(obj, SparkSession):
        return obj._jsparkSession
    if isinstance(obj, Enum):
        return getattr(
            getattr(
                sc._jvm.ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl,
                obj.__class__.__name__
            ),
            'swigToEnum'
        )(obj.value)
    if isinstance(obj, datetime.timedelta):
        return sc._jvm.java.time.Duration.ofMillis(obj // datetime.timedelta(milliseconds=1))
    if isinstance(obj, JavaParams):
        return obj._to_java()
    if isinstance(obj, collections.OrderedDict):
        return sc._jvm.java.util.LinkedHashMap(obj)
    return _standard_py2java(sc, obj)

def _java2py(sc, r, encoding="bytes"):
    if isinstance(r, JavaObject):
        enumValues = r.getClass().getEnumConstants()
        if (enumValues is not None) and (len(enumValues) > 0):
            return globals()[r.getClass().getSimpleName()](r.swigValue())

        clsName = r.getClass().getName()
        if clsName == 'java.time.Duration':
            return datetime.timedelta(milliseconds=r.toMillis())
        if clsName == 'ai.catboost.spark.Pool':
            return Pool(r)
        if clsName == 'java.util.LinkedHashMap':
            return collections.OrderedDict(r)
    return _standard_java2py(sc, r, encoding)

pyspark.ml.common._py2java = _py2java
pyspark.ml.common._java2py = _java2py

pyspark.ml.wrapper._py2java = _py2java
pyspark.ml.wrapper._java2py = _java2py


@inherit_doc
class CatBoostMLReader(JavaMLReader):
    "\""
    (Private) Specialization of :py:class:`JavaMLReader` for CatBoost types
    "\""

    @classmethod
    def _java_loader_class(cls, clazz):
        "\""
        Returns the full class name of the Java ML instance.
        "\""
        java_package = clazz.__module__.replace("catboost_spark.core", "ai.catboost.spark")
        print("CatBoostMLReader._java_loader_class. ", java_package + "." + clazz.__name__)
        return java_package + "." + clazz.__name__

"""
    )
  }

  def jvmToPyValueAsString[T](obj: T) : String = {
    if (obj.isInstanceOf[java.time.Duration]) {
      val durationInMilliseconds = obj.asInstanceOf[java.time.Duration].toMillis()
      s"datetime.timedelta(milliseconds=$durationInMilliseconds)"
    } else if (obj.isInstanceOf[String]) {
      "\"" + obj.toString().replace("\t", "\\t") + "\""
    } else {
      obj.toString()
    }
  }

  /**
   * @return "param=value, ..."
   */
  def generateParamsKeywordArgs(params: Params) : String = {
    params.params.map(
      param => {
        val value = params.getDefault(param) match {
          case Some(value) => jvmToPyValueAsString(value)
          case None => "None"
        }
        s"${param.name}=$value"
      }
    ).mkString(", ")
  }

  def getParamNameToPythonTypeMap[Params : universe.TypeTag](obj: Params) : Map[String,String] = {
    val result = mutable.Map[String,String]()

    val paramReg = """.*Param\[([\w\.]+)\]""".r
    val enumParamReg = """.*EnumParam\[([\w\.]+)\]""".r

    for (member <- universe.typeOf[Params].members) {
      val pyType = member.typeSignature.typeSymbol.name.toString match {
        case "BooleanParam" => Some("bool")
        case "IntParam" => Some("int")
        case "LongParam" => Some("long")
        case "FloatParam" => Some("float")
        case "DoubleParam" => Some("double")
        case "StringParam" => Some("str")
        case "StringArrayParam" | "IntArrayParam" | "LongArrayParam" | "ByteArrayParam" | "DoubleArrayParam" => Some("list")
        case "MapArrayParam" | "MapParam" | "OrderedStringMapParam" => Some("dict")
        case "DurationParam" => Some("datetime.timedelta")
        case "Param" => {
          member.typeSignature.toString match {
            case paramReg(pType) if (pType.equals("String")) => Some("str")
            case _ => throw new RuntimeException(s"unexpected Param type: '${member.typeSignature.toString}'")
          }
        }
        case "EnumParam" => {
          member.typeSignature.toString match {
            case enumParamReg(enumType) => Some(enumType.split("\\.").last)
            case _ => throw new RuntimeException(s"EnumParam bad match: '${member.typeSignature.toString}'")
          }
        }
        case _ => None
      }
      pyType match {
        case Some(pyType) => {
          result += (member.name.toString.trim -> pyType)
        }
        case None => ()
      }
    }

    result.toMap
  }

  def getEnumNamesUsedInParams[Params : universe.TypeTag](obj: Params) : Set[String] = {
    val result = new mutable.HashSet[String]()

    val enumReg = """.*EnumParam\[([\w\.]+)\]""".r

    for (member <- universe.typeOf[Params].members) {
      val pyType = member.typeSignature.typeSymbol.name.toString match {
        case "EnumParam" => {
          result += (
            member.typeSignature.toString match {
              case enumReg(enumType) => enumType.split("\\.").last
              case _ => throw new RuntimeException(s"EnumParam bad match: '${member.typeSignature.toString}'")
            }
          )
        }
        case _ => None
      }
    }

    result.toSet
  }

  def patchJvmToPyEnumValue(enumValue: Object) : String = {
    val valueAsString = enumValue.toString
    if (valueAsString.equals("None")) { "No" } else { valueAsString }
  }


  def generateEnumDefinitions(enumNames: Set[String], out: PrintWriter) = {
    val sortedEnumNames = enumNames.toSeq.sorted

    for (enumName <- sortedEnumNames) {
      val enumClass = Class.forName(s"ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl.$enumName")

      out.print(
        s"""
class $enumName(Enum):
"""
      )
      val enumValues = enumClass.getEnumConstants
      for ((enumValue, idx) <- enumValues.zipWithIndex) {
        out.println(s"    ${patchJvmToPyEnumValue(enumValue)} = $idx")
      }
      out.println("")
    }
  }

  /**
   * @return "param : type, default: <value>
   *         "    <description>"
   *         ...
   */
  def generateParamsDocStrings[ParamsClass: universe.TypeTag](paramsClass: ParamsClass, tabShift: Int) : String = {
    val paramNameToPythonTypeMap = getParamNameToPythonTypeMap(paramsClass)
    val tabOffset = "    " * tabShift
    val params = paramsClass.asInstanceOf[Params]
    params.params.map(
      param => {
        val defaultValueDescription = params.getDefault(param) match {
          case Some(value) => s", default: ${jvmToPyValueAsString(value)}"
          case None => ""
        }
        s"$tabOffset${param.name} : ${paramNameToPythonTypeMap(param.name)}${defaultValueDescription}\n$tabOffset    ${param.doc}"
      }
    ).mkString("\n")
  }

  /**
   * @return "       self.<param> = Param(...)"
   *         "       self._setDefault(<param>=<value>)" ...
   */
  def generateParamsInitialization(params: Params) : String = {
    params.params.filter(p => p.name != "classWeights").map(
      param => {
        val paramInit = s"""        self.${param.name} = Param(self, "${param.name}", "${param.doc}")"""
        params.getDefault(param) match {
          case Some(value) => {
            paramInit ++ s"\n        self._setDefault(${param.name}=${jvmToPyValueAsString(value)})"
          }
          case None => paramInit
        }
      }
    ).mkString("\n")
  }


  def generateParamsGettersAndSetters[ParamsClass: universe.TypeTag](paramsClass: ParamsClass) : String = {
    val paramNameToPythonTypeMap = getParamNameToPythonTypeMap(paramsClass)
    val params = paramsClass.asInstanceOf[Params]
    params.params.filter(p => p.name != "classWeights").map(
      param => {
        val nameInMethods = CaseFormat.LOWER_CAMEL.to(CaseFormat.UPPER_CAMEL, param.name)
        s"""
    def get${nameInMethods}(self):
        "\""
        Returns
        -------
        ${paramNameToPythonTypeMap(param.name)}
            ${param.doc}
        "\""
        return self.getOrDefault(self.${param.name})

    def set${nameInMethods}(self, value):
        "\""
        Parameters
        ----------
        value : ${paramNameToPythonTypeMap(param.name)}
            ${param.doc}
        "\""
        self._set(${param.name}=value)
        return self

"""
      }
    ).mkString("\n")
  }

  /**
   * without __init__
   */
  def generateParamsPart[Params: universe.TypeTag](params: Params, paramsKeywordArgs: String) : String = {
    s"""
    @keyword_only
    def setParams(self, $paramsKeywordArgs):
        "\""
        Set the (keyword only) parameters

        Parameters
        ----------
${generateParamsDocStrings(params, tabShift=2)}
        "\""
        if hasattr(self, "_input_kwargs"):
            kwargs = self._input_kwargs
        else:
            kwargs = self.__init__._input_kwargs
        return self._set(**kwargs)

${generateParamsGettersAndSetters(params)}
"""
  }

  def generateStandardParamsWrapper[ParamsClass: universe.TypeTag](params: ParamsClass, out: PrintWriter) = {
    val paramsAsParamsClass = params.asInstanceOf[Params]
    val paramsKeywordArgs = generateParamsKeywordArgs(paramsAsParamsClass)
    val pyClassName = params.getClass.getSimpleName
    out.println(
      s"""
class $pyClassName(JavaParams):
    "\""
    Parameters
    ----------
${generateParamsDocStrings(params, tabShift=1)}
    "\""

    @keyword_only
    def __init__(self, $paramsKeywordArgs):
        super($pyClassName, self).__init__()
        self._java_obj = self._new_java_obj("${params.getClass.getCanonicalName}")
${generateParamsInitialization(paramsAsParamsClass)}

        if hasattr(self, "_input_kwargs"):
            kwargs = self._input_kwargs
        else:
            kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

${generateParamsPart(params, paramsKeywordArgs)}
"""
    )
  }

  def generateForwardedAccessors(accessors: Seq[(String, String)]) : String = {
    accessors.map{
      case (accessor, docString) => s"""
    def $accessor(self):
        "\""
        $docString
        "\""
        return self._call_java("$accessor")
"""
    }.mkString("\n")
  }


  def generatePoolSerializationWrappers(out: PrintWriter) = {

    out.println(
      s"""
class PoolReader(JavaWrapper):
    "\""
    This class is used to load a :class:`Pool` from external storage systems (it is analogous to PySpark's DataFrameReader).
    Use Pool.read() to get it.
    "\""
    def __init__(self, sparkSession: "SparkSession"):
        super(PoolReader, self).__init__(JavaWrapper._new_java_obj("ai.catboost.spark.PoolReader", sparkSession))

    def dataFramesReaderFormat(self, source: str) -> "PoolReader":
        self._java_obj = self._java_obj.dataFramesReaderFormat(source)
        return self

    def dataFramesReaderOption(self, key: str, value: "OptionalPrimitiveType") -> "PoolReader":
        self._java_obj = self._java_obj.dataFramesReaderOption(key, to_str(value))
        return self

    def dataFramesReaderOptions(self, **options: "OptionalPrimitiveType") -> "PoolReader":
        for k in options:
            self._java_obj = self._java_obj.dataFramesReaderOption(key, to_str(options[k]))
        return self

    def load(self, path: str) -> "Pool":
        return Pool(self._java_obj.load(path))


class PoolWriter(JavaWrapper):
    "\""
    This class is used to save :class:`Pool` to external storage systems (it is analogous to PySpark's DataFrameWriter).
      Use Pool.write() to get it.
    "\""
    def __init__(self, pool: "Pool"):
        super(PoolWriter, self).__init__(pool._java_obj.write())

    def dataFramesWriterFormat(self, source: str) -> "PoolWriter":
        self._java_obj = self._java_obj.dataFramesWriterFormat(source)
        return self

    def dataFramesWriterOption(self, key: str, value: "OptionalPrimitiveType") -> "PoolWriter":
        self._java_obj = self._java_obj.dataFramesWriterOption(key, to_str(value))
        return self

    def dataFramesWriterOptions(self, **options: "OptionalPrimitiveType") -> "PoolWriter":
        for k in options:
            self._java_obj = self._java_obj.dataFramesWriterOption(key, to_str(options[k]))
        return self

    def mode(self, saveModeArg: str) -> "PoolWriter":
        self._java_obj = self._java_obj.mode(saveModeArg)
        return self

    def save(self, path: str) -> None:
        self._java_obj.save(path)

"""
    )
  }

  def generatePoolWrapper(out: PrintWriter) = {
    generatePoolSerializationWrappers(out)

    val pool = new Pool(null)

    val paramsKeywordArgs = generateParamsKeywordArgs(pool)
    val forwardedAccessors = Seq(
      ("isQuantized","Returns whether the main `data` has already been quantized."),
      ("getFeatureCount", "Returns the number of features."),
      ("getFeatureNames", "Returns the list of feature names."),
      ("count", "Returns the number of rows in the main `data` DataFrame."),
      ("pairsCount", "Returns the number of rows in the `pairsData` DataFrame."),
      ("getBaselineCount", "Returns the dimension of the baseline data (0 if not specified).")
    )

    out.println(
      s"""
class Pool(JavaParams):
    "\""
    CatBoost's abstraction of a dataset.
    Features data can be stored in raw (features column has pyspark.ml.linalg.Vector type)
    or quantized (float feature values are quantized into integer bin values, features column has
    Array[Byte] type) form.

    Raw Pool can be transformed to quantized form using `quantize` method.
    This is useful if this dataset is used for training multiple times and quantization parameters do not
    change. Pre-quantized Pool allows to cache quantized features data and so do not re-run
    feature quantization step at the start of an each training.
    "\""
    def __init__(self, data_frame_or_java_object, pairs_data_frame=None):
        "\""
        Construct Pool from DataFrame, optionally specifying pairs data in an additional DataFrame.
        "\""
        if isinstance(data_frame_or_java_object, JavaObject):
            java_obj = data_frame_or_java_object
        else:
            java_obj = JavaWrapper._new_java_obj("ai.catboost.spark.Pool", data_frame_or_java_object, pairs_data_frame)

        super(Pool, self).__init__(java_obj)
${generateParamsInitialization(pool)}

${generateParamsPart(pool, paramsKeywordArgs)}

    def _call_java(self, name, *args):
        self._transfer_params_to_java()
        return JavaWrapper._call_java(self, name, *args)

${generateForwardedAccessors(forwardedAccessors)}

    @property
    def data(self):
        "\""
        DataFrame with the main data (features, label, (optionally) weight etc.)
        "\""
        return self._call_java("data")

    @property
    def pairsData(self):
        "\""
        DataFrame with the pairs data (groupId, winnerId, loserId and optionally weight).
        Can be None.
        "\""
        return self._call_java("pairsData")

    def quantize(self, quantizationParams = None):
        "\""Create Pool with quantized features from Pool with raw features"\""
        if quantizationParams is None:
            quantizationParams = QuantizationParams()
        return self._call_java("quantize", quantizationParams)

    def repartition(self, partitionCount, byGroupColumnsIfPresent):
        "\""
        Repartion data to the specified number of partitions.
        Useful to repartition data to create one partition per executor for training
        (where each executor gets its' own CatBoost worker with a part of the training data).
        "\""
        return self._call_java("repartition", partitionCount, byGroupColumnsIfPresent)

    @staticmethod
    def load(sparkSession, dataPathWithScheme, columnDescription=None, poolLoadParams=None, pairsDataPathWithScheme=None):
        "\""
        Load dataset in one of CatBoost's natively supported formats:
           * dsv - https://catboost.ai/docs/concepts/input-data_values-file.html
           * libsvm - https://catboost.ai/docs/concepts/input-data_libsvm.html

        Parameters
        ----------
        sparkSession : SparkSession
        dataPathWithScheme : str
            Path with scheme to dataset in CatBoost format.
            For example, `dsv:///home/user/datasets/my_dataset/train.dsv` or
            `libsvm:///home/user/datasets/my_dataset/train.libsvm`
        columnDescription : str, optional
            Path to column description file. See https://catboost.ai/docs/concepts/input-data_column-descfile.html
        params : PoolLoadParams, optional
            Additional params specifying data format.
        pairsDataPathWithScheme : str, optional
            Path with scheme to dataset pairs in CatBoost format.
            Only "dsv-grouped" format is supported for now.
            For example, `dsv-grouped:///home/user/datasets/my_dataset/train_pairs.dsv`

        Returns
        -------
           Pool
               Pool containing loaded data
        "\""
        if poolLoadParams is None:
            poolLoadParams = PoolLoadParams()
        sc = sparkSession.sparkContext
        java_obj = sc._jvm.ai.catboost.spark.Pool.load(
            _py2java(sc, sparkSession),
            dataPathWithScheme,
            (sc._jvm.java.nio.file.Paths.get(columnDescription, sc._gateway.new_array(sc._jvm.String, 0))
             if columnDescription
             else None
            ),
            _py2java(sc, poolLoadParams),
            pairsDataPathWithScheme
        )
        return Pool(java_obj)

    @staticmethod
    def read(sparkSession) -> "PoolReader":
        "\""
        Interface for reading the content from external storage (API similar to PySpark's DataFrameReader)

        Returns
        -------
            PoolReader
                PoolReader helper class to read Pool data
        "\""
        return PoolReader(sparkSession)

    @property
    def write(self) -> "PoolWriter":
        "\""
        Interface for saving the content out into external storage (API similar to PySpark's DataFrameWriter)

        Returns
        -------
            PoolWriter
                PoolWriter helper class to save Pool data
        "\""
        return PoolWriter(self)

"""
    )
  }

  def generateEstimatorAndModelWrapper[EstimatorClass: universe.TypeTag, ModelClass: universe.TypeTag](
    estimator: EstimatorClass,
    model: ModelClass,
    modelBaseClassName: String,
    estimatorDoc: String,
    modelDoc: String,
    sparkCompatVersion: String,
    out: PrintWriter
  ) = {
    val estimatorClassName = typeOf[EstimatorClass].typeSymbol.name.toString.split("\\.").last
    val estimatorAsParams = estimator.asInstanceOf[Params]
    val estimatorParamsKeywordArgs = generateParamsKeywordArgs(estimatorAsParams)

    val modelClassName = typeOf[ModelClass].typeSymbol.name.toString.split("\\.").last
    val modelAsParams = model.asInstanceOf[Params]
    val modelParamsKeywordArgs = generateParamsKeywordArgs(modelAsParams)


    out.println(
      s"""

@inherit_doc
class $estimatorClassName(JavaEstimator, MLReadable, JavaMLWritable):
    "\""
    $estimatorDoc

    Init Parameters
    ---------------
${generateParamsDocStrings(estimator, tabShift=2)}
    "\""

    @keyword_only
    def __init__(self, $estimatorParamsKeywordArgs):
        super($estimatorClassName, self).__init__()
        self._java_obj = self._new_java_obj("ai.catboost.spark.$estimatorClassName")
${generateParamsInitialization(estimatorAsParams)}

        if hasattr(self, "_input_kwargs"):
            kwargs = self._input_kwargs
        else:
            kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

${generateParamsPart(estimator, estimatorParamsKeywordArgs)}

    @classmethod
    def read(cls):
        "\""Returns an MLReader instance for this class."\""
        return CatBoostMLReader(cls)

    def _create_model(self, java_model):
        return $modelClassName(java_model)

    def _fit_with_eval(self, trainDatasetAsJavaObject, evalDatasetsAsJavaObject, params=None):
        "\""
        Implementation of fit with eval datasets with no more than one set of optional parameters
        "\""
        if params:
            return self.copy(params)._fit_with_eval(trainDatasetAsJavaObject, evalDatasetsAsJavaObject)
        else:
            self._transfer_params_to_java()
            java_model = self._java_obj.fit(trainDatasetAsJavaObject, evalDatasetsAsJavaObject)
            return $modelClassName(java_model)

    def fit(self, dataset, params=None, evalDatasets=None):
        "\""
        Extended variant of standard Estimator's fit method
        that accepts CatBoost's Pool s and allows to specify additional
        datasets for computing evaluation metrics and overfitting detection similarily to CatBoost's other APIs.

        Parameters
        ----------
        dataset : Pool or DataFrame
          The input training dataset.
        params : dict or list or tuple, optional
          an optional param map that overrides embedded params. If a list/tuple of
          param maps is given, this calls fit on each param map and returns a list of
          models.
        evalDatasets : Pools, optional
          The validation datasets used for the following processes:
           - overfitting detector
           - best iteration selection
           - monitoring metrics' changes

        Returns
        -------
        trained model(s): $modelClassName or a list of trained $modelClassName
        "\""
        if (isinstance(dataset, DataFrame)):
            if evalDatasets is not None:
                raise RuntimeError("if dataset has type DataFrame no evalDatasets are supported")
            return JavaEstimator.fit(self, dataset, params)
        else:
            sc = SparkContext._active_spark_context

            trainDatasetAsJavaObject = _py2java(sc, dataset)
            evalDatasetCount = 0 if (evalDatasets is None) else len(evalDatasets)

            # need to create it because default mapping for python list is ArrayList, not Array
            evalDatasetsAsJavaObject = sc._gateway.new_array(sc._jvm.ai.catboost.spark.Pool, evalDatasetCount)
            for i in range(evalDatasetCount):
                evalDatasetsAsJavaObject[i] = _py2java(sc, evalDatasets[i])

            def _fit_with_eval(params):
                return self._fit_with_eval(trainDatasetAsJavaObject, evalDatasetsAsJavaObject, params)

            if (params is None) or isinstance(params, dict):
                return _fit_with_eval(params)
            if isinstance(params, (list, tuple)):
                models = []
                for paramsInstance in params:
                    models.append(_fit_with_eval(paramsInstance))
                return models
            else:
                raise TypeError("Params must be either a param map or a list/tuple of param maps, "
                                "but got %s." % type(params))

@inherit_doc"""
    )

    if (sparkCompatVersion.startsWith("3.")) {
      out.print(s"class $modelClassName($modelBaseClassName, MLReadable, JavaMLWritable):")
    } else {
      out.print(s"class $modelClassName(JavaModel, $modelBaseClassName, MLReadable, JavaMLWritable):")
    }

  out.println(
      s"""
    "\""
    $modelDoc
    "\""
    def __init__(self, java_model=None):
        super($modelClassName, self).__init__(java_model)
${generateParamsInitialization(modelAsParams)}
        if java_model is not None:
            self._transfer_params_from_java()

${generateParamsPart(model, modelParamsKeywordArgs)}

    @staticmethod
    def _from_java(java_model):
        return $modelClassName(java_model)

    @classmethod
    def read(cls):
        "\""Returns an MLReader instance for this class."\""
        return CatBoostMLReader(cls)

    def saveNativeModel(self, fileName, format=EModelType.CatboostBinary, exportParameters=None, pool=None):
        "\""
        Save the model to a local file.
        See https://catboost.ai/docs/concepts/python-reference_catboostclassifier_save_model.html
        for detailed parameters description
        "\""
        return self._call_java("saveNativeModel", fileName, format, exportParameters, pool)

    @staticmethod
    def loadNativeModel(fileName, format=EModelType.CatboostBinary):
        "\""
        Load the model from a local file.
        See https://catboost.ai/docs/concepts/python-reference_catboostclassifier_load_model.html
        for detailed parameters description
        "\""
        sc = SparkContext._active_spark_context
        java_model = sc._jvm.ai.catboost.spark.$modelClassName.loadNativeModel(fileName, _py2java(sc, format))
        return $modelClassName(java_model)


    def transformPool(self, pool):
        "\""
        This function is useful when the dataset has been already quantized but works with any Pool
        "\""
        return self._call_java("transformPool", pool)


    def getFeatureImportance(self,
                             fstrType=EFstrType.FeatureImportance,
                             data=None,
                             calcType=ECalcTypeShapValues.Regular
                            ):
        "\""
        Parameters
        ----------
        fstrType : EFstrType
            Supported values are FeatureImportance, PredictionValuesChange, LossFunctionChange, PredictionDiff
        data : Pool
            if fstrType is PredictionDiff it is required and must contain 2 samples
            if fstrType is PredictionValuesChange this param is required in case if model was explicitly trained
            with flag to store no leaf weights.
            otherwise it can be null
        calcType : ECalcTypeShapValues
            Used only for PredictionValuesChange.
            Possible values:
              - Regular
                 Calculate regular SHAP values
              - Approximate
                 Calculate approximate SHAP values
              - Exact
                 Calculate exact SHAP values

        Returns
        -------
        list of float
            array of feature importances (index corresponds to the order of features in the model)
        "\""
        return self._call_java("getFeatureImportance", fstrType, data, calcType)

    def getFeatureImportancePrettified(self,
                                       fstrType=EFstrType.FeatureImportance,
                                       data=None,
                                       calcType=ECalcTypeShapValues.Regular
                                      ):
        "\""
        Parameters
        ----------
        fstrType : EFstrType
            Supported values are FeatureImportance, PredictionValuesChange, LossFunctionChange, PredictionDiff
        data : Pool
            if fstrType is PredictionDiff it is required and must contain 2 samples
            if fstrType is PredictionValuesChange this param is required in case if model was explicitly trained
            with flag to store no leaf weights.
            otherwise it can be null
        calcType : ECalcTypeShapValues
            Used only for PredictionValuesChange.
            Possible values:

              - Regular
                 Calculate regular SHAP values
              - Approximate
                 Calculate approximate SHAP values
              - Exact
                 Calculate exact SHAP values

        Returns
        -------
        list of FeatureImportance
            array of feature importances sorted in descending order by importance
        "\""
        return self._call_java("getFeatureImportancePrettified", fstrType, data, calcType)

    def getFeatureImportanceShapValues(self,
                                       data,
                                       preCalcMode=EPreCalcShapValues.Auto,
                                       calcType=ECalcTypeShapValues.Regular,
                                       modelOutputType=EExplainableModelOutput.Raw,
                                       referenceData=None,
                                       outputColumns=None
                                      ):
        "\""
        Parameters
        ----------
        data : Pool
            dataset to calculate SHAP values for
        preCalcMode : EPreCalcShapValues
            Possible values:
               - Auto
                  Use direct SHAP Values calculation only if data size is smaller than average leaves number
                  (the best of two strategies below is chosen).
               - UsePreCalc
                  Calculate SHAP Values for every leaf in preprocessing. Final complexity is
                  O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
                  D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
                  This is much faster (because of a smaller constant) than direct calculation when N >> L
               - NoPreCalc
                  Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
                  is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)
        calcType : ECalcTypeShapValues
            Possible values:

              - Regular
                 Calculate regular SHAP values
              - Approximate
                 Calculate approximate SHAP values
              - Exact
                 Calculate exact SHAP values
        referenceData : Pool
            reference data for Independent Tree SHAP values from https://arxiv.org/abs/1905.04610v1
            if referenceData is not null, then Independent Tree SHAP values are calculated
        outputColumns : list of str
            columns from data to add to output DataFrame, if None - add all columns

        Returns
        -------
        DataFrame
            - for regression and binclass models:
              contains outputColumns and "shapValues" column with Vector of length (n_features + 1) with SHAP values
            - for multiclass models:
              contains outputColumns and "shapValues" column with Matrix of shape (n_classes x (n_features + 1)) with SHAP values
        "\""
        return self._call_java(
            "getFeatureImportanceShapValues",
            data,
            preCalcMode,
            calcType,
            modelOutputType,
            referenceData,
            outputColumns
        )

    def getFeatureImportanceShapInteractionValues(self,
                                                  data,
                                                  featureIndices=None,
                                                  featureNames=None,
                                                  preCalcMode=EPreCalcShapValues.Auto,
                                                  calcType=ECalcTypeShapValues.Regular,
                                                  outputColumns=None):
        "\""
        SHAP interaction values are calculated for all features pairs if nor featureIndices nor featureNames
          are specified.

        Parameters
        ----------
        data : Pool
            dataset to calculate SHAP interaction values
        featureIndices : (int, int), optional
            pair of features indices to calculate SHAP interaction values for.
        featureNames : (str, str), optional
            pair of features names to calculate SHAP interaction values for.
        preCalcMode : EPreCalcShapValues
            Possible values:

            - Auto
                Use direct SHAP Values calculation only if data size is smaller than average leaves number
                (the best of two strategies below is chosen).
            - UsePreCalc
                Calculate SHAP Values for every leaf in preprocessing. Final complexity is
                O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
                D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
                This is much faster (because of a smaller constant) than direct calculation when N >> L
            - NoPreCalc
                Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
                is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)
        calcType : ECalcTypeShapValues
            Possible values:

              - Regular
                  Calculate regular SHAP values
              - Approximate
                  Calculate approximate SHAP values
              - Exact
                  Calculate exact SHAP values
        outputColumns : list of str
            columns from data to add to output DataFrame, if None - add all columns

        Returns
        -------
        DataFrame
            - for regression and binclass models:
              contains outputColumns and "featureIdx1", "featureIdx2", "shapInteractionValue" columns
            - for multiclass models:
              contains outputColumns and "classIdx", "featureIdx1", "featureIdx2", "shapInteractionValue" columns
        "\""
        return self._call_java(
            "getFeatureImportanceShapInteractionValues",
            data,
            featureIndices,
            featureNames,
            preCalcMode,
            calcType,
            outputColumns
        )

    def getFeatureImportanceInteraction(self):
        "\""
        Returns
        -------
        list of FeatureInteractionScore
        "\""
        return self._call_java("getFeatureImportanceInteraction")

"""
    )
    if (!sparkCompatVersion.startsWith("3.") && (modelBaseClassName == "JavaClassificationModel")) {
      // Add methods that are defined in JavaClassificationModel only since Spark 3.0.0
      out.println(
s"""
    def predictRaw(self, value):
        "\""
        Raw prediction for each possible label.
        "\""
        return self._call_java("predictRaw", value)

    def predictProbability(self, value):
        "\""
        Predict the probability of each class given the features.
        "\""
        return self._call_java("predictProbability", value)
"""
      )
    }
  }

  def generateVersionPy(modulePath: File, version: String) = {
    val versionPyWriter = new PrintWriter(new File(modulePath, "version.py"))
    try {
      versionPyWriter.println(s"VERSION = '$version'")
    } finally {
      versionPyWriter.close
    }
  }

  def generateInitPy(modulePath: File, enumsUsedInParams: Set[String]) = {
    val exportList = Seq(
        "PoolLoadParams",
        "QuantizationParams",
        "PoolReader",
        "PoolWriter",
        "Pool",
        "CatBoostClassificationModel",
        "CatBoostClassifier",
        "CatBoostRegressionModel",
        "CatBoostRegressor"
    ) ++ enumsUsedInParams.toSeq.sorted

    val initPyWriter = new PrintWriter(new File(modulePath, "__init__.py"))
    try {
      initPyWriter.print("""
from .version import VERSION as __version__  # noqa
from .core import (
"""
      )
      for (symbol <- exportList) {
        initPyWriter.println(s"    $symbol,")
      }
      initPyWriter.print("""
)
__all__ = [
"""
      )
      for (symbol <- exportList) {
        initPyWriter.println(s"    '$symbol',")
      }
      initPyWriter.println("]")
    } finally {
      initPyWriter.close
    }
  }

  /**
   * @param args expects 3 arguments:
   *  1) package version
   *  2) output dir
   *  3) spark compat version (like '2.4')
   */
  def main(args: Array[String]) : Unit = {
    try {
      val sparkCompatVersion = args(2)

      val modulePath = new File(args(1))
      modulePath.mkdirs()

      generateVersionPy(modulePath, args(0))

      val enumsUsedInParams = (
          getEnumNamesUsedInParams(new params.QuantizationParams)
          ++ getEnumNamesUsedInParams(new Pool(null))
          ++ getEnumNamesUsedInParams(new CatBoostClassifier)
          ++ getEnumNamesUsedInParams(new CatBoostRegressor)
          + "EModelType"
          ++ Set("EFstrType", "ECalcTypeShapValues", "EPreCalcShapValues", "EExplainableModelOutput")
      )

      generateInitPy(modulePath, enumsUsedInParams)

      val corePyWriter = new PrintWriter(new File(modulePath, "core.py"))
      try {
        generateCorePyPrologue(sparkCompatVersion, corePyWriter)
        generateStandardParamsWrapper(new params.PoolLoadParams(), corePyWriter)
        generateStandardParamsWrapper(new params.QuantizationParams(), corePyWriter)
        generatePoolWrapper(corePyWriter)
        generateEnumDefinitions(enumsUsedInParams, corePyWriter)
        generateEstimatorAndModelWrapper(
          new CatBoostRegressor,
          new CatBoostRegressionModel(new native_impl.TFullModel()),
          sparkCompatVersion match {
            case "3.0" => "JavaRegressionModel"
            case "3.1" | "3.2" | "3.3" | "3.4" | "3.5" => "_JavaRegressionModel"
            case _ => "JavaPredictionModel"
          },
          "Class to train CatBoostRegressionModel",
          "Regression model trained by CatBoost. Use CatBoostRegressor to train it",
          sparkCompatVersion,
          corePyWriter
        )
        generateEstimatorAndModelWrapper(
          new CatBoostClassifier,
          new CatBoostClassificationModel(new native_impl.TFullModel()),
          sparkCompatVersion match {
            case "3.0" => "JavaProbabilisticClassificationModel"
            case "3.1" | "3.2" | "3.3" | "3.4" | "3.5" => "_JavaProbabilisticClassificationModel"
            case _ => "JavaClassificationModel"
          },
          "Class to train CatBoostClassificationModel",
          "Classification model trained by CatBoost. Use CatBoostClassifier to train it",
          sparkCompatVersion,
          corePyWriter
        )
      } finally {
        corePyWriter.close
      }
    } catch {
      case t : Throwable => {
        t.printStackTrace()
        sys.exit(1)
      }
    }
    sys.exit(0)
  }
}
