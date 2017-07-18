# coding: utf-8
# cython: wraparound=False

from six import iteritems, string_types, PY3
from json import dumps, loads
from copy import deepcopy
from collections import Sequence, defaultdict

from cython.operator cimport dereference

from libcpp cimport bool as bool_t
from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from util.generic.string cimport TString
from util.generic.string cimport TStringBuf
from util.generic.vector cimport yvector
from util.generic.maybe cimport TMaybe
from util.generic.hash cimport yhash


class CatboostError(Exception):
    pass

cdef public object PyCatboostExceptionType = <object>CatboostError

cdef extern from "catboost/libs/logging/logging.h":
    cdef void SetCustomLoggingFunction(void(*func)(const char*, size_t len) except * with gil)
    cdef void RestoreOriginalLogger()

cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef void ProcessException()
    cdef void SetPythonInterruptHandler() nogil
    cdef void ResetPythonInterruptHandler() nogil

cdef extern from "catboost/libs/data/pool.h":
    cdef cppclass TDocInfo:
        float Target
        float Weight
        yvector[float] Factors
        yvector[double] Baseline
        void Swap(TDocInfo& other) except +ProcessException

    cdef cppclass TPool:
        yvector[TDocInfo] Docs
        yvector[int] CatFeatures
        yvector[TString] FeatureId
        yhash[int, TString] CatFeaturesHashToString

cdef extern from "catboost/libs/data/load_data.h":
    cdef void ReadPool(const string& fdFile,
                        const string& poolFile,
                        int threadCount,
                        bool_t verbose,
                        TPool* pool,
                        const char fieldDelimiter,
                        bool_t has_header) nogil except +ProcessException
    cdef int CalcCatFeatureHash(const TStringBuf& feature) except +ProcessException
    cdef float ConvertCatFeatureHashToFloat(int hashVal) except +ProcessException

cdef extern from "catboost/libs/model/tensor_struct.h":
    cdef cppclass TTensorStructure3:
        pass

cdef extern from "catboost/libs/model/model.h":

    cdef cppclass TFullModel:
        TString ParamsJson
        yvector[int] CatFeatures
        yvector[TTensorStructure3] TreeStruct
        void Swap(TFullModel& other) except +ProcessException

    cdef cppclass EModelExportType:
        pass

    cdef EModelExportType EModelExportType_Catboost "EModelExportType::CatboostBinary"
    cdef EModelExportType EModelExportType_CoreML "EModelExportType::AppleCoreML"

    cdef void ExportModel(const TFullModel& model, const TString& modelFile, const EModelExportType format, const TString& userParametersJSON ) except +ProcessException

    cdef void OutputModel(const TFullModel& model, const string& modelFile) except +ProcessException
    cdef TFullModel ReadModel(const string& modelFile) nogil except +ProcessException
    cdef TString SerializeModel(const TFullModel& model) except +ProcessException
    cdef TFullModel DeserializeModel(const TString& serializeModelString) nogil except +ProcessException

cdef extern from "library/json/writer/json_value.h" namespace "NJson":
    cdef cppclass TJsonValue:
        pass

cdef extern from "library/containers/2d_array/2d_array.h":
    cdef cppclass TArray2D[T]:
        T* operator[] (size_t index) const

cdef extern from "catboost/libs/algo/error_holder.h":
    cdef cppclass TErrorHolder:
        double Error
        double Weight

cdef extern from "catboost/libs/algo/ders_holder.h":
    cdef cppclass TDer1Der2:
        double Der1
        double Der2

cdef extern from "catboost/libs/algo/params.h":
    cdef TJsonValue ReadTJsonValue(const TString& paramsJson) nogil except +ProcessException
    cdef cppclass EPredictionType:
        pass

    cdef EPredictionType EPredictionType_Class "EPredictionType::Class"
    cdef EPredictionType EPredictionType_Probability "EPredictionType::Probability"
    cdef EPredictionType EPredictionType_RawFormulaVal "EPredictionType::RawFormulaVal"

    cdef cppclass TCustomMetricDescriptor:
        void* CustomData
        TErrorHolder (*EvalFunc)(const yvector[yvector[double]]& approx,
                                 const yvector[float]& target,
                                 const yvector[float]& weight,
                                 int begin, int end, void* customData) except * with gil
        TString (*GetDescriptionFunc)(void *customData) except * with gil
        bool_t (*IsMaxOptimalFunc)(void *customData) except * with gil
        double (*GetFinalErrorFunc)(const TErrorHolder& error, void *customData) except * with gil

    cdef cppclass TCustomObjectiveDescriptor:
        void* CustomData
        void (*CalcDersRange)(int count, const double* approxes, const float* targets,
                              const float* weights, TDer1Der2* ders, void* customData) except * with gil
        void (*CalcDersMulti)(const yvector[double]& approx, float target, float weight,
                              yvector[double]* ders, TArray2D[double]* der2, void* customData) except * with gil

    cdef cppclass TCrossValidationParams:
        size_t FoldCount
        bool_t Inverted
        int RandSeed
        bool_t Shuffle
        int EvalPeriod
        bool_t EnableEarlyStopping

cdef extern from "catboost/libs/algo/train_model.h":
    cdef void TrainModel(const TJsonValue& params,
                         const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
                         const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
                         TPool& learnPool,
                         const TPool& testPool,
                         const TString& outputModelPath,
                         TFullModel* model,
                         yvector[yvector[double]]* testApprox) nogil except +ProcessException

cdef extern from "catboost/libs/algo/cross_validation.h":
    cdef cppclass TCVResult:
        TString Metric
        yvector[double] AverageTrain
        yvector[double] StdDevTrain
        yvector[double] AverageTest
        yvector[double] StdDevTest

    cdef void CrossValidate(const TJsonValue& jsonParams,
                            const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
                            const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
                            TPool& pool,
                            const TCrossValidationParams& cvParams,
                            yvector[TCVResult]* results) nogil except +ProcessException

cdef extern from "catboost/libs/algo/apply.h":
    cdef yvector[double] ApplyModel(const TFullModel& model,
                                    const TPool& pool,
                                    bool_t verbose,
                                    const EPredictionType predictionType,
                                    int begin,
                                    int end,
                                    int threadCount) nogil except +ProcessException

    cdef yvector[yvector[double]] ApplyModelMulti(const TFullModel& model,
                                                  const TPool& pool,
                                                  bool_t verbose,
                                                  const EPredictionType predictionType,
                                                  int begin,
                                                  int end,
                                                  int threadCount) nogil except +ProcessException

cdef extern from "catboost/libs/algo/calc_fstr.h":
    cdef yvector[double] CalcRegularFeatureEffect(const TFullModel& model,
                                                const TPool& pool,
                                                int threadCount) except +ProcessException

cdef TString _MetricGetDescription(void* customData) except * with gil:
    cdef metricObject = <object>customData
    return TString(<const char*>metricObject.__class__.__name__)

cdef bool_t _MetricIsMaxOptimal(void* customData) except * with gil:
    cdef metricObject = <object>customData
    return metricObject.is_max_optimal()

cdef double _MetricGetFinalError(const TErrorHolder& error, void *customData) except * with gil:
    cdef metricObject = <object>customData
    return metricObject.get_final_error(error.Error, error.Weight)

cdef class _FloatArrayWrapper:
    cdef const float* _arr
    cdef int _count

    @staticmethod
    cdef create(const float* arr, int count):
        wrapper = _FloatArrayWrapper()
        wrapper._arr = arr
        wrapper._count = count
        return wrapper

    def __getitem__(self, key):
        if key >= self._count:
            raise IndexError()

        return self._arr[key]

    def __len__(self):
        return self._count

# Cython does not have generics so using small copy-paste here and below
cdef class _DoubleArrayWrapper:
    cdef const double* _arr
    cdef int _count

    @staticmethod
    cdef create(const double* arr, int count):
        wrapper = _DoubleArrayWrapper()
        wrapper._arr = arr
        wrapper._count = count
        return wrapper

    def __getitem__(self, key):
        if key >= self._count:
            raise IndexError()

        return self._arr[key]

    def __len__(self):
        return self._count

cdef TErrorHolder _MetricEval(const yvector[yvector[double]]& approx,
                              const yvector[float]& target,
                              const yvector[float]& weight,
                              int begin,
                              int end,
                              void* customData) except * with gil:
    cdef metricObject = <object>customData
    cdef TErrorHolder holder

    approxes = [_DoubleArrayWrapper.create(approx[i].data() + begin, end - begin) for i in xrange(approx.size())]
    targets = _FloatArrayWrapper.create(target.data() + begin, end - begin)

    if weight.size() == 0:
        weights = None
    else:
        weights = _FloatArrayWrapper.create(weight.data() + begin, end - begin)

    error, weight_ = metricObject.evaluate(approxes, targets, weights)

    holder.Error = error
    holder.Weight = weight_
    return holder

cdef void _ObjectiveCalcDersRange(int count, const double* approxes, const float* targets,
                                  const float* weights, TDer1Der2* ders, void* customData) except * with gil:
    cdef objectiveObject = <object>(customData)

    approx = _DoubleArrayWrapper.create(approxes, count)
    target = _FloatArrayWrapper.create(targets, count)

    if weights:
        weight = _FloatArrayWrapper.create(weights, count)
    else:
        weight = None

    result = objectiveObject.calc_ders_range(approx, target, weight)
    index = 0
    for der1, der2 in result:
        ders[index].Der1 = der1
        ders[index].Der2 = der2
        index += 1

cdef void _ObjectiveCalcDersMulti(const yvector[double]& approx, float target, float weight,
                                  yvector[double]* ders, TArray2D[double]* der2, void* customData) except * with gil:
    cdef objectiveObject = <object>(customData)

    approxes = _DoubleArrayWrapper.create(approx.data(), approx.size())

    ders_vector, second_ders_matrix = objectiveObject.calc_ders_multi(approxes, target, weight)
    for index, der in enumerate(ders_vector):
        dereference(ders)[index] = der

    for ind1, line in enumerate(second_ders_matrix):
        for ind2, num in enumerate(line):
            dereference(der2)[ind1][ind2] = num

cdef TCustomMetricDescriptor _BuildCustomMetricDescriptor(object metricObject):
    cdef TCustomMetricDescriptor descriptor
    descriptor.CustomData = <void*>metricObject
    descriptor.EvalFunc = &_MetricEval
    descriptor.GetDescriptionFunc = &_MetricGetDescription
    descriptor.IsMaxOptimalFunc = &_MetricIsMaxOptimal
    descriptor.GetFinalErrorFunc = &_MetricGetFinalError
    return descriptor

cdef TCustomObjectiveDescriptor _BuildCustomObjectiveDescriptor(object objectiveObject):
    cdef TCustomObjectiveDescriptor descriptor
    descriptor.CustomData = <void*>objectiveObject
    descriptor.CalcDersRange = &_ObjectiveCalcDersRange
    descriptor.CalcDersMulti = &_ObjectiveCalcDersMulti
    return descriptor

cdef class PyPredictionType:
    cdef EPredictionType predictionType
    def __init__(self, prediction_type):
        if prediction_type == 'Class':
            self.predictionType = EPredictionType_Class
        elif prediction_type == 'Probability':
            self.predictionType = EPredictionType_Probability
        else:
            self.predictionType = EPredictionType_RawFormulaVal

cdef class PyExportType:
    cdef EModelExportType exportType
    def __init__(self, prediction_type):
        if prediction_type == 'coreml':
            self.exportType = EModelExportType_CoreML
        else:
            self.exportType = EModelExportType_Catboost

cdef to_binary_str(string):
    if PY3:
        return string.encode()
    return string

cdef to_native_str(binary):
    if PY3:
        return binary.decode()
    return binary

cdef class _PoolBase:
    cdef TPool* __pool
    cdef bool_t has_label

    def __cinit__(self):
        self.__pool = new TPool()
        self.has_label = False

    def __dealloc__(self):
        del self.__pool

    cpdef _init_cat_features(self, cat_features):
        self.__pool.CatFeatures.clear()
        for feature in cat_features:
            self.__pool.CatFeatures.push_back(int(feature))

    cpdef _read_pool(self, string pool_file, string cd_file, string delimiter, bool_t has_header, int thread_count):
        ReadPool(cd_file, pool_file, thread_count, False, self.__pool, ord(delimiter), has_header)
        if len(set([doc.Target for doc in self.__pool.Docs])) > 1:
            self.has_label = True

    cpdef _init_pool(self, data, label, weight, baseline, feature_names):
        self._set_data(data)
        num_class = 2
        if label is not None:
            self._set_label(label)
            num_class = len(set(label))
            self.has_label = True
        if baseline is not None:
            self._set_baseline(baseline)
        if weight is not None:
            self._set_weight(weight)
        if feature_names is not None:
            self._set_feature_names(feature_names)

    cpdef _set_data(self, data):
        self.__pool.Docs.clear()
        cdef TDocInfo doc
        cdef TString factor_str
        cat_features = set(self.get_cat_feature_indices())
        for i in range(len(data)):
            for j, factor in enumerate(data[i]):
                if j in cat_features:
                    if not isinstance(factor, string_types):
                        if int(factor) != factor:
                            raise CatboostError('Invalid type for cat_feature[{},{}]={} : cat_features must be integer or string, real number values should be converted to string.'.format(i, j, factor))
                        factor = str(int(factor))
                    factor = to_binary_str(factor)
                    factor_str = TString(<char*>factor)
                    factor = CalcCatFeatureHash(factor_str)
                    self.__pool.CatFeaturesHashToString[factor] = factor_str
                    factor = ConvertCatFeatureHashToFloat(factor)
                doc.Factors.push_back(float(factor))
            self.__pool.Docs.push_back(doc)
            doc.Factors.clear()

    cpdef _set_label(self, label):
        rows = self.num_row()
        for i in range(rows):
            self.__pool.Docs[i].Target = float(label[i])

    cpdef _set_weight(self, weight):
        rows = self.num_row()
        for i in range(rows):
            self.__pool.Docs[i].Weight = float(weight[i])

    cpdef _set_baseline(self, baseline):
        rows = self.num_row()
        cdef yvector[double] cbaseline
        for i in range(rows):
            for value in baseline[i]:
                cbaseline.push_back(float(value))
            self.__pool.Docs[i].Baseline = cbaseline
            cbaseline.clear()

    cpdef _set_feature_names(self, feature_names):
        self.__pool.FeatureId.clear()
        for value in feature_names:
            self.__pool.FeatureId.push_back(str(value))

    cpdef get_feature_names(self):
        if self.is_empty():
            return None
        feature_names = []
        cdef bytes pystr
        for value in self.__pool.FeatureId:
            pystr = value.c_str()
            feature_names.append(to_native_str(pystr))
        return feature_names

    cpdef num_row(self):
        """
        Get the number of rows in the Pool.

        Returns
        -------
        number of rows : int
        """
        if not self.is_empty():
            return self.__pool.Docs.size()
        return None

    cpdef num_col(self):
        """
        Get the number of columns in the Pool.

        Returns
        -------
        number of cols : int
        """
        if not self.is_empty():
            row = self.num_row()
            if row > 0:
                return self.__pool.Docs[0].Factors.size()
        return None

    @property
    def shape(self):
        """
        Get the shape of the Pool.

        Returns
        -------
        shape : (int, int)
            (rows, cols)
        """
        if not self.is_empty():
            return tuple([self.num_row(), self.num_col()])
        return None

    cpdef get_features(self):
        """
        Get feature matrix from Pool.

        Returns
        -------
        feature matrix : list(list)
        """
        if not self.is_empty():
            data = []
            for doc in self.__pool.Docs:
                factors = []
                for factor in doc.Factors:
                    factors.append(factor)
                data.append(factors)
            return data
        return None

    cpdef get_label(self):
        """
        Get labels from Pool.

        Returns
        -------
        labels : list
        """
        if self.has_label:
            return [doc.Target for doc in self.__pool.Docs]
        return None

    cpdef get_cat_feature_indices(self):
        """
        Get cat_feature indices from Pool.

        Returns
        -------
        cat_feature_indices : list
        """
        return [self.__pool.CatFeatures.at(i) for i in range(self.__pool.CatFeatures.size())]

    cpdef get_weight(self):
        """
        Get weight for each instance.

        Returns
        -------
        weight : list
        """
        if not self.is_empty():
            return [doc.Weight for doc in self.__pool.Docs]
        return None

    cpdef get_baseline(self):
        """
        Get baseline from Pool.

        Returns
        -------
        baseline : list(list)
        """
        if not self.is_empty():
            baseline = []
            for doc in self.__pool.Docs:
                doc_approxes = []
                for approx in doc.Baseline:
                    doc_approxes.append(approx)
                baseline.append(doc_approxes)
            return baseline
        return None

    cpdef is_empty(self):
        """
        Check if Pool is empty.

        Returns
        -------
        is_empty : bool
        """
        return self.__pool.Docs.empty()

cdef dict _PreprocessParams(dict params):
    eval_metric = params.get("eval_metric")
    objective = params.get("loss_function")

    is_custom_eval_metric = eval_metric is not None and not isinstance(eval_metric, string_types)
    is_custom_objective = objective is not None and not isinstance(objective, string_types)

    params_to_json = params

    if is_custom_objective or is_custom_eval_metric:
        keys_to_replace = set()
        if is_custom_objective:
            keys_to_replace.add("loss_function")
        if is_custom_eval_metric:
            keys_to_replace.add("eval_metric")

        params_to_json = {}

        for k, v in params.iteritems():
            if k in keys_to_replace:
                continue
            params_to_json[k] = deepcopy(v)

        for k in keys_to_replace:
            params_to_json[k] = "Custom"

    return params_to_json

cdef class _CatBoost:
    cdef TFullModel* __model
    cdef yvector[yvector[double]]* __test_eval

    def __cinit__(self):
        self.__model = new TFullModel()
        self.__test_eval = new yvector[yvector[double]]()

    def __dealloc__(self):
        del self.__model
        del self.__test_eval

    cpdef _train(self, _PoolBase train_pool, _PoolBase test_pool, dict params):
        params_to_json = _PreprocessParams(params)
        dumps_params = dumps(params_to_json)

        cdef TJsonValue tree
        cdef TMaybe[TCustomObjectiveDescriptor] customObjectiveDescriptor
        cdef TMaybe[TCustomMetricDescriptor] customMetricDescriptor

        if params_to_json.get("loss_function") == "Custom":
            customObjectiveDescriptor = _BuildCustomObjectiveDescriptor(params["loss_function"])
        if params_to_json.get("eval_metric") == "Custom":
            customMetricDescriptor = _BuildCustomMetricDescriptor(params["eval_metric"])

        dumps_params = to_binary_str(dumps_params)
        tree = ReadTJsonValue(TString(<const char*>dumps_params))
        with nogil:
            SetPythonInterruptHandler()
            try:
                TrainModel(tree,
                       customObjectiveDescriptor,
                       customMetricDescriptor,
                       dereference(train_pool.__pool),
                       dereference(test_pool.__pool),
                       TString(<const char*>""),
                       self.__model,
                       self.__test_eval)
            finally:
                ResetPythonInterruptHandler()

    cpdef set_test_eval(self, test_eval):
        cdef yvector[double] vector
        for row in test_eval:
            for value in row:
                vector.push_back(float(value))
            self.__test_eval.push_back(vector)
            vector.clear()

    cpdef get_test_eval(self):
        test_eval = []
        for i in range(self.__test_eval.size()):
            test_eval.append([value for value in dereference(self.__test_eval)[i]])
        return test_eval

    cpdef _get_cat_feature_indices(self):
        return [feature for feature in self.__model.CatFeatures]

    cpdef _base_predict(self, _PoolBase pool, str prediction_type, int ntree_limit, verbose):
        cdef yvector[double] pred
        cdef EPredictionType predictionType = PyPredictionType(prediction_type).predictionType
        pred = ApplyModel(dereference(self.__model),
                            dereference(pool.__pool),
                            verbose,
                            predictionType,
                            0,
                            ntree_limit, 1)
        return [value for value in pred]

    cpdef _base_predict_multi(self, _PoolBase pool, str prediction_type, int ntree_limit, verbose):
        cdef yvector[yvector[double]] pred
        cdef EPredictionType predictionType = PyPredictionType(prediction_type).predictionType
        pred = ApplyModelMulti(dereference(self.__model),
                                dereference(pool.__pool),
                                verbose,
                                predictionType,
                                0,
                                ntree_limit, 1)
        return [[value for value in vec] for vec in pred]

    cpdef _calc_fstr(self, _PoolBase pool, int thread_count):
        return [value for value in CalcRegularFeatureEffect(dereference(self.__model), dereference(pool.__pool), thread_count)]

    cpdef _load_model(self, string model_file):
        cdef TFullModel tmp_model
        tmp_model = ReadModel(model_file)
        self.__model.Swap(tmp_model)

    cpdef _save_model(self, output_file, format, export_parameters):
        cdef EModelExportType exportType = PyExportType(format).exportType
        ExportModel(dereference(self.__model), output_file, exportType, export_parameters)

    cpdef _serialize_model(self):
        cdef TString tstr = SerializeModel(dereference(self.__model))
        cdef const char* c_serialized_model_string = tstr.c_str()
        cpdef bytes py_serialized_model_str = c_serialized_model_string[:tstr.size()]
        return py_serialized_model_str

    cpdef _deserialize_model(self, TString serialized_model_str):
        cdef TFullModel tmp_model
        tmp_model = DeserializeModel(serialized_model_str);
        self.__model.Swap(tmp_model)

    cpdef _get_params(self):
        cdef const char* c_params_json = self.__model.ParamsJson.c_str()
        cdef bytes py_params_json = c_params_json
        params_json = to_native_str(py_params_json)
        params = {}
        if params_json:
            for key, value in loads(params_json).iteritems():
                if key not in params:
                    params[str(key)] = value
        return params

    def get_tree_count(self):
        return self.__model.TreeStruct.size()

class _CatBoostBase(object):
    def __init__(self, params):
        self._is_fitted = False
        self._init_params = params
        self.object = _CatBoost()

    def __getstate__(self):
        params = self.get_init_params()
        test_evals = self.get_test_eval()
        if test_evals:
            if test_evals[0]:
                params['_test_eval'] = test_evals
        if self._is_fitted:
            params['__model'] = self._serialize_model()
        return params

    def __setstate__(self, state):
        if 'object' not in dict(self.__dict__.items()):
            self.object = _CatBoost()
        if '_is_fitted' not in dict(self.__dict__.items()):
            self._is_fitted = False
        if '_init_params' not in dict(self.__dict__.items()):
            self._init_params = {}
        if '__model' in state:
            self._deserialize_model(state['__model'])
            self._is_fitted = True
            del state['__model']
        if '_test_eval' in state:
            self.set_test_eval(state['_test_eval'])
            del state['_test_eval']
        self._init_params.update(state)

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        state = self.__getstate__()
        model = self.__class__()
        model.__setstate__(state)
        return model

    def copy(self):
        return self.__copy__()

    def _train(self, train_pool, test_pool, params):
        self.object._train(train_pool, test_pool, params)
        self._is_fitted = True

    def set_test_eval(self, test_eval):
        self.object.set_test_eval(test_eval)

    def get_test_eval(self):
        return self.object.get_test_eval()

    def _get_cat_feature_indices(self):
        return self.object._get_cat_feature_indices()

    def _base_predict(self, pool, prediction_type, ntree_limit, verbose):
        return self.object._base_predict(pool, prediction_type, ntree_limit, verbose)

    def _base_predict_multi(self, pool, prediction_type, ntree_limit, verbose):
        return self.object._base_predict_multi(pool, prediction_type, ntree_limit, verbose)

    def _calc_fstr(self, pool, thread_count):
        return self.object._calc_fstr(pool, thread_count)

    def _save_model(self, output_file, format, export_parameters):
        if self._is_fitted:
            params_string = ""
            if export_parameters:
                params_string = dumps(export_parameters)

            self.object._save_model(output_file, format, params_string)

    def _load_model(self, model_file):
        self.object._load_model(model_file)
        self._is_fitted = True

    def _serialize_model(self):
        return self.object._serialize_model()

    def _deserialize_model(self, dump_model_str):
        self.object._deserialize_model(dump_model_str)

    def get_init_params(self):
        init_params = self._init_params.copy()
        if "kwargs" in init_params:
            init_params.update(init_params["kwargs"])
            del init_params["kwargs"]
        return init_params

    def _get_init_train_params(self):
        params = self._init_params.copy()
        if "kwargs" in params:
            del params["kwargs"]
        return params

    def _get_params(self):
        params = self.object._get_params()
        init_params = self.get_init_params()
        for key, value in iteritems(init_params):
            if key not in params:
                params[key] = value
        return params

    def get_tree_count(self):
        return self.object.get_tree_count()

    def _set_param(self, key, value):
        self._init_params[key] = value

    def is_fitted(self):
        return self._is_fitted

cpdef _cv(dict params, _PoolBase pool, int fold_count, bool_t inverted, int random_seed,
          bool_t shuffle, bool_t enable_early_stopping, int eval_period):
    params_to_json = _PreprocessParams(params)
    dumps_params = dumps(params_to_json)

    cdef TJsonValue tree
    cdef TCrossValidationParams cvParams
    cdef TMaybe[TCustomObjectiveDescriptor] customObjectiveDescriptor
    cdef TMaybe[TCustomMetricDescriptor] customMetricDescriptor
    cdef yvector[TCVResult] results

    if params_to_json.get("loss_function") == "Custom":
        customObjectiveDescriptor = _BuildCustomObjectiveDescriptor(params["loss_function"])
    if params_to_json.get("eval_metric") == "Custom":
        customMetricDescriptor = _BuildCustomMetricDescriptor(params["eval_metric"])

    dumps_params = to_binary_str(dumps_params)
    tree = ReadTJsonValue(TString(<const char*>dumps_params))
    cvParams.FoldCount = fold_count
    cvParams.RandSeed = random_seed
    cvParams.Shuffle = shuffle
    cvParams.Inverted = inverted
    cvParams.EnableEarlyStopping = enable_early_stopping
    cvParams.EvalPeriod = eval_period

    with nogil:
        SetPythonInterruptHandler()
        try:
            CrossValidate(
                tree,
                customObjectiveDescriptor,
                customMetricDescriptor,
                dereference(pool.__pool),
                cvParams,
                &results)
        finally:
            ResetPythonInterruptHandler()

    result = defaultdict(list)
    for metric_idx in xrange(results.size()):
        metric_name = str(results[metric_idx].Metric)
        for it in xrange(results[metric_idx].AverageTrain.size()):
            result[metric_name + "_train_avg"].append(results[metric_idx].AverageTrain[it])
            result[metric_name + "_train_stddev"].append(results[metric_idx].StdDevTrain[it])
            result[metric_name + "_test_avg"].append(results[metric_idx].AverageTest[it])
            result[metric_name + "_test_stddev"].append(results[metric_idx].StdDevTest[it])

    return result

log_out = None

cdef void _LogPrinter(const char* str, size_t len) except * with gil:
    cdef bytes bytes_str = str[:len]
    log_out.write(to_native_str(bytes_str))

cpdef _set_logger(out):
    global log_out
    log_out = out
    SetCustomLoggingFunction(&_LogPrinter)

cpdef _reset_logger():
    RestoreOriginalLogger()
