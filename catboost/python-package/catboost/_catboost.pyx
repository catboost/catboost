# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport *
from catboost.libs.model.cython cimport *
from catboost.libs.monoforest._monoforest cimport *

import atexit
import six
from six import iteritems, string_types
from cpython.version cimport PY_MAJOR_VERSION
import warnings

from six.moves import range
from json import dumps, loads, JSONEncoder
from copy import deepcopy
from collections import defaultdict
import functools
import inspect
import os
import traceback
import types

import sys
if sys.version_info >= (3, 3):
    from collections.abc import Iterable, Sequence
else:
    from collections import Iterable, Sequence
import platform

import numpy as np
cimport numpy as np  # noqa

import pandas as pd
import scipy.sparse

np.import_array()

cimport cython
from cpython cimport PyList_GET_ITEM, PyTuple_GET_ITEM, PyFloat_AsDouble
from cython.operator cimport dereference, preincrement

from libc.math cimport isnan, modf
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy
from libcpp cimport bool as bool_t
from libcpp cimport nullptr
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cpython.ref cimport PyObject

from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from util.generic.hash cimport THashMap
from util.generic.hash_set cimport THashSet
from util.generic.maybe cimport TMaybe
from util.generic.ptr cimport TAtomicSharedPtr, THolder, TIntrusivePtr, MakeHolder
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui8, ui16, ui32, ui64, i32, i64
from util.string.cast cimport StrToD, TryFromString, ToString


def fspath(path):
    if path is None:
        return None
    if sys.version_info >= (3, 6):
        return os.fspath(path)
    return str(path)


SPARSE_MATRIX_TYPES = (
    scipy.sparse.csr_matrix,
    scipy.sparse.coo_matrix,
    scipy.sparse.bsr_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.dok_matrix,
    scipy.sparse.lil_matrix,
)


ctypedef fused numpy_indices_dtype:
    np.int32_t
    np.int64_t
    np.uint32_t
    np.uint64_t


ctypedef fused numpy_num_dtype:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t


numpy_num_dtype_list = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64
]

custom_objective_methods_to_optimize = [
    'calc_ders_range',
    'calc_ders_multi',
]

custom_metric_methods_to_optimize = [
    'evaluate',
    'get_final_error'
]

from catboost.private.libs.cython cimport *
from catboost.libs.helpers.cython cimport *
from catboost.libs.data.cython cimport *
from catboost.private.libs.data_util.cython cimport TPathWithScheme

class _NumpyAwareEncoder(JSONEncoder):
    bool_types = (np.bool_)
    tolist_types = (np.ndarray,)
    def default(self, obj):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        if np.issubdtype(type(obj), np.floating):
            return float(obj)
        if isinstance(obj, self.bool_types):
            return bool(obj)
        if isinstance(obj, self.tolist_types):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class CatBoostError(Exception):
    pass


cdef public object PyCatboostExceptionType = <object>CatBoostError


@cython.embedsignature(True)
class MultiTargetCustomMetric:
    def evaluate(self, approxes, targets, weights):
        """
        Evaluates metric value.

        Parameters
        ----------
        approxes : list of lists of float
            Vectors of approx labels.

        targets : list of lists of float
            Vectors of true labels.

        weights : list of float, optional (default=None)
            Weight for each instance.

        Returns
        -------
            weighted error : float
            total weight : float

        """
        raise CatBoostError("evaluate method is not implemented")

    def is_max_optimal(self):
        raise CatBoostError("is_max_optimal method is not implemented")

    def get_final_error(self, error, weight):
        """
        Returns final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        metric value : float

        """
        raise CatBoostError("get_final_error method is not implemented")


@cython.embedsignature(True)
class MultiTargetCustomObjective:
    def calc_ders_multi(self, approxes, targets, weights):
        """
        Computes first derivative and Hessian matrix of the loss function with respect to the predicted value for each dimension.

        Parameters
        ----------
        approxes : list of float
            Vector of approx labels.

        targets : list of float
            Vector of true labels.

        weight : float, optional (default=None)
            Instance weight.

        Returns
        -------
            der1 : list of float
            der2 : list of lists of float

        """
        raise CatBoostError("calc_ders_multi method is not implemented")

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8AndSize(object s, Py_ssize_t* l)

cdef extern from "catboost/libs/logging/logging.h":
    ctypedef void(*TCustomLoggingFunctionPtr)(const char *, size_t len, void *) except * with gil
    cdef void SetCustomLoggingFunction(TCustomLoggingFunctionPtr, TCustomLoggingFunctionPtr, void*, void*)
    cdef void RestoreOriginalLogger()
    cdef void ResetTraceBackend(const TString&)


cdef extern from "catboost/libs/cat_feature/cat_feature.h":
    cdef ui32 CalcCatFeatureHash(TStringBuf feature) except +ProcessException
    cdef float ConvertCatFeatureHashToFloat(ui32 hashVal) except +ProcessException


cdef class Py_FloatSequencePtr:
    cdef ITypedSequencePtr[np.float32_t] result

    def __cinit__(self):
        pass

    cdef set_result(self, ITypedSequencePtr[np.float32_t] result):
        self.result = result

    cdef get_result(self, ITypedSequencePtr[np.float32_t]* result):
        result[0] = self.result

    def __dealloc__(self):
        pass

ctypedef TMaybeOwningConstArrayHolder[np.float32_t] TEmbeddingData

cdef class Py_EmbeddingSequencePtr:
    cdef ITypedSequencePtr[TEmbeddingData] result

    def __cinit__(self):
        pass

    cdef set_result(self, ITypedSequencePtr[TEmbeddingData] result):
        self.result = result

    cdef get_result(self, ITypedSequencePtr[TEmbeddingData]* result):
        result[0] = self.result

    def __dealloc__(self):
        pass


@cython.boundscheck(False)
@cython.wraparound(False)
def make_non_owning_type_cast_array_holder(np.ndarray[numpy_num_dtype, ndim=1] array):

    """
        older buffer interface is used instead of memory views because of
        https://github.com/cython/cython/issues/1772, https://github.com/cython/cython/issues/2485
    """

    cdef const numpy_num_dtype* array_begin = <const numpy_num_dtype*>nullptr
    cdef const numpy_num_dtype* array_end = <const numpy_num_dtype*>nullptr

    if array.shape[0] != 0:
        array_begin = &array[0]
        array_end = array_begin + array.shape[0]

    cdef Py_FloatSequencePtr py_result = Py_FloatSequencePtr()
    cdef ITypedSequencePtr[np.float32_t] result

    if numpy_num_dtype is np.int8_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.int8_t](array_begin, array_end)
    if numpy_num_dtype is np.int16_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.int16_t](array_begin, array_end)
    if numpy_num_dtype is np.int32_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.int32_t](array_begin, array_end)
    if numpy_num_dtype is np.int64_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.int64_t](array_begin, array_end)
    if numpy_num_dtype is np.uint8_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.uint8_t](array_begin, array_end)
    if numpy_num_dtype is np.uint16_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.uint16_t](array_begin, array_end)
    if numpy_num_dtype is np.uint32_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.uint32_t](array_begin, array_end)
    if numpy_num_dtype is np.uint64_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.uint64_t](array_begin, array_end)
    if numpy_num_dtype is np.float32_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.float32_t](array_begin, array_end)
    if numpy_num_dtype is np.float64_t:
        result = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.float64_t](array_begin, array_end)

    py_result.set_result(result)

    return py_result


# returns (Py_EmbeddingSequencePtr, new data holders array)
@cython.boundscheck(False)
@cython.wraparound(False)
def make_embedding_type_cast_array_holder(
    size_t flat_feature_idx,
    np.ndarray[numpy_num_dtype, ndim=1] first_element,
    np.ndarray elements): #

    cdef np.ndarray[numpy_num_dtype, ndim=1] element
    cdef TVector[TMaybeOwningConstArrayHolder[numpy_num_dtype]] data
    cdef size_t embedding_dimension = len(first_element)
    cdef size_t object_count = len(elements)
    cdef size_t object_idx

    data_holders = []

    for object_idx in range(object_count):
        element = elements[object_idx]
        if len(element) != embedding_dimension:
            raise CatBoostError(
                (
                    'Inсonsistent array size for embedding_feature[object_idx={},feature_idx={}]={}, should be '
                    + 'equal to array size for the first object ={}'
                ).format(
                    object_idx,
                    flat_feature_idx,
                    len(element),
                    embedding_dimension
                )
            )

        if not element.flags.c_contiguous:
            element = np.ascontiguousarray(element)
            data_holders.append(element)

        data.push_back(
            TMaybeOwningConstArrayHolder[numpy_num_dtype].CreateNonOwning(
                TConstArrayRef[numpy_num_dtype](
                    <numpy_num_dtype*>&element[0],
                    embedding_dimension
                )
            )
        )

    if len(data_holders) != len(elements):
        data_holders.append(elements)

    cdef ITypedSequencePtr[TMaybeOwningConstArrayHolder[np.float32_t]] result

    if numpy_num_dtype is np.int8_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.int8_t](data)
    if numpy_num_dtype is np.int16_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.int16_t](data)
    if numpy_num_dtype is np.int32_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.int32_t](data)
    if numpy_num_dtype is np.int64_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.int64_t](data)
    if numpy_num_dtype is np.uint8_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.uint8_t](data)
    if numpy_num_dtype is np.uint16_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.uint16_t](data)
    if numpy_num_dtype is np.uint32_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.uint32_t](data)
    if numpy_num_dtype is np.uint64_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.uint64_t](data)
    if numpy_num_dtype is np.float32_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.float32_t](data)
    if numpy_num_dtype is np.float64_t:
        result = MakeTypeCastArraysHolderFromVector[np.float32_t, np.float64_t](data)

    cdef Py_EmbeddingSequencePtr py_result = Py_EmbeddingSequencePtr()
    py_result.set_result(result)

    return py_result, data_holders


cdef extern from "catboost/private/libs/quantized_pool/serialization.h" namespace "NCB":
    cdef void SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName) except +ProcessException


cdef extern from "catboost/private/libs/data_util/line_data_reader.h" namespace "NCB":
    cdef cppclass TDsvFormatOptions:
        bool_t HasHeader
        char Delimiter
        bool_t IgnoreCsvQuoting

cdef extern from "catboost/private/libs/options/load_options.h" namespace "NCatboostOptions":
    cdef cppclass TColumnarPoolFormatParams:
        TDsvFormatOptions DsvFormat
        TPathWithScheme CdFilePath


cdef class Py_ObjectsOrderBuilderVisitor:
    cdef TDataProviderBuilderOptions options
    cdef TAtomicSharedPtr[TTbbLocalExecutor] local_executor
    cdef THolder[IDataProviderBuilder] data_provider_builder
    cdef IRawObjectsOrderDataVisitor* builder_visitor
    cdef const TFeaturesLayout* features_layout

    def __cinit__(self, int thread_count):
        self.local_executor = GetCachedLocalExecutor(thread_count)
        CreateDataProviderBuilderAndVisitor(
            self.options,
            <ILocalExecutor*>self.local_executor.Get(),
            &self.data_provider_builder,
            &self.builder_visitor
        )

    cdef get_raw_objects_order_data_visitor(self, IRawObjectsOrderDataVisitor** builder_visitor):
        builder_visitor[0] = self.builder_visitor

    cdef set_features_layout(self, const TFeaturesLayout* features_layout):
        self.features_layout = features_layout

    cdef get_features_layout(self, const TFeaturesLayout** features_layout):
        features_layout[0] = self.features_layout

    def __dealloc__(self):
        pass


cdef class Py_FeaturesOrderBuilderVisitor:
    cdef TDataProviderBuilderOptions options
    cdef TAtomicSharedPtr[TTbbLocalExecutor] local_executor
    cdef THolder[IDataProviderBuilder] data_provider_builder
    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    cdef const TFeaturesLayout* features_layout

    def __cinit__(self, int thread_count):
        self.local_executor = GetCachedLocalExecutor(thread_count)
        CreateDataProviderBuilderAndVisitor(
            self.options,
            <ILocalExecutor*>self.local_executor.Get(),
            &self.data_provider_builder,
            &self.builder_visitor
        )
        #self.features_layout = features_layout

    cdef get_raw_features_order_data_visitor(self, IRawFeaturesOrderDataVisitor** builder_visitor):
        builder_visitor[0] = self.builder_visitor

    cdef set_features_layout(self, const TFeaturesLayout* features_layout):
        self.features_layout = features_layout

    cdef get_features_layout(self, const TFeaturesLayout** features_layout):
        features_layout[0] = self.features_layout

    def __dealloc__(self):
        pass


cdef extern from "catboost/libs/data/load_data.h" namespace "NCB":
    cdef TDataProviderPtr ReadDataset(
        TMaybe[ETaskType] taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const TPathWithScheme& timestampsFilePath,
        const TPathWithScheme& baselineFilePath,
        const TPathWithScheme& featureNamesPath,
        const TPathWithScheme& poolMetaInfoPath,
        const TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector[ui32]& ignoredFeatures,
        EObjectsOrder objectsOrder,
        int threadCount,
        bool_t verbose,
        bool_t forceUnitAutoPAirweights
    ) nogil except +ProcessException


cdef extern from "catboost/libs/data/load_and_quantize_data.h" namespace "NCB":
    cdef TDataProviderPtr ReadAndQuantizeDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const TPathWithScheme& timestampsFilePath,
        const TPathWithScheme& baselineFilePath,
        const TPathWithScheme& featureNamesPath,
        const TPathWithScheme& poolMetaInfoPath,
        const TPathWithScheme& inputBordersPath,
        const TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector[ui32]& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TJsonValue plainJsonParams,
        TMaybe[ui32] blockSize,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        int threadCount,
        bool_t verbose
    ) nogil except +ProcessException


cdef extern from "catboost/libs/model/ctr_provider.h":
    cdef cppclass ECtrTableMergePolicy:
        pass


ctypedef const TFullModel* TFullModel_const_ptr

cdef extern from "catboost/libs/model/model.h":
    cdef TFullModel SumModels(TVector[TFullModel_const_ptr], const TVector[double]&, const TVector[TString]&, ECtrTableMergePolicy) nogil except +ProcessException

cdef extern from "catboost/libs/model/model_export/model_exporter.h" namespace "NCB":
    cdef void ExportModel(
        const TFullModel& model,
        const TString& modelFile,
        const EModelType format,
        const TString& userParametersJson,
        bool_t addFileFormatExtension,
        const TVector[TString]* featureId,
        const THashMap[ui32, TString]* catFeaturesHashToString
    ) nogil except +ProcessException

    cdef TString ConvertTreeToOnnxProto(
        const TFullModel& model,
        const TString& userParametersJson)

cdef extern from "catboost/libs/model/utils.h":
    cdef TJsonValue GetPlainJsonWithAllOptions(const TFullModel& model) nogil except +ProcessException

cdef extern from "library/cpp/containers/2d_array/2d_array.h":
    cdef cppclass TArray2D[T]:
        T* operator[] (size_t index) const


cdef extern from "util/system/info.h" namespace "NSystemInfo":
    cdef size_t CachedNumberOfCpus() except +ProcessException
    cdef size_t TotalMemorySize() except +ProcessException


cdef extern from "util/system/atexit.h":
    cdef void ManualRunAtExitFinalizers()


cdef extern from "catboost/libs/metrics/metric_holder.h":
    cdef cppclass TMetricHolder:
        TVector[double] Stats

        void Add(TMetricHolder& other) except +ProcessException

cdef extern from "catboost/libs/metrics/metric.h":
    cdef cppclass IMetric:
        TString GetDescription() except +ProcessException
        bool_t IsAdditiveMetric() except +ProcessException

cdef extern from "catboost/libs/metrics/metric.h":
    cdef bool_t IsMaxOptimal(const IMetric& metric) except +ProcessException
    cdef TJsonValue ExportAllMetricsParamsToJson() except +ProcessException

def AllMetricsParams():
    return loads(to_native_str(WriteTJsonValue(ExportAllMetricsParamsToJson())))

cdef extern from "catboost/private/libs/algo/tree_print.h":
    TVector[TString] GetTreeSplitsDescriptions(
        const TFullModel& model,
        size_t treeIdx,
        const TDataProviderPtr pool
    ) nogil except +ProcessException

    TVector[TString] GetTreeLeafValuesDescriptions(
        const TFullModel& model,
        size_t treeIdx
    ) nogil except +ProcessException

    TConstArrayRef[TNonSymmetricTreeStepNode] GetTreeStepNodes(
        const TFullModel& model,
        size_t treeIdx
    ) nogil except +ProcessException

    TVector[ui32] GetTreeNodeToLeaf(
        const TFullModel& model,
        size_t treeIdx
    ) nogil except +ProcessException


cdef extern from "catboost/libs/metrics/metric.h":
    cdef cppclass TCustomMetricDescriptor:
        void* CustomData

        ctypedef TMetricHolder (*TEvalFuncPtr)(
            TConstArrayRef[TConstArrayRef[double]]& approx,
            TConstArrayRef[float] target,
            TConstArrayRef[float] weight,
            int begin, int end, void* customData) with gil

        ctypedef TMetricHolder (*TEvalMultiTargetFuncPtr)(
            TConstArrayRef[TConstArrayRef[double]] approx,
            TConstArrayRef[TConstArrayRef[float]] target,
            TConstArrayRef[float] weight,
            int begin, int end, void* customData) with gil

        TMaybe[TEvalFuncPtr] EvalFunc
        TMaybe[TEvalMultiTargetFuncPtr] EvalMultiTargetFunc

        TString (*GetDescriptionFunc)(void *customData) except * with gil
        bool_t (*IsMaxOptimalFunc)(void *customData) except * with gil
        double (*GetFinalErrorFunc)(const TMetricHolder& error, void *customData) except * with gil
    cdef bool_t IsMaxOptimal(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsMinOptimal(const TString& metricName) nogil except +ProcessException


cdef extern from "catboost/private/libs/algo_helpers/custom_objective_descriptor.h":
    cdef cppclass TCustomObjectiveDescriptor:
        void* CustomData

        void (*CalcDersRange)(
            int count,
            const double* approxes,
            const float* targets,
            const float* weights,
            TDers* ders,
            void* customData
        ) with gil

        void (*CalcDersMultiClass)(
            const TVector[double]& approx,
            float target,
            float weight,
            TVector[double]* ders,
            THessianInfo* der2,
            void* customData
        ) with gil

        void (*CalcDersMultiTarget)(
            TConstArrayRef[double] approx,
            TConstArrayRef[float] target,
            float weight,
            TVector[double]* ders,
            THessianInfo* der2,
            void* customData
        ) with gil

ctypedef pair[TVector[TVector[ui32]], TVector[TVector[ui32]]] TCustomTrainTestSubsets
cdef extern from "catboost/private/libs/options/cross_validation_params.h":
    cdef cppclass TCrossValidationParams:
        ui32 FoldCount
        ECrossValidation Type
        int PartitionRandSeed
        bool_t Shuffle
        bool_t Stratified
        TMaybe[TVector[TVector[ui32]]] customTrainSubsets
        TMaybe[TVector[TVector[ui32]]] customTestSubsets
        double MetricUpdateInterval
        ui32 DevMaxIterationsBatchSize
        bool_t IsCalledFromSearchHyperparameters
        bool_t ReturnModels

cdef extern from "catboost/private/libs/options/split_params.h":
    cdef cppclass TTrainTestSplitParams:
        int PartitionRandSeed
        bool_t Shuffle
        bool_t Stratified
        double TrainPart

cdef extern from "catboost/private/libs/options/check_train_options.h":
    cdef void CheckFitParams(
        const TJsonValue& tree,
        const TCustomObjectiveDescriptor* objectiveDescriptor,
        const TCustomMetricDescriptor* evalMetricDescriptor
    ) nogil except +ProcessException

cdef extern from "catboost/private/libs/options/json_helper.h":
    cdef TJsonValue ReadTJsonValue(const TString& paramsJson) nogil except +ProcessException

cdef extern from "catboost/libs/loggers/catboost_logger_helpers.h":
    cdef cppclass TMetricsAndTimeLeftHistory:
        TVector[THashMap[TString, double]] LearnMetricsHistory
        TVector[TVector[THashMap[TString, double]]] TestMetricsHistory
        TMaybe[size_t] BestIteration
        THashMap[TString, double] LearnBestError
        TVector[THashMap[TString, double]] TestBestError

cdef extern from "catboost/libs/train_lib/train_model.h":
    cdef void TrainModel(
        TJsonValue params,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        const TMaybe[TCustomCallbackDescriptor]& callbackDescriptor,
        TDataProviders pools,
        TMaybe[TFullModel*] initModel,
        THolder[TLearnProgress]* initLearnProgress,
        const TString& outputModelPath,
        TFullModel* dstModel,
        const TVector[TEvalResult*]& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        THolder[TLearnProgress]* dstLearnProgress
    ) nogil except +ProcessException

    cdef cppclass TCustomCallbackDescriptor:
        void* CustomData

        bool_t (*AfterIterationFunc)(
            const TMetricsAndTimeLeftHistory& history,
            void *customData
        ) except * with gil

cdef extern from "catboost/libs/data/quantization.h"  namespace "NCB":
    cdef TQuantizedObjectsDataProviderPtr ConstructQuantizedPoolFromRawPool(
        TDataProviderPtr pool,
        TJsonValue plainJsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
    ) nogil except +ProcessException

cdef extern from "catboost/libs/train_lib/cross_validation.h":
    cdef cppclass TCVResult:
        TString Metric
        TVector[ui32] Iterations
        TVector[double] AverageTrain
        TVector[double] StdDevTrain
        TVector[double] AverageTest
        TVector[double] StdDevTest
        TVector[TFullModel] CVFullModels

    cdef void CrossValidate(
        TJsonValue jsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        TDataProviderPtr data,
        const TCrossValidationParams& cvParams,
        TVector[TCVResult]* results
    ) nogil except +ProcessException

cdef extern from "catboost/private/libs/algo/apply.h":
    cdef cppclass TModelCalcerOnPool:
        TModelCalcerOnPool(
            const TFullModel& model,
            TIntrusivePtr[TObjectsDataProvider] objectsData,
            ILocalExecutor* executor
        ) nogil except +ProcessException
        void ApplyModelMulti(
            const EPredictionType predictionType,
            int begin,
            int end,
            TVector[double]* flatApprox,
            TVector[TVector[double]]* approx
        ) nogil except +ProcessException

    cdef cppclass TLeafIndexCalcerOnPool:
        TLeafIndexCalcerOnPool(
            const TFullModel& model,
            TIntrusivePtr[TObjectsDataProvider] objectsData,
            int treeStart,
            int treeEnd
        ) nogil except +ProcessException

        bool_t Next() nogil except +ProcessException;
        bool_t CanGet() nogil except +ProcessException;
        TVector[ui32] Get() nogil except +ProcessException;

    cdef TVector[TVector[double]] ApplyModelMulti(
        const TFullModel& calcer,
        const TDataProvider& objectsData,
        bool_t verbose,
        const EPredictionType predictionType,
        int begin,
        int end,
        int threadCount
    ) nogil except +ProcessException

    cdef TVector[TVector[double]] ApplyUncertaintyPredictions(
        const TFullModel& calcer,
        const TDataProvider& objectsData,
        bool_t verbose,
        const EPredictionType predictionType,
        int end,
        int virtualEnsemblesCount,
        int threadCount
    ) nogil except +ProcessException

    cdef TVector[ui32] CalcLeafIndexesMulti(
        const TFullModel& model,
        TIntrusivePtr[TObjectsDataProvider] objectsData,
        bool_t verbose,
        int treeStart,
        int treeEnd,
        int threadCount
    )  nogil except +ProcessException

cdef extern from "catboost/private/libs/algo/helpers.h":
    cdef void ConfigureMalloc() nogil except *

cdef extern from "catboost/private/libs/algo/confusion_matrix.h":
    cdef TVector[double] MakeConfusionMatrix(
        const TFullModel &model,
        const TDataProviderPtr datasets,
        int threadCount
    ) nogil except +ProcessException

cdef extern from "catboost/private/libs/algo/roc_curve.h":
    cdef cppclass TRocPoint:
        double Boundary
        double FalseNegativeRate
        double FalsePositiveRate

        TRocPoint() nogil

        TRocPoint(double boundary, double FalseNegativeRate, double FalsePositiveRate) nogil

    cdef cppclass TRocCurve:
        TRocCurve() nogil

        TRocCurve(
            const TFullModel& model,
            const TVector[TDataProviderPtr]& datasets,
            int threadCount
        ) nogil  except +ProcessException

        TRocCurve(const TVector[TRocPoint]& points) nogil except +ProcessException

        double SelectDecisionBoundaryByFalsePositiveRate(
            double falsePositiveRate
        ) nogil except +ProcessException

        double SelectDecisionBoundaryByFalseNegativeRate(
            double falseNegativeRate
        ) nogil except +ProcessException

        double SelectDecisionBoundaryByIntersection() nogil except +ProcessException

        TVector[TRocPoint] GetCurvePoints() nogil except +ProcessException

        void Output(const TString& outputPath) except +ProcessException

cdef extern from "catboost/libs/eval_result/eval_helpers.h" namespace "NCB":
    cdef TVector[TVector[double]] PrepareEvalForInternalApprox(
        const EPredictionType predictionType,
        const TFullModel& model,
        const TVector[TVector[double]]& approx,
        int threadCount
    ) nogil except +ProcessException

cdef extern from "catboost/libs/eval_result/eval_result.h" namespace "NCB":
    cdef cppclass TEvalResult:
        TVector[TVector[TVector[double]]] GetRawValuesRef() except * with gil
        void ClearRawValues() except * with gil

cdef extern from "catboost/private/libs/init/init_reg.h" namespace "NCB":
    cdef void LibraryInit() nogil except *

cdef extern from "catboost/libs/fstr/partial_dependence.h":
    cdef TVector[double] GetPartialDependence(
        const TFullModel& model,
        TVector[int] features,
        const TDataProviderPtr dataset,
        int threadCount
    ) nogil except +ProcessException

cdef extern from "catboost/libs/fstr/calc_fstr.h":
    cdef TVector[TVector[double]] GetFeatureImportances(
        const EFstrType type,
        const TFullModel& model,
        const TDataProviderPtr dataset,
        const TDataProviderPtr referenceDataset,
        int threadCount,
        EPreCalcShapValues mode,
        int logPeriod,
        ECalcTypeShapValues calcType,
        EExplainableModelOutput modelOutputType,
        size_t sageNSamples,
        size_t sageBatchSize,
        bool_t sageDetectConvergence
    ) nogil except +ProcessException

    cdef TVector[TVector[TVector[double]]] GetFeatureImportancesMulti(
        const EFstrType type,
        const TFullModel& model,
        const TDataProviderPtr dataset,
        const TDataProviderPtr referenceDataset,
        int threadCount,
        EPreCalcShapValues mode,
        int logPeriod,
        ECalcTypeShapValues calcType,
        EExplainableModelOutput modelOutputType
    ) nogil except +ProcessException

    cdef TVector[TVector[TVector[TVector[double]]]] CalcShapFeatureInteractionMulti(
        const EFstrType type,
        const TFullModel& model,
        const TDataProviderPtr dataset,
        const TMaybe[pair[int, int]]& pairOfFeatures,
        int threadCount,
        EPreCalcShapValues mode,
        int logPeriod,
        ECalcTypeShapValues calcType
    ) nogil except +ProcessException

    TVector[TString] GetMaybeGeneratedModelFeatureIds(
        const TFullModel& model,
        const TDataProviderPtr dataset
    ) nogil except +ProcessException


cdef extern from "catboost/private/libs/documents_importance/docs_importance.h":
    cdef cppclass TDStrResult:
        TVector[TVector[uint32_t]] Indices
        TVector[TVector[double]] Scores
    cdef TDStrResult GetDocumentImportances(
        const TFullModel& model,
        const TDataProvider& trainData,
        const TDataProvider& testData,
        const TString& dstrType,
        int topSize,
        const TString& updateMethod,
        const TString& importanceValuesSign,
        int threadCount,
        int logPeriod
    ) nogil except +ProcessException


cdef float _FLOAT_NAN = float('nan')

cdef extern from "catboost/libs/data/borders_io.h" namespace "NCB" nogil:
    void LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
        const TString& path,
        TQuantizedFeaturesInfo* quantizedFeaturesInfo) except +ProcessException
    void SaveBordersAndNanModesToFileInMatrixnetFormat(
        const TString& path,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo) except +ProcessException

cdef extern from "catboost/libs/data/loader.h" namespace "NCB" nogil:
    int IsMissingValue(const TStringBuf& s)

cdef inline float _FloatOrNanFromString(const TString& s) except *:
    cdef char* stop = NULL
    cdef double parsed = StrToD(s.data(), &stop)
    cdef float res
    if IsMissingValue(<TStringBuf>s):
        res = _FLOAT_NAN
    elif stop == s.data() + s.size():
        res = parsed
    else:
        raise TypeError("Cannot convert '{}' to float".format(str(s)))
    return res

cdef extern from "catboost/libs/gpu_config/interface/get_gpu_device_count.h" namespace "NCB":
    cdef int GetGpuDeviceCount() except +ProcessException


cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef void SetPythonInterruptHandler() nogil
    cdef void ResetPythonInterruptHandler() nogil
    cdef void ThrowCppExceptionWithMessage(const TString&) nogil
    cdef void SetDataFromScipyCsrSparse[TFloatOrUi64](
        TConstArrayRef[ui32] rowMarkup,
        TConstArrayRef[TFloatOrUi64] values,
        TConstArrayRef[ui32] indices,
        bool_t hasSeparateEmbeddingFeaturesData,
        TConstArrayRef[ui32] mainDataFeatureIdxToDstFeatureIdx,
        TConstArrayRef[bool_t] catFeaturesMask,
        IRawObjectsOrderDataVisitor* builderVisitor,
        ILocalExecutor* localExecutor) nogil except +ProcessException
    cdef size_t GetNumPairs(const TDataProvider& dataProvider) except +ProcessException
    cdef TConstArrayRef[TPair] GetUngroupedPairs(const TDataProvider& dataProvider) except +ProcessException
    cdef void TrainEvalSplit(
        const TDataProvider& srcDataProvider,
        TDataProviderPtr* trainDataProvider,
        TDataProviderPtr* evalDataProvider,
        const TTrainTestSplitParams& splitParams,
        bool_t saveEvalDataset,
        int threadCount,
        ui64 cpuUsedRamLimit
    ) except +ProcessException
    cdef TAtomicSharedPtr[TTbbLocalExecutor] GetCachedLocalExecutor(int threadsCount)


cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef TVector[TVector[double]] EvalMetrics(
        const TFullModel& model,
        const TDataProvider& srcData,
        const TVector[TString]& metricsDescription,
        int begin,
        int end,
        int evalPeriod,
        int threadCount,
        const TString& resultDir,
        const TString& tmpDir
    ) nogil except +ProcessException

    cdef TVector[TString] GetMetricNames(
        const TFullModel& model,
        const TVector[TString]& metricsDescription
    ) nogil except +ProcessException

    cdef TVector[double] EvalMetricsForUtils(
        TConstArrayRef[TVector[float]] label,
        const TVector[TVector[double]]& approx,
        const TString& metricName,
        const TVector[float]& weight,
        const TVector[TGroupId]& groupId,
        const TVector[float]& groupWeight,
        const TVector[TSubgroupId]& subgroup_id,
        const TVector[TPair]& pairs,
        int threadCount
    ) nogil except +ProcessException

    cdef cppclass TMetricsPlotCalcerPythonWrapper:
        TMetricsPlotCalcerPythonWrapper(TVector[TString]& metrics, TFullModel& model, int ntree_start, int ntree_end,
                                        int eval_period, int thread_count, TString& tmpDir,
                                        bool_t flag) except +ProcessException
        TVector[const IMetric*] GetMetricRawPtrs() const
        TVector[TVector[double]] ComputeScores() except +ProcessException
        void AddPool(const TDataProvider& srcData) except +ProcessException

    cdef TJsonValue GetTrainingOptions(
        const TJsonValue& plainOptions,
        const TDataMetaInfo& trainDataMetaInfo,
        const TMaybe[TDataMetaInfo]& trainDataMetaInfo
    ) nogil except +ProcessException

    cdef cppclass TPythonStreamWrapper(IInputStream):
        ctypedef size_t (*TReadCallback)(char* target, size_t len, PyObject* stream, TString*)
        TPythonStreamWrapper(TReadCallback readCallback, PyObject* stream) except +ProcessException

cdef extern from "catboost/private/libs/quantized_pool_analysis/quantized_pool_analysis.h" namespace "NCB":
    cdef cppclass TBinarizedFeatureStatistics:
        TVector[float] Borders
        TVector[int] BinarizedFeature
        TVector[float] MeanTarget
        TVector[float] MeanWeightedTarget
        TVector[float] MeanPrediction
        TVector[size_t] ObjectsPerBin
        TVector[double] PredictionsOnVaryingFeature

    cdef cppclass TFeatureTypeAndInternalIndex:
        EFeatureType Type
        int Index

    cdef TVector[TBinarizedFeatureStatistics] GetBinarizedStatistics(
        const TFullModel& model,
        TDataProvider& dataset,
        const TVector[size_t]& catFeaturesNums,
        const TVector[size_t]& floatFeaturesNums,
        const EPredictionType predictionType,
        const int threadCount) nogil except +ProcessException

    cdef ui32 GetCatFeaturePerfectHash(
        const TFullModel& model,
        const TStringBuf& value,
        const size_t featureNum) nogil except +ProcessException

    cdef TFeatureTypeAndInternalIndex GetFeatureTypeAndInternalIndex(
        const TFullModel& model,
        const int flatFeatureIndex) nogil except +ProcessException

    cdef TVector[TString] GetCatFeatureValues(
        const TDataProvider& dataset,
        const int flatFeatureIndex) nogil except +ProcessException

cdef extern from "catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h" namespace "NCB":
    cdef cppclass TCustomRandomDistributionGenerator:
        void* CustomData
        double (*EvalFunc)(void* customData) with gil

    cdef cppclass TBestOptionValuesWithCvResult:
        TVector[TCVResult] CvResult
        THashMap[TString, bool_t] BoolOptions
        THashMap[TString, int] IntOptions
        THashMap[TString, ui32] UIntOptions
        THashMap[TString, double] DoubleOptions
        THashMap[TString, TString] StringOptions
        THashMap[TString, TVector[double]] ListOfDoublesOptions

    cdef void GridSearch(
        const TJsonValue& grid,
        const TJsonValue& params,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TCrossValidationParams& cvParams,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        TDataProviderPtr pool,
        TBestOptionValuesWithCvResult* results,
        TMetricsAndTimeLeftHistory* trainTestResult,
        bool_t isSearchUsingCV,
        bool_t isReturnCvResults,
        int verbose) nogil except +ProcessException

    cdef void RandomizedSearch(
        ui32 numberOfTries,
        const THashMap[TString, TCustomRandomDistributionGenerator]& randDistGenerators,
        const TJsonValue& grid,
        const TJsonValue& params,
        const TTrainTestSplitParams& trainTestSplitParams,
        const TCrossValidationParams& cvParams,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        TDataProviderPtr pool,
        TBestOptionValuesWithCvResult* results,
        TMetricsAndTimeLeftHistory* trainTestResult,
        bool_t isSearchUsingCV,
        bool_t isReturnCvResults,
        int verbose) nogil except +ProcessException


cdef extern from "catboost/libs/features_selection/select_features.h" namespace "NCB":
    cdef TJsonValue SelectFeatures(
        const TJsonValue& params,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        const TDataProviders& pools,
        TFullModel* dstModel,
        const TVector[TEvalResult*]& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory
    ) nogil except +ProcessException


cpdef run_atexit_finalizers():
    ManualRunAtExitFinalizers()


if not getattr(sys, "is_standalone_binary", False) and platform.system() == 'Windows':
    atexit.register(run_atexit_finalizers)


cdef inline float _FloatOrNan(object obj) except *:
    # here lies fastpath
    cdef type obj_type = type(obj)
    if obj is None:
        return _FLOAT_NAN
    elif obj_type is float:
        return <float>obj
    elif obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npbytes_ or obj_type is _npunicode_ or isinstance(obj, string_types + (_npbytes_, _npunicode_)):
        return _FloatOrNanFromString(to_arcadia_string(obj))
    try:
        return float(obj)
    except:
        raise TypeError("Cannot convert obj {} to float".format(str(obj)))


cpdef _float_or_nan(obj):
    return _FloatOrNan(obj)


cdef TString _MetricGetDescription(void* customData) except * with gil:
    cdef metricObject = <object>customData
    name = metricObject.__class__.__name__
    if PY_MAJOR_VERSION >= 3:
        name = name.encode()
    return TString(<const char*>name)

cdef bool_t _MetricIsMaxOptimal(void* customData) except * with gil:
    cdef metricObject = <object>customData
    return metricObject.is_max_optimal()

cdef double _MetricGetFinalError(const TMetricHolder& error, void *customData) except * with gil:
    # TODO(nikitxskv): use error.Stats for custom metrics.
    cdef metricObject = <object>customData
    return metricObject.get_final_error(error.Stats[0], error.Stats[1])

cdef bool_t _CallbackAfterIteration(
        const TMetricsAndTimeLeftHistory& history,
        void* customData
    ) except * with gil:
    cdef callbackObject = <object>customData
    if PY_MAJOR_VERSION >= 3:
        info = types.SimpleNamespace()
    else:
        from argparse import Namespace
        info = Namespace()
    info.iteration = history.LearnMetricsHistory.size()
    info.metrics = _get_metrics_evals_pydict(history)
    return callbackObject.after_iteration(info)

cdef _constarrayref_of_double_to_np_array(const TConstArrayRef[double] arr):
    result = np.empty(arr.size(), dtype=_npfloat64)
    for i in xrange(arr.size()):
        result[i] = arr[i]
    return result


cdef _vector_of_double_to_np_array(const TVector[double]& vec):
    result = np.empty(vec.size(), dtype=_npfloat64)
    for i in xrange(vec.size()):
        result[i] = vec[i]
    return result


cdef _2d_vector_of_double_to_np_array(const TVector[TVector[double]]& vectors):
    cdef size_t subvec_size = vectors[0].size() if not vectors.empty() else 0
    result = np.empty([vectors.size(), subvec_size], dtype=_npfloat64)
    for i in xrange(vectors.size()):
        assert vectors[i].size() == subvec_size, "All subvectors should have the same length"
        for j in xrange(subvec_size):
            result[i][j] = vectors[i][j]
    return result


cdef _3d_vector_of_double_to_np_array(const TVector[TVector[TVector[double]]]& vectors):
    cdef size_t subvec_size = vectors[0].size() if not vectors.empty() else 0
    cdef size_t sub_subvec_size = vectors[0][0].size() if subvec_size != 0 else 0
    result = np.empty([vectors.size(), subvec_size, sub_subvec_size], dtype=_npfloat64)
    for i in xrange(vectors.size()):
        assert vectors[i].size() == subvec_size, "All subvectors should have the same length"
        for j in xrange(subvec_size):
            assert vectors[i][j].size() == sub_subvec_size, "All subvectors should have the same length"
            for k in xrange(sub_subvec_size):
                result[i][j][k] = vectors[i][j][k]
    return result

cdef _reorder_axes_for_python_3d_shap_values(TVector[TVector[TVector[TVector[double]]]]& vectors):
    cdef size_t featuresCount = vectors.size() if not vectors.empty() else 0
    assert featuresCount == vectors[0].size()
    cdef size_t approx_dimension = vectors[0][0].size() if featuresCount != 0 else 0
    assert approx_dimension == 1
    cdef size_t doc_size = vectors[0][0][0].size() if approx_dimension != 0 else 0
    result = np.empty([doc_size, featuresCount, featuresCount], dtype=_npfloat64)
    cdef size_t doc, feature1, feature2
    for doc in xrange(doc_size):
        for feature1 in xrange(featuresCount):
            for feature2 in xrange(featuresCount):
                result[doc][feature1][feature2] = vectors[feature1][feature2][0][doc]
    return result

cdef _reorder_axes_for_python_4d_shap_values(TVector[TVector[TVector[TVector[double]]]]& vectors):
    cdef size_t featuresCount = vectors.size() if not vectors.empty() else 0
    assert featuresCount == vectors[0].size()
    cdef size_t approx_dimension = vectors[0][0].size() if featuresCount != 0 else 0
    assert approx_dimension > 1
    cdef size_t doc_size = vectors[0][0][0].size() if approx_dimension != 0 else 0
    result = np.empty([doc_size, approx_dimension, featuresCount, featuresCount], dtype=_npfloat64)
    cdef size_t doc, dim, feature1, feature2
    for doc in xrange(doc_size):
        for dim in xrange(approx_dimension):
            for feature1 in xrange(featuresCount):
                for feature2 in xrange(featuresCount):
                    result[doc][dim][feature1][feature2] = vectors[feature1][feature2][dim][doc]
    return result


cdef _vector_of_uints_to_np_array(const TVector[ui32]& vec):
    result = np.empty(vec.size(), dtype=np.uint32)
    for i in xrange(vec.size()):
        result[i] = vec[i]
    return result


cdef _vector_of_ints_to_np_array(const TVector[int]& vec):
    result = np.empty(vec.size(), dtype=np.int)
    for i in xrange(vec.size()):
        result[i] = vec[i]
    return result


cdef _vector_of_uints_to_2d_np_array(const TVector[ui32]& vec, int row_count, int column_count):
    assert vec.size() == row_count * column_count
    result = np.empty((row_count, column_count), dtype=np.uint32)
    for row_num in xrange(row_count):
        for col_num in xrange(column_count):
            result[row_num][col_num] = vec[row_num * column_count + col_num]
    return result


cdef _vector_of_floats_to_np_array(const TVector[float]& vec):
    result = np.empty(vec.size(), dtype=_npfloat32)
    for i in xrange(vec.size()):
        result[i] = vec[i]
    return result


cdef _vector_of_size_t_to_np_array(const TVector[size_t]& vec):
    result = np.empty(vec.size(), dtype=np.uint32)
    for i in xrange(vec.size()):
        result[i] = vec[i]
    return result


cdef np.ndarray _CreateNumpyFloatArrayView(const float* array, int count):
    cdef np.npy_intp dims[1]
    dims[0] = count
    return np.PyArray_SimpleNewFromData(1, dims, np.NPY_FLOAT, <void*>array)


cdef np.ndarray _CreateNumpyDoubleArrayView(const double* array, int count):
    cdef np.npy_intp dims[1]
    dims[0] = count
    return np.PyArray_SimpleNewFromData(1, dims, np.NPY_DOUBLE, <void*>array)


cdef np.ndarray _CreateNumpyUI64ArrayView(const ui64* array, int count):
    cdef np.npy_intp dims[1]
    dims[0] = count
    return np.PyArray_SimpleNewFromData(1, dims, np.NPY_UINT64, <void*>array)


cdef _ToPythonObjArrayOfArraysOfDoubles(const TConstArrayRef[double]* values, int size, int begin, int end):
    # https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    # numba doesn't like python lists, so using tuple instead
    return tuple(_CreateNumpyDoubleArrayView(values[i].data() + begin, end - begin) for i in xrange(size))

cdef _ToPythonObjArrayOfArraysOfFloats(const TConstArrayRef[float]* values, int size, int begin, int end):
    # using tuple as in _ToPythonObjArrayOfArraysOfDoubles
    return tuple(_CreateNumpyFloatArrayView(values[i].data() + begin, end - begin) for i in xrange(size))

cdef TMetricHolder _MetricEval(
    TConstArrayRef[TConstArrayRef[double]]& approx,
    TConstArrayRef[float] target,
    TConstArrayRef[float] weight,
    int begin,
    int end,
    void* customData
) with gil:
    cdef metricObject = <object>customData
    cdef TString errorMessage
    cdef TMetricHolder holder
    holder.Stats.resize(2)

    approxes = _ToPythonObjArrayOfArraysOfDoubles(approx.data(), approx.size(), begin, end)
    targets = _CreateNumpyFloatArrayView(target.data() + begin, end - begin)

    if weight.size() == 0:
        weights = None
    else:
        weights = _CreateNumpyFloatArrayView(weight.data() + begin, end - begin)

    try:
        error, weight_ = metricObject.evaluate(approxes, targets, weights)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    holder.Stats[0] = error
    holder.Stats[1] = weight_
    return holder

cdef TMetricHolder _MultiTargetMetricEval(
    TConstArrayRef[TConstArrayRef[double]] approx,
    TConstArrayRef[TConstArrayRef[float]] target,
    TConstArrayRef[float] weight,
    int begin,
    int end,
    void* customData
) with gil:
    cdef metricObject = <object>customData
    cdef TString errorMessage
    cdef TMetricHolder holder
    holder.Stats.resize(2)

    approxes = _ToPythonObjArrayOfArraysOfDoubles(approx.data(), approx.size(), begin, end)
    targets = _ToPythonObjArrayOfArraysOfFloats(target.data(), target.size(), begin, end)

    if weight.size() == 0:
        weights = None
    else:
        weights = _CreateNumpyFloatArrayView(weight.data() + begin, end - begin)

    try:
        error, weight_ = metricObject.evaluate(approxes, targets, weights)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    holder.Stats[0] = error
    holder.Stats[1] = weight_
    return holder

cdef double _RandomDistGen(
    void* customFunction
) with gil:
    cdef randomDistGenerator = <object>customFunction
    cdef TString errorMessage
    try:
        rand_value = randomDistGenerator.rvs()
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)
    return rand_value

cdef void _ObjectiveCalcDersRange(
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,
    TDers* ders,
    void* customData
) with gil:
    cdef objectiveObject = <object>(customData)
    cdef TString errorMessage
    cdef Py_ssize_t index
    cdef np.float32_t[:,:] pairs_np_float
    cdef np.float64_t[:,:] pairs_np_double

    approx = _CreateNumpyDoubleArrayView(approxes, count)
    target = _CreateNumpyFloatArrayView(targets, count)

    if weights:
        weight = _CreateNumpyFloatArrayView(weights, count)
    else:
        weight = None

    try:
        result = objectiveObject.calc_ders_range(approx, target, weight)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    if len(result) == 0:
        return

    if (type(result) == np.ndarray and len(result.shape) == 2 and result.shape[1] == 2 and
          result.dtype in [np.float32, np.float64]):
        if result.dtype == np.float32:
            pairs_np_float = result
            for index in range(len(pairs_np_float)):
                ders[index].Der1 = pairs_np_float[index, 0]
                ders[index].Der2 = pairs_np_float[index, 1]
        elif result.dtype == np.float64:
            pairs_np_double = result
            for index in range(len(pairs_np_double)):
                ders[index].Der1 = pairs_np_double[index, 0]
                ders[index].Der2 = pairs_np_double[index, 1]
    else:
        index = 0
        for der1, der2 in result:
            ders[index].Der1 = <double>der1
            ders[index].Der2 = <double>der2
            index += 1

cdef void _ObjectiveCalcDersMultiClass(
    const TVector[double]& approx,
    float target,
    float weight,
    TVector[double]* ders,
    THessianInfo* der2,
    void* customData
) with gil:
    cdef objectiveObject = <object>(customData)
    cdef TString errorMessage

    approxes = _CreateNumpyDoubleArrayView(approx.data(), approx.size())

    try:
        ders_vector, second_ders_matrix = objectiveObject.calc_ders_multi(approxes, target, weight)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    for index, der in enumerate(ders_vector):
        dereference(ders)[index] = der

    if der2:
        index = 0
        for indY, line in enumerate(second_ders_matrix):
            for num in line[indY:]:
                dereference(der2).Data[index] = num
                index += 1

cdef void _ObjectiveCalcDersMultiTarget(
    TConstArrayRef[double] approx,
    TConstArrayRef[float] target,
    float weight,
    TVector[double]* ders,
    THessianInfo* der2,
    void* customData
) with gil:
    cdef objectiveObject = <object>(customData)
    cdef TString errorMessage

    approxes = _CreateNumpyDoubleArrayView(approx.data(), approx.size())
    targetes = _CreateNumpyFloatArrayView(target.data(), target.size())

    try:
        ders_vector, second_ders_matrix = objectiveObject.calc_ders_multi(approxes, targetes, weight)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    for index, der in enumerate(ders_vector):
        dereference(ders)[index] = der

    if der2:
        index = 0
        for indY, line in enumerate(second_ders_matrix):
            for num in line[indY:]:
                dereference(der2).Data[index] = num
                index += 1

def _is_self_unused_in_method(method):
    args = inspect.getfullargspec(method)
    if not args.args:
        return False
    self_arg_name = args.args[0]
    return inspect.getsource(method).count(self_arg_name) == 1

def _check_object_and_class_methods_match(object_method, class_method):
    return inspect.getsource(object_method) == inspect.getsource(class_method)

def _try_jit_method(obj, method_name):
    import numba

    object_method = getattr(obj, method_name, None)
    class_method = getattr(obj.__class__, method_name, None)

    if not object_method:
        warnings.warn("Can't find method \"{}\" in the passed object".format(method_name))
        return
    if not class_method:
        warnings.warn("Can't find method \"{}\" in the class of the passed object".format(method_name))
        return
    if type(object_method) != types.MethodType:
        warnings.warn("Got unexpected type for method \"{}\" in the passed object: {}".format(method_name, type(object_method)))
        return
    if not _check_object_and_class_methods_match(object_method, class_method):
        warnings.warn("Methods \"{}\" in the passed object and its class don't match".format(method_name))
        return
    if not _is_self_unused_in_method(object_method):
        warnings.warn("Can't optimze method \"{}\" because self argument is used".format(method_name))
        return

    try:
        optimized = numba.njit(class_method)
    except numba.core.errors.NumbaError as err:
        warnings.warn("Failed to optimize method \"{}\" in the passed object:\n{}".format(method_name, err))
        return

    def new_method(*args):
        if not new_method.initialized:
            try:
                value = optimized(0, *args)
                return value
            except numba.core.errors.NumbaError as err:
                warnings.warn("Failed to optimize method \"{}\" in the passed object:\n{}".format(method_name, err))
                new_method.use_optimized = False
                return object_method(*args)
            finally:
                new_method.initialized = True
        elif new_method.use_optimized:
            return optimized(0, *args)
        else:
            return object_method(*args)
    setattr(new_method, "initialized", False)
    setattr(new_method, "use_optimized", True)
    setattr(obj, method_name, new_method)

def _try_jit_methods(obj, method_names):
    if hasattr(obj, "no_jit"):
        return

    if hasattr(obj, "_jited"): # everything already done
        return

    setattr(obj, "_jited", True)

    try:
        import numba
    except:
        warnings.warn('Failed to import numba for optimizing custom metrics and objectives')
        return

    for method_name in method_names:
        if hasattr(obj, method_name):
            _try_jit_method(obj, method_name)


# customGenerator should have method rvs()
cdef TCustomRandomDistributionGenerator _BuildCustomRandomDistributionGenerator(object customGenerator):
    cdef TCustomRandomDistributionGenerator descriptor
    descriptor.CustomData = <void*>customGenerator
    descriptor.EvalFunc = &_RandomDistGen
    return descriptor

cdef TCustomMetricDescriptor _BuildCustomMetricDescriptor(object metricObject):
    cdef TCustomMetricDescriptor descriptor
    _try_jit_methods(metricObject, custom_metric_methods_to_optimize)
    descriptor.CustomData = <void*>metricObject
    if (issubclass(metricObject.__class__, MultiTargetCustomMetric)):
        descriptor.EvalMultiTargetFunc = &_MultiTargetMetricEval
    else:
        descriptor.EvalFunc = &_MetricEval
    descriptor.GetDescriptionFunc = &_MetricGetDescription
    descriptor.IsMaxOptimalFunc = &_MetricIsMaxOptimal
    descriptor.GetFinalErrorFunc = &_MetricGetFinalError
    return descriptor

cdef TCustomCallbackDescriptor _BuildCustomCallbackDescritor(object callbackObject):
    cdef TCustomCallbackDescriptor descriptor
    descriptor.CustomData = <void*>callbackObject
    descriptor.AfterIterationFunc = &_CallbackAfterIteration
    return descriptor

cdef TCustomObjectiveDescriptor _BuildCustomObjectiveDescriptor(object objectiveObject):
    cdef TCustomObjectiveDescriptor descriptor
    _try_jit_methods(objectiveObject, custom_objective_methods_to_optimize)
    descriptor.CustomData = <void*>objectiveObject
    descriptor.CalcDersRange = &_ObjectiveCalcDersRange
    descriptor.CalcDersMultiTarget = &_ObjectiveCalcDersMultiTarget
    descriptor.CalcDersMultiClass = &_ObjectiveCalcDersMultiClass
    return descriptor


cdef EPredictionType string_to_prediction_type(prediction_type_str) except *:
    cdef EPredictionType prediction_type
    if not TryFromString[EPredictionType](to_arcadia_string(prediction_type_str), prediction_type):
        raise CatBoostError("Unknown prediction type {}.".format(prediction_type_str))
    return prediction_type

cdef transform_predictions(const TVector[TVector[double]]& predictions, EPredictionType predictionType, int thread_count, TFullModel* model):
    approx_dimension = model.GetDimensionsCount()

    if approx_dimension == 1:
        if predictionType == EPredictionType_Class:
            return np.array(_convert_to_visible_labels(predictionType, predictions, thread_count, model)[0])
        elif predictionType == EPredictionType_LogProbability:
            return np.transpose(_convert_to_visible_labels(predictionType, predictions, thread_count, model))
        else:
            pred_single_dim = _vector_of_double_to_np_array(predictions[0])
            if predictionType == EPredictionType_Probability:
                return np.transpose([1 - pred_single_dim, pred_single_dim])
            return pred_single_dim

    assert(approx_dimension > 1)
    return np.transpose(_convert_to_visible_labels(predictionType, predictions, thread_count, model))


cdef EModelType string_to_model_type(model_type_str) except *:
    cdef EModelType model_type
    if not TryFromString[EModelType](to_arcadia_string(model_type_str), model_type):
        raise CatBoostError("Unknown model type {}.".format(model_type_str))
    return model_type


cdef EFstrType string_to_fstr_type(fstr_type_str) except *:
    cdef EFstrType fstr_type
    if not TryFromString[EFstrType](to_arcadia_string(fstr_type_str), fstr_type):
        raise CatBoostError("Unknown type {}.".format(fstr_type_str))
    return fstr_type

cdef EPreCalcShapValues string_to_shap_mode(shap_mode_str) except *:
    cdef EPreCalcShapValues shap_mode
    if not TryFromString[EPreCalcShapValues](to_arcadia_string(shap_mode_str), shap_mode):
        raise CatBoostError("Unknown shap values mode {}.".format(shap_mode_str))
    return shap_mode

cdef ECalcTypeShapValues string_to_calc_type(shap_calc_type) except *:
    cdef ECalcTypeShapValues calc_type
    if not TryFromString[ECalcTypeShapValues](to_arcadia_string(shap_calc_type), calc_type):
        raise CatBoostError("Unknown shap values calculation type {}.".format(shap_calc_type))
    return calc_type


cdef EExplainableModelOutput string_to_model_output(model_output_str) except *:
    cdef EExplainableModelOutput model_output
    if not TryFromString[EExplainableModelOutput](to_arcadia_string(model_output_str), model_output):
        raise CatBoostError("Unknown shap values model output {}.".format(model_output_str))
    return model_output


cdef class _PreprocessParams:
    cdef TJsonValue tree
    cdef TMaybe[TCustomObjectiveDescriptor] customObjectiveDescriptor
    cdef TMaybe[TCustomMetricDescriptor] customMetricDescriptor
    cdef TMaybe[TCustomCallbackDescriptor] customCallbackDescriptor
    def __init__(self, dict params):
        eval_metric = params.get("eval_metric")
        objective = params.get("loss_function")
        callback = params.get("callbacks")

        is_custom_eval_metric = eval_metric is not None and not isinstance(eval_metric, string_types)
        is_custom_objective = objective is not None and not isinstance(objective, string_types)
        is_custom_callback = callback is not None

        devices = params.get('devices')
        if devices is not None and isinstance(devices, list):
            params['devices'] = ':'.join(map(str, devices))

        if 'verbose' in params:
            params['verbose'] = int(params['verbose'])

        params_to_json = params

        if is_custom_objective or is_custom_eval_metric or is_custom_callback:
            if params.get("task_type") == "GPU":
                raise CatBoostError("User defined loss functions, metrics and callbacks are not supported for GPU")
            keys_to_replace = set()
            if is_custom_objective:
                keys_to_replace.add("loss_function")
            if is_custom_eval_metric:
                keys_to_replace.add("eval_metric")
            if is_custom_callback:
                keys_to_replace.add("callbacks")

            params_to_json = {}

            for k, v in params.iteritems():
                if k in keys_to_replace:
                    continue
                params_to_json[k] = deepcopy(v)

            for k in keys_to_replace:
                params_to_json[k] = "PythonUserDefinedPerObject"

        if params_to_json.get("loss_function") == "PythonUserDefinedPerObject":
            self.customObjectiveDescriptor = _BuildCustomObjectiveDescriptor(params["loss_function"])
            if (issubclass(params["loss_function"].__class__, MultiTargetCustomObjective)):
                params_to_json["loss_function"] = "PythonUserDefinedMultiTarget"

        if params_to_json.get("eval_metric") == "PythonUserDefinedPerObject":
            self.customMetricDescriptor = _BuildCustomMetricDescriptor(params["eval_metric"])
            is_multitarget_metric = issubclass(params["eval_metric"].__class__, MultiTargetCustomMetric)
            if is_multitarget_objective(params_to_json["loss_function"]):
                assert is_multitarget_metric, \
                    "Custom eval metric should be inherited from MultiTargetCustomMetric for multi-target objective"
                params_to_json["eval_metric"] = "PythonUserDefinedMultiTarget"
            else:
                assert not is_multitarget_metric, \
                    "Custom eval metric should not be inherited from MultiTargetCustomMetric for single-target objective"

        if params_to_json.get("callbacks") == "PythonUserDefinedPerObject":
            self.customCallbackDescriptor = _BuildCustomCallbackDescritor(params["callbacks"])

        dumps_params = dumps(params_to_json, cls=_NumpyAwareEncoder)

        self.tree = ReadTJsonValue(to_arcadia_string(dumps_params))

cdef class _PreprocessGrids:
    cdef TJsonValue tree
    cdef THashMap[TString, TCustomRandomDistributionGenerator] custom_rnd_dist_gens
    cdef int rdg_enumeration
    def __prepare_grid(self, dict grid):
        loss_functions = grid.get("loss_function", [])

        params_to_json = {}
        for param_name, values in grid.iteritems():
            if isinstance(values, Iterable):
                params_to_json[param_name] = list(values)
            else:
                if hasattr(values, "rvs"):
                    rnd_name = "CustomRandomDistributionGenerator_" + str(self.rdg_enumeration)
                    params_to_json[param_name] = [rnd_name]
                    self.custom_rnd_dist_gens[to_arcadia_string(rnd_name)] = _BuildCustomRandomDistributionGenerator(values)
                    self.rdg_enumeration += 1
                else:
                    raise CatBoostError("Error: not iterable and not random distribytion generator object at grid")

        return params_to_json

    def __init__(self, list grids_list):
        prepared_grids = []
        self.rdg_enumeration = 0
        for grid in grids_list:
            prepared_grids.append(self.__prepare_grid(grid))
        dumps_grid = dumps(prepared_grids, cls=_NumpyAwareEncoder)
        self.tree = ReadTJsonValue(to_arcadia_string(dumps_grid))

cdef TString to_arcadia_string(s) except *:
    cdef const unsigned char[:] bytes_s
    cdef const char* utf8_str_pointer
    cdef Py_ssize_t utf8_str_size
    cdef type s_type = type(s)
    if len(s) == 0:
        return TString()
    if s_type is unicode or s_type is _npunicode_:
        # Fast path for most common case(s).
        if PY_MAJOR_VERSION >= 3:
            # we fallback to calling .encode method to properly report error
            utf8_str_pointer = PyUnicode_AsUTF8AndSize(s, &utf8_str_size)
            if utf8_str_pointer != nullptr:
                return TString(utf8_str_pointer, utf8_str_size)
        else:
            tmp = (<unicode>s).encode('utf8')
            return TString(<const char*>tmp, len(tmp))
    elif s_type is bytes or s_type is _npbytes_:
        return TString(<const char*>s, len(s))

    if PY_MAJOR_VERSION >= 3 and hasattr(s, 'encode'):
        # encode to the specific encoding used inside of the module
        bytes_s = s.encode('utf8')
    else:
        bytes_s = s
    return TString(<const char*>&bytes_s[0], len(bytes_s))

cdef to_native_str(binary):
    if PY_MAJOR_VERSION >= 3 and hasattr(binary, 'decode'):
        return binary.decode()
    return binary

cdef all_string_types_plus_bytes = string_types + (bytes,)

cdef _npbytes_ = np.bytes_
cdef _npunicode_ = np.unicode_
cdef _npint8 = np.int8
cdef _npint16 = np.int16
cdef _npuint8 = np.uint8
cdef _npuint16 = np.uint16

cdef _npint32 = np.int32
cdef _npint64 = np.int64
cdef _npuint32 = np.uint32
cdef _npuint64 = np.uint64
cdef _npfloat16 = np.float16
cdef _npfloat32 = np.float32
cdef _npfloat64 = np.float64

cpdef _prepare_cv_result(metric_name, const TVector[ui32]& iterations,
        const TVector[double]& average_train, const TVector[double]& std_dev_train,
        const TVector[double]& average_test, const TVector[double]& std_dev_test, result):
    # sklearn-style preparation
    fill_iterations_column = 'iterations' not in result
    if fill_iterations_column:
        result['iterations'] = list()
    for it in xrange(iterations.size()):
        iteration = iterations[it]
        if fill_iterations_column:
            result['iterations'].append(iteration)
        else:
            # ensure that all metrics have the same iterations specified
            assert(result['iterations'][it] == iteration)
        result["test-" + metric_name + "-mean"].append(average_test[it])
        result["test-" + metric_name + "-std"].append(std_dev_test[it])

        if average_train.size() != 0:
            result["train-" + metric_name + "-mean"].append(average_train[it])
            result["train-" + metric_name + "-std"].append(std_dev_train[it])
    return result


cdef inline bool_t is_np_int_type(type obj_type):
    return obj_type is _npint32 or obj_type is _npint64 or obj_type is _npint8 or obj_type is _npint16


cdef inline bool_t is_np_uint_type(type obj_type):
    return obj_type is _npuint32 or obj_type is _npuint64 or obj_type is _npuint8 or obj_type is _npuint16


cdef inline get_id_object_bytes_string_representation(
    object id_object,
    TString* bytes_string_buf_representation
):
    """
        returns python object, holding bytes_string_buf_representation memory,
        keep until bytes_string_buf_representation no longer needed

        Internal CatBoostError is typically catched up the calling stack to provide more detailed error
        description.
    """
    cdef double double_val
    cdef type obj_type = type(id_object)

    # For some reason Cython does not allow assignment to dereferenced pointer, so we are using ptr[0] trick
    # Here we have shortcuts for most of base types
    if obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npbytes_ or obj_type is _npunicode_:
        bytes_string_buf_representation[0] = to_arcadia_string(id_object)
    elif obj_type is int or obj_type is long or is_np_int_type(obj_type):
        bytes_string_buf_representation[0] = ToString[i64](<i64>id_object)
    elif is_np_uint_type(obj_type):
        bytes_string_buf_representation[0] = ToString[ui64](<ui64>id_object)
    elif obj_type is float or obj_type is _npfloat32 or obj_type is _npfloat64 or obj_type is _npfloat16:
        raise CatBoostError("bad object for id: {}".format(id_object))
    else:
        # this part is really heavy as it uses lot's of python internal magic, so put it down
        if isinstance(id_object, all_string_types_plus_bytes):
            # for some reason Cython does not allow assignment to dereferenced pointer, so use this trick instead
            bytes_string_buf_representation[0] = to_arcadia_string(id_object)
        else:
            if isnan(id_object) or int(id_object) != id_object:
                raise CatBoostError("bad object for id: {}".format(id_object))
            bytes_string_buf_representation[0] = ToString[i64](int(id_object))

cdef UpdateThreadCount(thread_count):
    if thread_count == -1:
        thread_count = CachedNumberOfCpus()
    if thread_count < 1:
        raise CatBoostError("Invalid thread_count value={} : must be > 0".format(thread_count))
    return thread_count


class FeaturesData(object):
    """
       class to store features data in optimized form to pass to Pool constructor

       stores the following:
          num_feature_data  - np.ndarray of (object_count x num_feature_count) with dtype=np.float32 or None
          cat_feature_data  - np.ndarray of (object_count x cat_feature_count) with dtype=object,
                              elements must have 'bytes' type, containing utf-8 encoded strings
          num_feature_names - sequence of (str or bytes) or None
          cat_feature_names - sequence of (str or bytes) or None

          if feature names are not specified they are initialized to empty strings.
    """

    def __init__(
        self,
        num_feature_data=None,
        cat_feature_data=None,
        num_feature_names=None,
        cat_feature_names=None
    ):
        if (num_feature_data is None) and (cat_feature_data is None):
            raise CatBoostError('at least one of num_feature_data, cat_feature_data params must be non-None')

        # list to pass 'by reference'
        all_feature_count_ref = [0]
        self._check_and_set_part('num', num_feature_data, np.float32, num_feature_names, all_feature_count_ref)
        self._check_and_set_part('cat', cat_feature_data, object, cat_feature_names, all_feature_count_ref)

        if all_feature_count_ref[0] == 0:
            raise CatBoostError('both num_feature_data and cat_feature_data contain 0 features')

        if (num_feature_data is not None) and (cat_feature_data is not None):
            if num_feature_data.shape[0] != cat_feature_data.shape[0]:
                raise CatBoostError(
                    'object_counts in num_feature_data ({}) and in cat_feature_data ({}) are different'.format(
                        len(num_feature_data), len(cat_feature_data)
                    )
                )

    def _check_and_set_part(
        self,
        part_name,
        feature_data,
        supported_element_type,
        feature_names,
        all_feature_count_ref # 1-element list to emulate pass-by-reference
    ):
        if (feature_names is not None) and (feature_data is None):
            raise CatBoostError(
                '{}_feature_names specified with not specified {}_feature_data'.format(
                    part_name, part_name
                )
            )
        if feature_data is not None:
            if not isinstance(feature_data, np.ndarray):
                raise CatBoostError(
                    'only np.ndarray type is supported for {}_feature_data'.format(part_name)
                )
            if len(feature_data.shape) != 2:
                raise CatBoostError(
                    '{}_feature_data must be 2D numpy.ndarray, it has shape {} instead'.format(
                        part_name, feature_data.shape
                    )
                )
            if feature_data.dtype != supported_element_type:
                raise CatBoostError(
                    '{}_feature_data element type must be {}, found {} instead'.format(
                        part_name, supported_element_type, feature_data.dtype
                    )
                )
            if feature_names is not None:
                for i, name in enumerate(feature_names):
                    if type(name) != str:
                        raise CatBoostError(
                            'type of {}_feature_names[{}]: expected str, found {}'.format(
                                part_name, i, type(name)
                            )
                        )
                if feature_data.shape[1] != len(feature_names):
                    raise CatBoostError(
                        (
                            'number of features in {}_feature_data (={}) is different from '
                            ' len({}_feature_names) (={})'.format(
                                part_name, feature_data.shape[1], part_name, len(feature_names)
                            )
                        )
                    )
            all_feature_count_ref[0] += feature_data.shape[1]

        setattr(self, part_name + '_feature_data', feature_data)
        setattr(
            self,
            part_name + '_feature_names',
            (
                feature_names if feature_names is not None
                else (['']*feature_data.shape[1] if feature_data is not None else [])
            )
        )

    def get_object_count(self):
        if self.num_feature_data is not None:
            return self.num_feature_data.shape[0]
        if self.cat_feature_data is not None:
            return self.cat_feature_data.shape[0]

    def get_num_feature_count(self):
        return self.num_feature_data.shape[1] if self.num_feature_data is not None else 0

    def get_cat_feature_count(self):
        return self.cat_feature_data.shape[1] if self.cat_feature_data is not None else 0

    def get_feature_count(self):
        return self.get_num_feature_count() + self.get_cat_feature_count()

    def get_feature_names(self):
        """
            empty strings are returned for features for which no names data was specified
        """
        return self.num_feature_names + self.cat_feature_names


cdef void list_to_vector(values_list, TVector[ui32]* values_vector) except *:
    if values_list is not None:
        for value in values_list:
            values_vector[0].push_back(value)


cdef TFeaturesLayout* _init_features_layout(
    data,
    embedding_features_data,
    cat_features,
    text_features,
    embedding_features,
    feature_names,
    feature_tags
) except*:
    cdef TVector[ui32] cat_features_vector
    cdef TVector[ui32] text_features_vector
    cdef TVector[ui32] embedding_features_vector
    cdef TVector[TString] feature_names_vector
    cdef THashMap[TString, TTagDescription] feature_tags_map
    cdef bool_t all_features_are_sparse

    if isinstance(data, FeaturesData):
        feature_count = data.get_feature_count()
        cat_features = [i for i in range(data.get_num_feature_count(), feature_count)]
        feature_names = data.get_feature_names()
    else:
        feature_count = np.shape(data)[1]
        if embedding_features_data is not None:
            feature_count += len(embedding_features_data)

    list_to_vector(cat_features, &cat_features_vector)
    list_to_vector(text_features, &text_features_vector)
    list_to_vector(embedding_features, &embedding_features_vector)

    if feature_names is not None:
        for feature_name in feature_names:
            feature_names_vector.push_back(to_arcadia_string(str(feature_name)))

    if feature_tags is not None:
        for tag_name in feature_tags:
            tag_key = to_arcadia_string(str(tag_name))
            list_to_vector(feature_tags[tag_name]['features'], &feature_tags_map[tag_key].Features)
            feature_tags_map[tag_key].Cost = feature_tags[tag_name]['cost']

    all_features_are_sparse = False
    if isinstance(data, SPARSE_MATRIX_TYPES):
        if embedding_features_data is not None:
            all_features_are_sparse = all([isinstance(embedding_data, SPARSE_MATRIX_TYPES) for embedding_data in embedding_features_data])
        else:
            all_features_are_sparse = True

    return new TFeaturesLayout(
        <ui32>feature_count,
        cat_features_vector,
        text_features_vector,
        embedding_features_vector,
        feature_names_vector,
        feature_tags_map,
        all_features_are_sparse)

cdef TVector[bool_t] _get_is_feature_type_mask(const TFeaturesLayout* featuresLayout, EFeatureType featureType) except *:
    cdef TVector[bool_t] mask
    mask.resize(featuresLayout.GetExternalFeatureCount(), False)

    cdef ui32 idx
    for idx in range(featuresLayout.GetExternalFeatureCount()):
        if featuresLayout[0].GetExternalFeatureType(idx) == featureType:
            mask[idx] = True

    return mask

cdef TVector[ui32] _get_main_data_feature_idx_to_dst_feature_idx(const TFeaturesLayout* featuresLayout, bool_t hasSeparateEmbeddingFeaturesData):
    cdef TVector[ui32] result

    if hasSeparateEmbeddingFeaturesData:
        result.reserve(featuresLayout.GetExternalFeatureCount() - featuresLayout.GetEmbeddingFeatureCount())
        for idx in range(featuresLayout.GetExternalFeatureCount()):
            if not (featuresLayout[0].GetExternalFeatureType(idx) == EFeatureType_Embedding):
                result.push_back(idx)
    else:
        result.reserve(featuresLayout.GetExternalFeatureCount())
        for idx in range(featuresLayout.GetExternalFeatureCount()):
            result.push_back(idx)

    return result


cdef _get_object_count(data):
    if isinstance(data, FeaturesData):
        return data.get_object_count()
    else:
        return np.shape(data)[0]

@cython.boundscheck(False)
@cython.wraparound(False)
def _set_features_order_data_features_data(
    np.ndarray[numpy_num_dtype, ndim=2] num_feature_values,
    np.ndarray[object, ndim=2] cat_feature_values,
    Py_FeaturesOrderBuilderVisitor py_builder_visitor
):

    """
        older buffer interface is used instead of memory views because of
        https://github.com/cython/cython/issues/1772, https://github.com/cython/cython/issues/2485
    """

    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    py_builder_visitor.get_raw_features_order_data_visitor(&builder_visitor)

    if (num_feature_values is None) and (cat_feature_values is None):
        raise CatBoostError('both num_feature_values and cat_feature_values are empty')

    cdef ui32 doc_count = <ui32>(
        num_feature_values.shape[0] if num_feature_values is not None else cat_feature_values.shape[0]
    )

    cdef ui32 num_feature_count = <ui32>(num_feature_values.shape[1] if num_feature_values is not None else 0)
    cdef ui32 cat_feature_count = <ui32>(cat_feature_values.shape[1] if cat_feature_values is not None else 0)

    cdef Py_FloatSequencePtr py_num_factor_data
    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TString factor_string
    cdef TVector[TString] cat_factor_data
    cdef ui32 doc_idx
    cdef ui32 num_feature_idx
    cdef ui32 cat_feature_idx

    cdef ui32 dst_feature_idx

    cat_factor_data.reserve(doc_count)
    dst_feature_idx = <ui32>0

    dst_feature_idx = 0
    for num_feature_idx in range(num_feature_count):
        py_num_factor_data = make_non_owning_type_cast_array_holder(num_feature_values[:,num_feature_idx])
        py_num_factor_data.get_result(&num_factor_data)
        builder_visitor[0].AddFloatFeature(dst_feature_idx, num_factor_data)

        dst_feature_idx += 1
    for cat_feature_idx in range(cat_feature_count):
        cat_factor_data.clear()
        for doc_idx in range(doc_count):
            factor_string = to_arcadia_string(cat_feature_values[doc_idx, cat_feature_idx])
            cat_factor_data.push_back(factor_string)
        builder_visitor[0].AddCatFeature(dst_feature_idx, <TConstArrayRef[TString]>cat_factor_data)
        dst_feature_idx += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def _set_features_order_data_ndarray(
    np.ndarray[numpy_num_dtype, ndim=2] feature_values,
    ui32 [:] src_feature_idx_to_dst_feature_idx,
    bool_t [:] is_cat_feature_mask,
    bool_t [:] is_text_feature_mask,
    bool_t [:] is_embedding_feature_mask,
    Py_FeaturesOrderBuilderVisitor py_builder_visitor
):

    """
        older buffer interface is used instead of memory views because of
        https://github.com/cython/cython/issues/1772, https://github.com/cython/cython/issues/2485
    """

    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    py_builder_visitor.get_raw_features_order_data_visitor(&builder_visitor)

    cdef ui32 doc_count = <ui32>(feature_values.shape[0])
    cdef ui32 src_feature_count = <ui32>(feature_values.shape[1])

    cdef Py_FloatSequencePtr py_num_factor_data
    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TVector[TString] string_factor_data
    cdef ui32 doc_idx

    cdef ui32 src_flat_feature_idx
    cdef ui32 flat_feature_idx

    string_factor_data.reserve(doc_count)

    for src_flat_feature_idx in range(src_feature_count):
        flat_feature_idx = src_feature_idx_to_dst_feature_idx[src_flat_feature_idx]
        if is_cat_feature_mask[flat_feature_idx] or is_text_feature_mask[flat_feature_idx]:
            string_factor_data.clear()
            for doc_idx in range(doc_count):
                string_factor_data.push_back(ToString(feature_values[doc_idx, src_flat_feature_idx]))
            if is_cat_feature_mask[flat_feature_idx]:
                builder_visitor[0].AddCatFeature(flat_feature_idx, <TConstArrayRef[TString]>string_factor_data)
            else:
                builder_visitor[0].AddTextFeature(flat_feature_idx, <TConstArrayRef[TString]>string_factor_data)
        elif is_embedding_feature_mask[flat_feature_idx]:
            raise CatBoostError(
                (
                    'Feature_idx={}: feature is Embedding but features data is specified as numpy.ndarray '
                    + ' with non-object type'
                ).format(
                    feature_idx=flat_feature_idx
                )
            )
        else:
            py_num_factor_data = make_non_owning_type_cast_array_holder(feature_values[:,src_flat_feature_idx])
            py_num_factor_data.get_result(&num_factor_data)
            builder_visitor[0].AddFloatFeature(flat_feature_idx, num_factor_data)


# returns new data holders array
cdef object _set_features_order_embedding_features_data(
    embedding_features_data,
    feature_names,
    const TFeaturesLayout* features_layout,
    IRawFeaturesOrderDataVisitor* builder_visitor
):
    cdef ui32 flat_feature_idx = 0
    cdef ui32 embedding_feature_idx = 0
    cdef ITypedSequencePtr[TEmbeddingData] embedding_factor_data

    cdef bool_t src_is_dict = isinstance(embedding_features_data, dict)
    if src_is_dict and (feature_names is None):
        feature_names = [i for i in range(features_layout[0].GetExternalFeatureCount())]

    new_data_holders = []
    for flat_feature_idx in features_layout[0].GetEmbeddingFeatureInternalIdxToExternalIdx():
        new_data_holders += create_embedding_factor_data(
            flat_feature_idx,
            embedding_features_data[feature_names[flat_feature_idx] if src_is_dict else embedding_feature_idx],
            &embedding_factor_data
        )
        builder_visitor[0].AddEmbeddingFeature(flat_feature_idx, embedding_factor_data)
        embedding_feature_idx += 1

    return new_data_holders

# returns new data holders array
cdef object _set_objects_order_embedding_features_data(
    embedding_features_data,
    feature_names,
    const TFeaturesLayout* features_layout,
    IRawObjectsOrderDataVisitor* builder_visitor
):
    cdef ui32 object_count = 0
    cdef ui32 object_idx = 0
    cdef ui32 flat_feature_idx = 0
    cdef ui32 embedding_feature_idx = 0
    cdef TEmbeddingData object_embedding_data

    cdef bool_t src_is_dict = isinstance(embedding_features_data, dict)

    if src_is_dict and (feature_names is None):
        feature_names = [i for i in range(features_layout[0].GetExternalFeatureCount())]

    new_data_holders = []

    if len(embedding_features_data) > 0:
        object_count = len(next(iter(embedding_features_data.values())) if src_is_dict else embedding_features_data[0])
        if object_count > 0:
            for flat_feature_idx in features_layout[0].GetEmbeddingFeatureInternalIdxToExternalIdx():
                embedding_feature_data = embedding_features_data[feature_names[flat_feature_idx] if src_is_dict else embedding_feature_idx]
                embedding_dimension = len(embedding_feature_data[0])
                for object_idx in range(object_count):
                    new_data_holders += get_embedding_array_data(
                        object_idx,
                        flat_feature_idx,
                        embedding_dimension,
                        embedding_feature_data[object_idx],
                        &object_embedding_data
                    )
                    builder_visitor[0].AddEmbeddingFeature(object_idx, flat_feature_idx, object_embedding_data)
                embedding_feature_idx += 1

    return new_data_holders


cdef float get_float_feature(ui32 non_default_doc_idx, ui32 flat_feature_idx, src_value) except*:
    try:
        return _FloatOrNan(src_value)
    except TypeError as e:
        raise CatBoostError(
            'Bad value for num_feature[non_default_doc_idx={},feature_idx={}]="{}": {}'.format(
                non_default_doc_idx,
                flat_feature_idx,
                src_value,
                e
            )
        )

# returns new data holders array
cdef create_num_factor_data(
    ui32 flat_feature_idx,
    np.ndarray column_values,
    ITypedSequencePtr[np.float32_t]* result
):
    # two pointers are needed as a workaround for Cython assignment of derived types restrictions
    cdef TIntrusivePtr[TVectorHolder[float]] num_factor_data
    cdef TIntrusivePtr[IResourceHolder] num_factor_data_holder

    cdef Py_FloatSequencePtr py_num_factor_data

    cdef ui32 doc_idx

    if len(column_values) == 0:
        result[0] = MakeNonOwningTypeCastArrayHolder[np.float32_t, np.float32_t](
            <const np.float32_t*>nullptr,
            <const np.float32_t*>nullptr
        )
        return []
    elif column_values.dtype in numpy_num_dtype_list: # Cython cannot use fused type lists in normal code
        if not column_values.flags.c_contiguous:
            column_values = np.ascontiguousarray(column_values, dtype=np.float32)
        py_num_factor_data = make_non_owning_type_cast_array_holder(column_values)
        py_num_factor_data.get_result(result)
        return [column_values]
    else:
        num_factor_data = new TVectorHolder[float]()
        num_factor_data.Get()[0].Data.resize(len(column_values))
        for doc_idx in range(len(column_values)):
            num_factor_data.Get()[0].Data[doc_idx] = get_float_feature(
                doc_idx,
                flat_feature_idx,
                column_values[doc_idx]
            )
        num_factor_data_holder.Reset(num_factor_data.Get())
        result[0] = MakeTypeCastArrayHolder[np.float32_t, np.float32_t](
            TMaybeOwningConstArrayHolder[np.float32_t].CreateOwning(
                <TConstArrayRef[np.float32_t]>num_factor_data.Get()[0].Data,
                num_factor_data_holder
            )
        )

        return []

cdef get_cat_factor_bytes_representation(
    int non_default_doc_idx, # can be -1 - that means default value for sparse data
    ui32 feature_idx,
    object factor,
    TString* factor_strbuf
):
    try:
        get_id_object_bytes_string_representation(factor, factor_strbuf)
    except CatBoostError:
        if non_default_doc_idx == -1:
            doc_description = 'default value for sparse data'
        else:
            doc_description = 'non-default value idx={}'.format(non_default_doc_idx)

        raise CatBoostError(
            'Invalid type for cat_feature[{},feature_idx={}]={} :'
            ' cat_features must be integer or string, real number values and NaN values'
            ' should be converted to string.'.format(doc_description, feature_idx, factor)
        )

cdef get_text_factor_bytes_representation(
    int non_default_doc_idx, # can be -1 - that means default value for sparse data
    ui32 feature_idx,
    object factor,
    TString* factor_strbuf
):
    cdef type obj_type = type(factor)
    if obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npbytes_ or obj_type is _npunicode_:
        factor_strbuf[0] = to_arcadia_string(factor)
    else:
        if non_default_doc_idx == -1:
            doc_description = 'default value for sparse data'
        else:
            doc_description = 'non-default value idx={}'.format(non_default_doc_idx)

        raise CatBoostError(
            'Invalid type for text_feature[{},feature_idx={}]={} :'
            ' text_features must have string type'.format(doc_description, feature_idx, factor)
        )


cdef TVector[np.float32_t] get_embedding_array_as_vector(
    ui32 object_idx,
    ui32 flat_feature_idx,
    size_t embedding_dimension,
    src_array
):
    cdef TVector[np.float32_t] object_embedding_data
    if len(src_array) != embedding_dimension:
        raise CatBoostError(
            (
                'Inсonsistent array size for embedding_feature[object_idx={},feature_idx={}]={}, '
                + 'should be equal to array size for the first object ={}'
            ).format(
                object_idx,
                flat_feature_idx,
                len(src_array),
                embedding_dimension
            )
        )

    # TODO(akhropov): make yresize accessible in Cython
    object_embedding_data.resize(embedding_dimension)
    for element_idx in range(embedding_dimension):
        try:
            object_embedding_data[element_idx] = _FloatOrNan(src_array[element_idx])
        except TypeError as e:
            raise CatBoostError(
                'Bad value for embedding[object_idx={},feature_idx={},array_idx={}]="{}": {}'.format(
                    object_idx,
                    flat_feature_idx,
                    element_idx,
                    src_array[element_idx],
                    e
                )
            )
    return object_embedding_data


# returns new data holders array
cdef get_embedding_array_data(
    ui32 object_idx,
    ui32 flat_feature_idx,
    size_t embedding_dimension,
    src_array,
    TEmbeddingData* result
):
    cdef np.float32_t[:] src_array_data
    cdef TVector[np.float32_t] embedding_data

    if len(src_array) != embedding_dimension:
        raise CatBoostError(
            (
                'Inсonsistent array size for embedding_feature[object_idx={},feature_idx={}]={}, '
                + 'should be equal to array size for the first object ={}'
            ).format(
                object_idx,
                flat_feature_idx,
                len(src_array),
                embedding_dimension
            )
        )
    if isinstance(src_array, np.ndarray) and (src_array.dtype == np.float32) and src_array.flags.c_contiguous:
        src_array_data = src_array
        result[0] = TMaybeOwningConstArrayHolder[np.float32_t].CreateNonOwning(
            TConstArrayRef[np.float32_t](
                <np.float32_t*>&src_array_data[0],
                embedding_dimension
            )
        )
        return [src_array]
    else:
        embedding_data = get_embedding_array_as_vector(
            object_idx,
            flat_feature_idx,
            embedding_dimension,
            src_array
        )
        result[0] = TMaybeOwningConstArrayHolder[np.float32_t].CreateOwningMovedFrom(embedding_data)
        return []


# returns new data holders array
cdef create_embedding_factor_data(
    ui32 flat_feature_idx,
    np.ndarray column_values,
    ITypedSequencePtr[TEmbeddingData]* result
):
    cdef TVector[TEmbeddingData] data
    cdef TVector[np.float32_t] object_embedding_data

    cdef Py_EmbeddingSequencePtr py_embedding_factor_data

    cdef size_t object_count = len(column_values)
    cdef size_t embedding_dimension = len(column_values[0])

    if object_count == 0:
        result[0] = MakeNonOwningTypeCastArrayHolder[TEmbeddingData, TEmbeddingData](
            <const TEmbeddingData*>nullptr,
            <const TEmbeddingData*>nullptr
        )
        return []
    elif isinstance(column_values[0], np.ndarray) and (column_values[0].dtype in numpy_num_dtype_list):
        py_embedding_factor_data, data_holders = make_embedding_type_cast_array_holder(
            flat_feature_idx,
            column_values[0],
            column_values
        )
        py_embedding_factor_data.get_result(result)
        return data_holders
    else:
        data.reserve(object_count)
        for object_idx in range(object_count):
            object_embedding_data = get_embedding_array_as_vector(
                object_idx,
                flat_feature_idx,
                embedding_dimension,
                column_values[object_idx]
            )
            data.push_back(
                TMaybeOwningConstArrayHolder[np.float32_t].CreateOwningMovedFrom(object_embedding_data)
            )

        result[0] = MakeTypeCastArrayHolderFromVector[TEmbeddingData, TEmbeddingData](data)
        return []


# returns new data holders array
cdef get_canonical_type_indexing_array(np.ndarray indices, TMaybeOwningConstArrayHolder[ui32] * result):

    """
        older buffer interface is used instead of memory views because of
        https://github.com/cython/cython/issues/1772, https://github.com/cython/cython/issues/2485
    """
    cdef np.ndarray[np.int32_t, ndim=1] indices_i32
    cdef np.ndarray[np.int64_t, ndim=1] indices_i64

    if len(indices) == 0:
        result[0] = TMaybeOwningConstArrayHolder[ui32].CreateNonOwning(TConstArrayRef[ui32]())
        return []
    elif indices.dtype == np.int32:
        indices_i32 = np.ascontiguousarray(indices, dtype=np.int32)
        result[0] = TMaybeOwningConstArrayHolder[ui32].CreateNonOwning(
            TConstArrayRef[ui32](<ui32*>&indices_i32[0], len(indices_i32))
        )
        return [indices_i32]
    elif indices.dtype == np.int64:
        indices_i64 = np.ascontiguousarray(indices, dtype=np.int64)
        result[0] = CreateConstOwningWithMaybeTypeCast[ui32, i64](
            TMaybeOwningArrayHolder[i64].CreateNonOwning(
                TArrayRef[i64](<i64*>&indices_i64[0], len(indices_i64))
            )
        )
        return [indices_i64]
    else:
        raise CatBoostError('unsupported index dtype = {}'.format(indices.dtype))

# returns new data holders array
cdef get_sparse_array_indexing(pandas_sparse_index, ui32 doc_count, TSparseArrayIndexingPtr[ui32] * result):
    cdef TMaybeOwningConstArrayHolder[ui32] array_indices
    cdef TMaybeOwningConstArrayHolder[ui32] block_starts
    cdef TMaybeOwningConstArrayHolder[ui32] block_lengths

    new_data_holders = []

    if isinstance(pandas_sparse_index, pd._libs.sparse.IntIndex):
        new_data_holders += get_canonical_type_indexing_array(pandas_sparse_index.indices, & array_indices)
        result[0] = MakeSparseArrayIndexing(doc_count, array_indices)
    elif isinstance(pandas_sparse_index, pd._libs.sparse.BlockIndex):
        new_data_holders += get_canonical_type_indexing_array(pandas_sparse_index.blocs, & block_starts)
        new_data_holders += get_canonical_type_indexing_array(pandas_sparse_index.blengths, & block_lengths)
        result[0] = MakeSparseBlockIndexing(doc_count, block_starts, block_lengths)
    else:
        raise CatBoostError('unknown pandas sparse index type = {}'.format(type(pandas_sparse_index)))

    return new_data_holders

# returns new data holders array
cdef object _set_features_order_data_pd_data_frame_sparse_column(
    ui32 doc_count,
    ui32 flat_feature_idx,
    bool_t is_cat_feature,
    bool_t is_embedding_feature,
    object column_values, # pd.SparseArray, but Cython requires cimport to provide type here
    TVector[TString]* cat_factor_data,  # pass from parent to avoid reallocations
    IRawFeaturesOrderDataVisitor* builder_visitor
):
    cdef TSparseArrayIndexingPtr[ui32] indexing

    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TString factor_string

    cdef int non_default_idx

    new_data_holders = []

    new_data_holders += get_sparse_array_indexing(column_values._sparse_index, doc_count, & indexing)

    non_default_values = column_values._sparse_values

    if is_cat_feature:
        cat_factor_data[0].clear()
        for non_default_idx in range(non_default_values.shape[0]):
            get_cat_factor_bytes_representation(
                non_default_idx,
                flat_feature_idx,
                non_default_values[non_default_idx],
                &factor_string
            )
            cat_factor_data[0].push_back(factor_string)

        # get default value
        get_cat_factor_bytes_representation(
            -1,
            flat_feature_idx,
            column_values.fill_value,
            &factor_string
        )

        builder_visitor[0].AddCatFeature(
            flat_feature_idx,
            MakeConstPolymorphicValuesSparseArray[TString, TString, ui32](
                indexing,
                TMaybeOwningConstArrayHolder[TString].CreateNonOwning(
                    <TConstArrayRef[TString]> cat_factor_data[0]
                ),
                factor_string
            )
        )
    elif is_embedding_feature:
        raise CatBoostError(
            'Feature_idx={}: Sparse data for embedding features is not supported yet'.format(
                feature_idx=flat_feature_idx
            )
        )
    else:
        new_data_holders += create_num_factor_data(flat_feature_idx, non_default_values, &num_factor_data)
        builder_visitor[0].AddFloatFeature(
            flat_feature_idx,
            MakeConstPolymorphicValuesSparseArrayGeneric[float, ui32](
                indexing,
                num_factor_data,
                column_values.fill_value
            )
        )

    return new_data_holders


cdef _set_features_order_data_pd_data_frame_categorical_column(
    ui32 flat_feature_idx,
    object column_values, # pd.Categorical, but Cython requires cimport to provide type here
    TString* factor_string,

    # array of [dst_value_for_cateory0, dst_value_for_category1 ...]
    TVector[ui32]* categories_as_hashed_cat_values,

    IRawFeaturesOrderDataVisitor* builder_visitor
):
    cdef np.ndarray categories_values = column_values.categories.values
    cdef ui32 categories_values_size = categories_values.shape[0]

    cdef np.ndarray categories_codes = column_values.codes
    cdef ui32 doc_count = categories_codes.shape[0]

    # access through TArrayRef is faster
    cdef TArrayRef[ui32] categories_as_hashed_cat_values_ref
    cdef TVector[ui32] hashed_cat_values

    cdef ui32 category_idx
    cdef ui32 doc_idx
    cdef i32 category_code


    # TODO(akhropov): make yresize accessible in Cython
    categories_as_hashed_cat_values[0].resize(categories_values_size)
    categories_as_hashed_cat_values_ref = <TArrayRef[ui32]>categories_as_hashed_cat_values[0]
    for category_idx in range(categories_values_size):
        try:
            get_id_object_bytes_string_representation(categories_values[category_idx], factor_string)
        except CatBoostError:
            raise CatBoostError(
                'Invalid type for cat_feature category for [feature_idx={}]={} :'
                ' cat_features must be integer or string, real number values and NaN values'
                ' should be converted to string.'.format(flat_feature_idx, categories_values[category_idx])
            )

        categories_as_hashed_cat_values_ref[category_idx]  = builder_visitor[0].GetCatFeatureValue(
            flat_feature_idx,
            factor_string[0]
        )

    # TODO(akhropov): make yresize accessible in Cython
    hashed_cat_values.resize(doc_count)
    for doc_idx in range(doc_count):
        category_code = categories_codes[doc_idx]
        if category_code == -1:
            raise CatBoostError(
                'Invalid type for cat_feature[object_idx={},feature_idx={}]=NaN :'
                ' cat_features must be integer or string, real number values and NaN values'
                ' should be converted to string.'.format(doc_idx, flat_feature_idx)
            )

        hashed_cat_values[doc_idx] = categories_as_hashed_cat_values_ref[category_code]

    builder_visitor[0].AddCatFeature(
        flat_feature_idx,
        TMaybeOwningConstArrayHolder[ui32].CreateOwningMovedFrom(hashed_cat_values)
    )


# returns new data holders array
cdef object _set_features_order_data_pd_data_frame(
    data_frame,
    bool_t has_separate_embedding_features_data,
    const TFeaturesLayout* features_layout,
    IRawFeaturesOrderDataVisitor* builder_visitor
):
    cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(features_layout, has_separate_embedding_features_data)
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
    cdef TVector[bool_t] is_text_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)
    cdef TVector[bool_t] is_embedding_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Embedding)
    cdef ui32 doc_count = data_frame.shape[0]

    cdef TString factor_string

    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TVector[TString] string_factor_data

    cdef ITypedSequencePtr[TEmbeddingData] embedding_factor_data

    # this buffer for categorical processing is here to avoid reallocations in
    # _set_features_order_data_pd_data_frame_categorical_column

    # array of [dst_value_for_cateory0, dst_value_for_category1 ...]
    cdef TVector[ui32] categories_as_hashed_cat_values

    cdef ui32 doc_idx
    cdef ui32 flat_feature_idx
    cdef np.ndarray column_values # for columns that are not Sparse or Categorical

    string_factor_data.reserve(doc_count)

    new_data_holders = []
    for src_flat_feature_idx, (column_name, column_data) in enumerate(data_frame.items()):
        flat_feature_idx = main_data_feature_idx_to_dst_feature_idx[src_flat_feature_idx]
        if isinstance(column_data.dtype, pd.SparseDtype):
            new_data_holders += _set_features_order_data_pd_data_frame_sparse_column(
                doc_count,
                flat_feature_idx,
                is_cat_feature_mask[flat_feature_idx],
                is_embedding_feature_mask[flat_feature_idx],
                column_data.values,
                &string_factor_data,
                builder_visitor
            )
        elif column_data.dtype.name == 'category':
            if not is_cat_feature_mask[flat_feature_idx]:
                raise CatBoostError(
                    ("features data: pandas.DataFrame column '%s' has dtype 'category' but is not in "
                    + " cat_features list") % column_name
                )

            _set_features_order_data_pd_data_frame_categorical_column(
                flat_feature_idx,
                column_data.values,
                &factor_string,
                &categories_as_hashed_cat_values,
                builder_visitor
            )
        else:
            column_values = column_data.to_numpy()
            if is_cat_feature_mask[flat_feature_idx]:
                string_factor_data.clear()
                for doc_idx in range(doc_count):
                    get_cat_factor_bytes_representation(
                        doc_idx,
                        flat_feature_idx,
                        column_values[doc_idx],
                        &factor_string
                    )
                    string_factor_data.push_back(factor_string)
                builder_visitor[0].AddCatFeature(flat_feature_idx, <TConstArrayRef[TString]>string_factor_data)
            elif is_text_feature_mask[flat_feature_idx]:
                string_factor_data.clear()
                for doc_idx in range(doc_count):
                    get_text_factor_bytes_representation(
                        doc_idx,
                        flat_feature_idx,
                        column_values[doc_idx],
                        &factor_string
                    )
                    string_factor_data.push_back(factor_string)
                builder_visitor[0].AddTextFeature(flat_feature_idx, <TConstArrayRef[TString]>string_factor_data)
            elif is_embedding_feature_mask[flat_feature_idx]:
                new_data_holders += create_embedding_factor_data(
                    flat_feature_idx,
                    column_values,
                    &embedding_factor_data
                )
                builder_visitor[0].AddEmbeddingFeature(flat_feature_idx, embedding_factor_data)
            else:
                new_data_holders += create_num_factor_data(
                    flat_feature_idx,
                    column_values,
                    &num_factor_data
                )
                builder_visitor[0].AddFloatFeature(flat_feature_idx, num_factor_data)

    return new_data_holders


cdef _set_data_np(
    const float [:,:] num_feature_values,
    object [:,:] cat_feature_values, # cannot be const due to https://github.com/cython/cython/issues/2485
    bool_t has_separate_embedding_features_data,
    const TFeaturesLayout* features_layout,
    IRawObjectsOrderDataVisitor* builder_visitor
):
    if (num_feature_values is None) and (cat_feature_values is None):
        raise CatBoostError('both num_feature_values and cat_feature_values are empty')

    cdef ui32 doc_count = <ui32>(
        num_feature_values.shape[0] if num_feature_values is not None else cat_feature_values.shape[0]
    )

    cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(features_layout, has_separate_embedding_features_data)
    cdef ui32 num_feature_count = <ui32>(num_feature_values.shape[1] if num_feature_values is not None else 0)
    cdef ui32 cat_feature_count = <ui32>(cat_feature_values.shape[1] if cat_feature_values is not None else 0)

    cdef ui32 doc_idx
    cdef ui32 num_feature_idx
    cdef ui32 cat_feature_idx

    cdef ui32 src_feature_idx
    for doc_idx in range(doc_count):
        src_feature_idx = <ui32>0
        for num_feature_idx in range(num_feature_count):
            builder_visitor[0].AddFloatFeature(
                doc_idx,
                main_data_feature_idx_to_dst_feature_idx[src_feature_idx],
                num_feature_values[doc_idx, num_feature_idx]
            )
            src_feature_idx += 1
        for cat_feature_idx in range(cat_feature_count):
            builder_visitor[0].AddCatFeature(
                doc_idx,
                main_data_feature_idx_to_dst_feature_idx[src_feature_idx],
                <TStringBuf>to_arcadia_string(cat_feature_values[doc_idx, cat_feature_idx])
            )
            src_feature_idx += 1

# scipy.sparse matrixes always have default value 0
cdef _set_cat_features_default_values_for_scipy_sparse(
    const TFeaturesLayout * features_layout,
    IRawObjectsOrderDataVisitor * builder_visitor
):
    cdef TString default_value = "0"
    cdef TConstArrayRef[ui32] cat_features_flat_indices = features_layout[0].GetCatFeatureInternalIdxToExternalIdx()

    for flat_feature_idx in cat_features_flat_indices:
        builder_visitor.AddCatFeatureDefaultValue(flat_feature_idx, default_value)

cdef _get_categorical_feature_value_from_scipy_sparse(
    int doc_idx,
    int feature_idx,
    value,
    bool_t is_float_value,
    TString * factor_string_buf
):
    if is_float_value:
        raise CatBoostError(
            'Invalid value for cat_feature[{doc_idx},{feature_idx}]={value}'
            +' cat_features must be integer or string, real number values and NaN values'
            +' should be converted to string'.format(doc_idx=doc_idx, feature_idx=feature_idx, value=value))
    else:
        factor_string_buf[0] = ToString[i64](<i64>value)

cdef _add_single_feature_value_from_scipy_sparse(
    int doc_idx,
    ui32 feature_idx,
    value,
    bool_t is_float_value,
    TConstArrayRef[bool_t] is_cat_feature_mask,
    TString * factor_string_buf,
    IRawObjectsOrderDataVisitor * builder_visitor
):
    if is_cat_feature_mask[feature_idx]:
        _get_categorical_feature_value_from_scipy_sparse(
            doc_idx,
            feature_idx,
            value,
            is_float_value,
            factor_string_buf
        )
        builder_visitor[0].AddCatFeature(doc_idx, feature_idx, <TStringBuf>factor_string_buf[0])
    else:
        builder_visitor[0].AddFloatFeature(doc_idx, feature_idx, value)

cdef _set_data_from_scipy_bsr_sparse(
    data,
    TConstArrayRef[ui32] main_data_feature_idx_to_dst_feature_idx,
    TConstArrayRef[bool_t] is_cat_feature_mask,
    IRawObjectsOrderDataVisitor * builder_visitor
):
    data_shape = np.shape(data)
    cdef int doc_count = data_shape[0]
    cdef int feature_count = data_shape[1]

    if doc_count == 0:
        return

    cdef int doc_block_size = data.blocksize[0]
    cdef int feature_block_size = data.blocksize[1]

    cdef int doc_block_count = doc_count // doc_block_size
    cdef int feature_block_count = feature_count // feature_block_size

    cdef TString factor_string_buf
    cdef int doc_block_idx
    cdef int doc_in_block_idx
    cdef int doc_block_start_idx
    cdef int feature_block_idx
    cdef int feature_in_block_idx
    cdef int feature_block_start_idx
    cdef int indptr_begin
    cdef int indptr_end
    cdef bool_t is_float_value = (data.dtype == np.float32) or (data.dtype == np.float64)

    for doc_block_idx in xrange(doc_block_count):
        doc_block_start_idx = doc_block_idx * doc_block_size
        indptr_begin = data.indptr[doc_block_idx]
        indptr_end = data.indptr[doc_block_idx + 1]
        for indptr in range(indptr_begin, indptr_end, 1):
            feature_block_idx = data.indices[indptr]
            feature_block_start_idx = feature_block_idx * feature_block_size
            values_block = data.data[indptr]
            for doc_in_block_idx in xrange(doc_block_size):
                for feature_in_block_idx in xrange(feature_block_size):
                    _add_single_feature_value_from_scipy_sparse(
                        doc_block_start_idx + doc_in_block_idx,
                        main_data_feature_idx_to_dst_feature_idx[feature_block_start_idx + feature_in_block_idx],
                        values_block[doc_in_block_idx, feature_in_block_idx],
                        is_float_value,
                        is_cat_feature_mask,
                        & factor_string_buf,
                        builder_visitor
                    )

cdef _set_data_from_scipy_coo_sparse(
    data,
    row,
    col,
    TConstArrayRef[ui32] main_data_feature_idx_to_dst_feature_idx,
    TConstArrayRef[bool_t] is_cat_feature_mask,
    IRawObjectsOrderDataVisitor * builder_visitor
):
    cdef int nonzero_count = data.shape[0]

    cdef TString factor_string_buf
    cdef int nonzero_idx
    cdef int doc_idx
    cdef int feature_idx

    cdef bool_t is_float_value = (data.dtype == np.float32) or (data.dtype == np.float64)

    for nonzero_idx in xrange(nonzero_count):
        doc_idx = row[nonzero_idx]
        src_feature_idx = col[nonzero_idx]
        value = data[nonzero_idx]
        _add_single_feature_value_from_scipy_sparse(
            doc_idx,
            main_data_feature_idx_to_dst_feature_idx[src_feature_idx],
            value,
            is_float_value,
            is_cat_feature_mask,
            & factor_string_buf,
            builder_visitor
        )


@cython.boundscheck(False)
def _set_data_from_scipy_csr_sparse(
    numpy_num_dtype[:] data,
    numpy_indices_dtype[:] indices,
    numpy_indices_dtype[:] indptr,
    bool_t has_separate_embedding_features_data,
    Py_ObjectsOrderBuilderVisitor py_builder_visitor
):
    cdef int doc_count = indptr.shape[0] - 1
    cdef IRawObjectsOrderDataVisitor * builder_visitor = py_builder_visitor.builder_visitor

    if doc_count == 0:
        return

    cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(
        py_builder_visitor.features_layout,
        has_separate_embedding_features_data
    )
    cdef TConstArrayRef[ui32] main_data_feature_idx_to_dst_feature_idx_ref = <TConstArrayRef[ui32]>main_data_feature_idx_to_dst_feature_idx

    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(py_builder_visitor.features_layout, EFeatureType_Categorical)
    cdef TConstArrayRef[bool_t] is_cat_feature_ref = <TConstArrayRef[bool_t]>is_cat_feature_mask

    def cast_to_nparray(array, dtype):
        if isinstance(array, np.ndarray) and array.dtype == dtype and array.flags.c_contiguous:
            return array
        return np.ascontiguousarray(array, dtype=dtype)

    assert numpy_indices_dtype == np.int32_t, "Type of indices in CSR sparse arrays must be int32"
    cdef np.ndarray[np.int32_t, ndim=1] indptr_i32 = cast_to_nparray(indptr, np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] indices_i32 = cast_to_nparray(indices, np.int32)

    cdef TConstArrayRef[ui32] indptr_i32_ref = TConstArrayRef[ui32](<ui32*>&indptr_i32[0], len(indptr_i32))
    cdef TConstArrayRef[ui32] indices_i32_ref = TConstArrayRef[ui32](<ui32*>&indices_i32[0], len(indices_i32))

    cdef np.ndarray[numpy_num_dtype, ndim=1] data_np

    if numpy_num_dtype == np.float32_t:
        data_np = cast_to_nparray(data, np.float32)
        return SetDataFromScipyCsrSparse[np.float32_t](
            indptr_i32_ref,
            TConstArrayRef[np.float32_t](<np.float32_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.float64_t:
        data_np = cast_to_nparray(data, np.float64)
        return SetDataFromScipyCsrSparse[np.float64_t](
            indptr_i32_ref,
            TConstArrayRef[np.float64_t](<np.float64_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.int8_t:
        data_np = cast_to_nparray(data, np.int8)
        return SetDataFromScipyCsrSparse[np.int8_t](
            indptr_i32_ref,
            TConstArrayRef[np.int8_t](<np.int8_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.uint8_t:
        data_np = cast_to_nparray(data, np.uint8)
        return SetDataFromScipyCsrSparse[np.uint8_t](
            indptr_i32_ref,
            TConstArrayRef[np.uint8_t](<np.uint8_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.int16_t:
        data_np = cast_to_nparray(data, np.int16)
        return SetDataFromScipyCsrSparse[np.int16_t](
            indptr_i32_ref,
            TConstArrayRef[np.int16_t](<np.int16_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.uint16_t:
        data_np = cast_to_nparray(data, np.uint16)
        return SetDataFromScipyCsrSparse[np.uint16_t](
            indptr_i32_ref,
            TConstArrayRef[np.uint16_t](<np.uint16_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.int32_t:
        data_np = cast_to_nparray(data, np.int32)
        return SetDataFromScipyCsrSparse[np.int32_t](
            indptr_i32_ref,
            TConstArrayRef[np.int32_t](<np.int32_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.uint32_t:
        data_np = cast_to_nparray(data, np.uint32)
        return SetDataFromScipyCsrSparse[np.uint32_t](
            indptr_i32_ref,
            TConstArrayRef[np.uint32_t](<np.uint32_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.int64_t:
        data_np = cast_to_nparray(data, np.int64)
        return SetDataFromScipyCsrSparse[np.int64_t](
            indptr_i32_ref,
            TConstArrayRef[np.int64_t](<np.int64_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())

    elif numpy_num_dtype == np.uint64_t:
        data_np = cast_to_nparray(data, np.uint64)
        return SetDataFromScipyCsrSparse[np.uint64_t](
            indptr_i32_ref,
            TConstArrayRef[np.uint64_t](<np.uint64_t*>&data_np[0], len(data_np)),
            indices_i32_ref,
            has_separate_embedding_features_data,
            main_data_feature_idx_to_dst_feature_idx_ref,
            is_cat_feature_ref,
            builder_visitor,
            <ILocalExecutor*>py_builder_visitor.local_executor.Get())
    else:
        assert False, "CSR sparse arrays support only numeric data types"


cdef _set_data_from_scipy_lil_sparse(
    data,
    TConstArrayRef[ui32] main_data_feature_idx_to_dst_feature_idx,
    TConstArrayRef[bool_t] is_cat_feature_mask,
    IRawObjectsOrderDataVisitor * builder_visitor
):
    data_shape = np.shape(data)
    cdef int doc_count = data_shape[0]

    if doc_count == 0:
        return

    cdef TString factor_string_buf
    cdef int doc_idx
    cdef int feature_idx
    cdef int nonzero_column_idx
    cdef int row_indices_count

    cdef bool_t is_float_value = (data.dtype == np.float32) or (data.dtype == np.float64)

    for doc_idx in xrange(doc_count):
        row_indices = data.rows[doc_idx]
        row_data = data.data[doc_idx]
        row_indices_count = len(row_indices)
        for nonzero_column_idx in xrange(row_indices_count):
            feature_idx = main_data_feature_idx_to_dst_feature_idx[row_indices[nonzero_column_idx]]
            value = row_data[nonzero_column_idx]
            _add_single_feature_value_from_scipy_sparse(
                doc_idx,
                feature_idx,
                value,
                is_float_value,
                is_cat_feature_mask,
                & factor_string_buf,
                builder_visitor
            )

cdef _set_objects_order_data_scipy_sparse_matrix(
    data,
    bool_t has_separate_embedding_features_data,
    const TFeaturesLayout * features_layout,
    Py_ObjectsOrderBuilderVisitor py_builder_visitor
):
    cdef IRawObjectsOrderDataVisitor * builder_visitor = py_builder_visitor.builder_visitor
    _set_cat_features_default_values_for_scipy_sparse(features_layout, builder_visitor)

    cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(features_layout, has_separate_embedding_features_data)
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
    cdef TVector[bool_t] is_text_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)
    cdef TVector[bool_t] is_embedding_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Embedding)
    if np.any(is_text_feature_mask):
        raise CatBoostError('Text features reading is not supported in sparse matrix format')
    if (not has_separate_embedding_features_data) and np.any(is_embedding_feature_mask):
        raise CatBoostError('Embedding features reading is not supported in sparse matrix format')

    if isinstance(data, scipy.sparse.bsr_matrix):
        _set_data_from_scipy_bsr_sparse(
            data,
            <TConstArrayRef[ui32]>main_data_feature_idx_to_dst_feature_idx,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )
    elif isinstance(data, scipy.sparse.coo_matrix):
        _set_data_from_scipy_coo_sparse(
            data.data,
            data.row,
            data.col,
            <TConstArrayRef[ui32]>main_data_feature_idx_to_dst_feature_idx,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )
    elif isinstance(data, scipy.sparse.csr_matrix):
        if not data.has_sorted_indices:
            data = data.sorted_indices()
        _set_data_from_scipy_csr_sparse(
            data.data,
            data.indices,
            data.indptr,
            has_separate_embedding_features_data,
            py_builder_visitor
        )
    elif isinstance(data, scipy.sparse.dok_matrix):
        coo_matrix = data.tocoo()
        _set_data_from_scipy_coo_sparse(
            coo_matrix.data,
            coo_matrix.row,
            coo_matrix.col,
            <TConstArrayRef[ui32]>main_data_feature_idx_to_dst_feature_idx,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )
    elif isinstance(data, scipy.sparse.lil_matrix):
        _set_data_from_scipy_lil_sparse(
            data,
            <TConstArrayRef[ui32]>main_data_feature_idx_to_dst_feature_idx,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )

# returns new data holders array
def _set_features_order_data_scipy_sparse_csc_matrix(
    ui32 doc_count,
    numpy_num_dtype [:] data,
    numpy_indices_dtype [:] indices,
    numpy_indices_dtype [:] indptr,
    bool_t has_sorted_indices,
    bool_t has_num_features,
    bool_t has_separate_embedding_features_data,
    Py_FeaturesOrderBuilderVisitor py_builder_visitor
):
    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    py_builder_visitor.get_raw_features_order_data_visitor(&builder_visitor)

    cdef const TFeaturesLayout* features_layout
    py_builder_visitor.get_features_layout(&features_layout)

    cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(features_layout, has_separate_embedding_features_data)
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)

    cdef np.float32_t float_default_value = 0.0
    cdef TString cat_default_value = "0"

    cdef ui32 src_feature_count = indptr.shape[0] - 1
    cdef ui32 src_feature_idx
    cdef ui32 dst_feature_idx
    cdef ui32 feature_nonzero_count
    cdef ui32 data_idx

    cdef TMaybeOwningConstArrayHolder[ui32] feature_indices_holder

    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TString factor_string_buf
    cdef TVector[TString] cat_feature_values

    cdef bool_t is_float_value = False

    cdef int indptr_begin
    cdef int indptr_end

    if (numpy_num_dtype is np.float32_t) or (numpy_num_dtype is np.float64_t):
        is_float_value = True

    new_data_holders = []

    for src_feature_idx in xrange(src_feature_count):
        feature_nonzero_count = indptr[src_feature_idx + 1] - indptr[src_feature_idx]
        new_data_holders += get_canonical_type_indexing_array(
            np.asarray(indices[indptr[src_feature_idx]:indptr[src_feature_idx + 1]]),
            &feature_indices_holder
        )
        dst_feature_idx = main_data_feature_idx_to_dst_feature_idx[src_feature_idx]

        if is_cat_feature_mask[dst_feature_idx]:
            cat_feature_values.clear()
            indptr_begin = indptr[src_feature_idx]
            indptr_end = indptr[src_feature_idx + 1]
            for data_idx in range(indptr_begin, indptr_end, 1):
                value = data[data_idx]
                _get_categorical_feature_value_from_scipy_sparse(
                    indices[data_idx],
                    dst_feature_idx,
                    value,
                    is_float_value,
                    &factor_string_buf
                )

                cat_feature_values.push_back(factor_string_buf)

            builder_visitor[0].AddCatFeature(
                dst_feature_idx,
                MakeConstPolymorphicValuesSparseArrayWithArrayIndex[TString, TString, ui32](
                    doc_count,
                    feature_indices_holder,
                    TMaybeOwningConstArrayHolder[TString].CreateNonOwning(
                        <TConstArrayRef[TString]>cat_feature_values
                    ),
                    has_sorted_indices,
                    cat_default_value
                )
            )
        else:
            new_data_holders += create_num_factor_data(
                dst_feature_idx,
                np.asarray(data[indptr[src_feature_idx]:indptr[src_feature_idx + 1]]),
                &num_factor_data
            )
            builder_visitor[0].AddFloatFeature(
                dst_feature_idx,
                MakeConstPolymorphicValuesSparseArrayWithArrayIndexGeneric[float, ui32](
                    doc_count,
                    feature_indices_holder,
                    num_factor_data,
                    has_sorted_indices,
                    float_default_value
                )
            )

    return new_data_holders

# returns new data holders array
cdef _set_features_order_data_scipy_sparse_matrix(
    data,
    bool_t has_separate_embedding_features_data,
    const TFeaturesLayout* features_layout,
    Py_FeaturesOrderBuilderVisitor py_builder_visitor
):
    new_data_holders = []

    if isinstance(data, scipy.sparse.csc_matrix):
        new_data_holders = _set_features_order_data_scipy_sparse_csc_matrix(
            data.shape[0],
            data.data,
            data.indices,
            data.indptr,
            data.has_sorted_indices,
            features_layout[0].GetFloatFeatureCount() != 0,
            has_separate_embedding_features_data,
            py_builder_visitor
        )

    return new_data_holders

# returns new data holders array
cdef _set_data_from_generic_matrix(
    data,
    bool_t has_separate_embedding_features_data,
    const TFeaturesLayout* features_layout,
    IRawObjectsOrderDataVisitor* builder_visitor
):
    data_shape = np.shape(data)
    cdef int doc_count = data_shape[0]
    cdef int src_feature_count = data_shape[1]

    if doc_count == 0:
        return []

    cdef TString factor_strbuf
    cdef TEmbeddingData object_embedding_data
    cdef int doc_idx
    cdef int src_feature_idx
    cdef ui32 dst_feature_idx
    cdef int cat_feature_idx

    cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(features_layout, has_separate_embedding_features_data)
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
    cdef TVector[bool_t] is_text_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)
    cdef TVector[bool_t] is_embedding_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Embedding)
    cdef TVector[ui32] embedding_dimensions

    # TODO(akhropov): make yresize accessible in Cython
    embedding_dimensions.resize(src_feature_count)

    new_data_holders = []

    for doc_idx in xrange(doc_count):
        doc_data = data[doc_idx]
        for src_feature_idx in xrange(src_feature_count):
            factor = doc_data[src_feature_idx]
            dst_feature_idx = main_data_feature_idx_to_dst_feature_idx[src_feature_idx]
            if is_cat_feature_mask[dst_feature_idx]:
                get_cat_factor_bytes_representation(
                    doc_idx,
                    dst_feature_idx,
                    factor,
                    &factor_strbuf
                )
                builder_visitor[0].AddCatFeature(doc_idx, dst_feature_idx, <TStringBuf>factor_strbuf)
            elif is_text_feature_mask[dst_feature_idx]:
                get_text_factor_bytes_representation(
                    doc_idx,
                    dst_feature_idx,
                    factor,
                    &factor_strbuf
                )
                builder_visitor[0].AddTextFeature(doc_idx, dst_feature_idx, <TStringBuf>factor_strbuf)
            elif is_embedding_feature_mask[dst_feature_idx]:
                if doc_idx == 0:
                    embedding_dimensions[src_feature_idx] = len(factor)
                new_data_holders += get_embedding_array_data(
                    doc_idx,
                    dst_feature_idx,
                    embedding_dimensions[src_feature_idx],
                    factor,
                    &object_embedding_data
                )
                builder_visitor[0].AddEmbeddingFeature(doc_idx, dst_feature_idx, object_embedding_data)
            else:
                builder_visitor[0].AddFloatFeature(
                    doc_idx,
                    dst_feature_idx,
                    get_float_feature(doc_idx, dst_feature_idx, factor)
                )

    return new_data_holders


# returns new data holders array
cdef _set_data(data, embedding_features_data, feature_names, const TFeaturesLayout* features_layout, Py_ObjectsOrderBuilderVisitor py_builder_visitor):
    new_data_holders = []

    if isinstance(data, FeaturesData):
        _set_data_np(data.num_feature_data, data.cat_feature_data, embedding_features_data is not None, features_layout, py_builder_visitor.builder_visitor)
    elif isinstance(data, np.ndarray) and data.dtype == np.float32:
        _set_data_np(data, None, embedding_features_data is not None, features_layout, py_builder_visitor.builder_visitor)
    elif isinstance(data, SPARSE_MATRIX_TYPES):
        _set_objects_order_data_scipy_sparse_matrix(data, embedding_features_data is not None, features_layout, py_builder_visitor)
    else:
        new_data_holders = _set_data_from_generic_matrix(
            data,
            embedding_features_data is not None,
            features_layout,
            py_builder_visitor.builder_visitor
        )

    if embedding_features_data is not None:
        new_data_holders += _set_objects_order_embedding_features_data(embedding_features_data, feature_names, features_layout, py_builder_visitor.builder_visitor)

    return new_data_holders


cdef TString obj_to_arcadia_string(obj) except *:
    INT64_MIN = -9223372036854775808
    INT64_MAX =  9223372036854775807
    cdef type obj_type = type(obj)

    if obj_type is float or obj_type is _npfloat32 or obj_type is _npfloat64 or obj_type is _npfloat16:
        return ToString[double](<double>obj)
    elif ((obj_type is int or obj_type is long) and (INT64_MIN <= obj <= INT64_MAX)) or is_np_int_type(obj_type):
        return ToString[i64](<i64>obj)
    elif is_np_uint_type(obj_type):
        return ToString[ui64](<ui64>obj)
    elif obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npbytes_ or obj_type is _npunicode_:
        return to_arcadia_string(obj)
    else:
        return to_arcadia_string(str(obj))


ctypedef fused IBuilderVisitor:
    IRawObjectsOrderDataVisitor
    IRawFeaturesOrderDataVisitor


cdef TVector[TPair] _make_pairs_vector(pairs, pairs_weight=None) except *:
    if pairs_weight:
        if len(pairs) != len(pairs_weight):
            raise CatBoostError(
                'len(pairs_weight) = {} is not equal to len(pairs) = {} '.format(
                    len(pairs_weight), len(pairs)
                )
            )

    cdef TVector[TPair] pairs_vector
    pairs_vector.resize(len(pairs))

    for pair_idx, pair in enumerate(pairs):
        pairs_vector[pair_idx].WinnerId = <ui32>pair[0]
        pairs_vector[pair_idx].LoserId = <ui32>pair[1]
        pairs_vector[pair_idx].Weight = <float>(pairs_weight[pair_idx] if pairs_weight else 1.0)
    return pairs_vector


cdef _set_pairs(pairs, pairs_weight, IBuilderVisitor* builder_visitor):
    cdef TVector[TPair] pairs_vector = _make_pairs_vector(pairs, pairs_weight)
    builder_visitor[0].SetPairs(TConstArrayRef[TPair](pairs_vector.data(), pairs_vector.size()))

cdef _set_weight(weight, IRawObjectsOrderDataVisitor* builder_visitor):
    cdef int i
    cdef int weights_len = len(weight)
    for i in xrange(weights_len):
        builder_visitor[0].AddWeight(i, float(weight[i]))

cdef _set_weight_features_order(weight, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef TVector[float] weightVector
    cdef int weights_len = len(weight)
    weightVector.reserve(weights_len)
    for i in xrange(weights_len):
        weightVector.push_back(float(weight[i]))
    builder_visitor[0].AddWeights(<TConstArrayRef[float]>weightVector)

cdef TGroupId _calc_group_id_for(i, py_group_ids) except *:
    cdef TString id_as_strbuf

    try:
        get_id_object_bytes_string_representation(py_group_ids[i], &id_as_strbuf)
    except CatBoostError:
        raise CatBoostError(
            "group_id[{}] object ({}) is unsuitable (should be string or integral type)".format(
                i, py_group_ids[i]
            )
        )
    return CalcGroupIdFor(<TStringBuf>id_as_strbuf)

cdef _set_group_id(group_id, IBuilderVisitor* builder_visitor):
    cdef int group_id_len = len(group_id)
    cdef int i
    for i in xrange(group_id_len):
        builder_visitor[0].AddGroupId(i, _calc_group_id_for(i, group_id))

cdef _set_group_weight(group_weight, IRawObjectsOrderDataVisitor* builder_visitor):
    cdef int group_weight_len = len(group_weight)
    cdef int i
    for i in xrange(group_weight_len):
        builder_visitor[0].AddGroupWeight(i, float(group_weight[i]))

cdef _set_group_weight_features_order(group_weight, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef TVector[float] groupWeightVector
    cdef int group_weight_len = len(group_weight)
    cdef int i
    groupWeightVector.reserve(group_weight_len)
    for i in xrange(group_weight_len):
        groupWeightVector.push_back(float(group_weight[i]))
    builder_visitor[0].AddGroupWeights(<TConstArrayRef[float]>groupWeightVector)

cdef TSubgroupId _calc_subgroup_id_for(i, py_subgroup_ids) except *:
    cdef TString id_as_strbuf

    try:
        get_id_object_bytes_string_representation(py_subgroup_ids[i], &id_as_strbuf)
    except CatBoostError:
        raise CatBoostError(
            "subgroup_id[{}] object ({}) is unsuitable (should be string or integral type)".format(
                i, py_subgroup_ids[i]
            )
        )
    return CalcSubgroupIdFor(<TStringBuf>id_as_strbuf)

cdef _set_subgroup_id(subgroup_id, IBuilderVisitor* builder_visitor):
    cdef ui32 subgroup_id_len = len(subgroup_id)
    cdef int i
    for i in xrange(subgroup_id_len):
        builder_visitor[0].AddSubgroupId(i, _calc_subgroup_id_for(i, subgroup_id))

cdef _set_baseline(baseline, IRawObjectsOrderDataVisitor* builder_visitor):
    cdef ui32 baseline_len = len(baseline)
    cdef int i
    for i in range(baseline_len):
        for j, value in enumerate(baseline[i]):
            builder_visitor[0].AddBaseline(i, j, float(value))

cdef _set_baseline_features_order(baseline, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef ui32 baseline_count = len(baseline[0])
    cdef TVector[float] one_dim_baseline
    cdef ui32 baseline_idx
    for baseline_idx in xrange(baseline_count):
        one_dim_baseline.clear()
        one_dim_baseline.reserve(len(baseline))
        for i in range(len(baseline)):
            one_dim_baseline.push_back(float(baseline[i][baseline_idx]))
        builder_visitor[0].AddBaseline(baseline_idx, <TConstArrayRef[float]>one_dim_baseline)

cdef _set_timestamp(timestamp, IBuilderVisitor* builder_visitor):
    cdef int i
    cdef int timestamps_len = len(timestamp)
    for i in xrange(timestamps_len):
        builder_visitor[0].AddTimestamp(i, <ui64>timestamp[i])


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_label_from_num_nparray_objects_order(
    np.ndarray[numpy_num_dtype, ndim=2] label,
    Py_ObjectsOrderBuilderVisitor py_builder_visitor
):

    """
        older buffer interface is used instead of memory views because of
        https://github.com/cython/cython/issues/1772, https://github.com/cython/cython/issues/2485
    """

    cdef IRawObjectsOrderDataVisitor* builder_visitor = py_builder_visitor.builder_visitor
    cdef ui32 object_count = label.shape[0]
    cdef ui32 target_count = label.shape[1]
    cdef ui32 target_idx
    cdef ui32 object_idx

    for target_idx in range(target_count):
        for object_idx in range(object_count):
            builder_visitor[0].AddTarget(target_idx, object_idx, <float>label[object_idx][target_idx])

cdef ERawTargetType _py_target_type_to_raw_target_data(py_label_type) except *:
    if np.issubdtype(py_label_type, np.floating):
        return ERawTargetType_Float
    elif np.issubdtype(py_label_type, np.integer):
        return ERawTargetType_Integer
    else:
        return ERawTargetType_String


cdef class _PoolBase:
    cdef TDataProviderPtr __pool
    cdef object target_type

    # possibly hold list of references to data to allow using views to them in __pool
    # also useful to simplify get_label
    cdef object __target_data_holders # [target_idx]

    # possibly hold reference or list of references to data to allow using views to them in __pool
    cdef object __data_holders

    def __cinit__(self):
        self.__pool = TDataProviderPtr()
        self.target_type = None
        self.__target_data_holders = None
        self.__data_holders = None

    def __dealloc__(self):
        self.__pool.Drop()

    def __deepcopy__(self, _):
        raise CatBoostError('Can\'t deepcopy _PoolBase object')

    def __eq__(self, _PoolBase other):
        return dereference(self.__pool.Get()) == dereference(other.__pool.Get())

    def _set_label_objects_order(self, label, Py_ObjectsOrderBuilderVisitor py_builder_visitor):
        cdef IRawObjectsOrderDataVisitor* builder_visitor = py_builder_visitor.builder_visitor
        cdef ui32 object_count = len(label)
        cdef ui32 target_count = len(label[0])
        cdef ui32 target_idx
        cdef ui32 object_idx

        self.target_type = type(label[0][0])
        raw_target_type = _py_target_type_to_raw_target_data(self.target_type)
        if (raw_target_type == ERawTargetType_Integer) or (raw_target_type == ERawTargetType_Float):
            if isinstance(label, np.ndarray) and (self.target_type in numpy_num_dtype_list):
                _set_label_from_num_nparray_objects_order(label, py_builder_visitor)
            else:
                for target_idx in range(target_count):
                    for object_idx in range(object_count):
                        builder_visitor[0].AddTarget(
                            target_idx,
                            object_idx,
                            <float>(label[object_idx][target_idx])
                        )
        else:
            for target_idx in range(target_count):
                for object_idx in range(object_count):
                    builder_visitor[0].AddTarget(
                        target_idx,
                        object_idx,
                        obj_to_arcadia_string(label[object_idx][target_idx])
                    )

    cdef _set_label_features_order(self, label, IRawFeaturesOrderDataVisitor* builder_visitor):
        cdef Py_FloatSequencePtr py_num_target_data
        cdef ITypedSequencePtr[np.float32_t] num_target_data
        cdef np.ndarray target_array
        cdef TVector[TString] string_target_data
        cdef ui32 object_count = len(label)
        cdef ui32 target_count = len(label[0])
        cdef ui32 target_idx
        cdef ui32 object_idx

        self.target_type = type(label[0][0])
        raw_target_type = _py_target_type_to_raw_target_data(self.target_type)
        if (raw_target_type == ERawTargetType_Integer) or (raw_target_type == ERawTargetType_Float):
            self.__target_data_holders = []
            for target_idx in range(target_count):
                if isinstance(label, np.ndarray) and (self.target_type in numpy_num_dtype_list):
                    target_array = np.ascontiguousarray(label[:, target_idx])
                else:
                    target_array = np.empty(object_count, dtype=np.float32)
                    for object_idx in range(object_count):
                        target_array[object_idx] = label[object_idx][target_idx]

                self.__target_data_holders.append(target_array)
                py_num_target_data = make_non_owning_type_cast_array_holder(target_array)
                py_num_target_data.get_result(&num_target_data)
                builder_visitor[0].AddTarget(target_idx, num_target_data)
        else:
            string_target_data.reserve(object_count)
            for target_idx in range(target_count):
                string_target_data.clear()
                for object_idx in range(object_count):
                    string_target_data.push_back(obj_to_arcadia_string(label[object_idx][target_idx]))
                builder_visitor[0].AddTarget(target_idx, <TConstArrayRef[TString]>string_target_data)


    cpdef _read_pool(self, pool_file, cd_file, pairs_file, feature_names_file, delimiter, bool_t has_header, bool_t ignore_csv_quoting, int thread_count, dict quantization_params):
        cdef TPathWithScheme pool_file_path
        pool_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(pool_file)), TStringBuf(<char*>'dsv'))

        cdef TPathWithScheme pairs_file_path
        if pairs_file:
            pairs_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(pairs_file)), TStringBuf(<char*>'dsv-flat'))

        cdef TPathWithScheme feature_names_file_path
        if feature_names_file:
            feature_names_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(feature_names_file)), TStringBuf(<char*>'dsv'))

        cdef TColumnarPoolFormatParams columnarPoolFormatParams
        columnarPoolFormatParams.DsvFormat.HasHeader = has_header
        columnarPoolFormatParams.DsvFormat.Delimiter = ord(delimiter)
        columnarPoolFormatParams.DsvFormat.IgnoreCsvQuoting = ignore_csv_quoting
        if cd_file:
            columnarPoolFormatParams.CdFilePath = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(cd_file)), TStringBuf(<char*>'dsv'))

        thread_count = UpdateThreadCount(thread_count)

        cdef TVector[ui32] emptyIntVec
        cdef TPathWithScheme input_borders_file_path
        if quantization_params is not None:
            input_borders = quantization_params.pop("input_borders", None)
            block_size = quantization_params.pop("dev_block_size", None)
            prep_params = _PreprocessParams(quantization_params)
            if input_borders:
                input_borders_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(input_borders)), TStringBuf(<char*>'dsv'))
            self.__pool = ReadAndQuantizeDataset(
                pool_file_path,
                pairs_file_path,
                TPathWithScheme(),
                TPathWithScheme(),
                TPathWithScheme(),
                feature_names_file_path,
                TPathWithScheme(),
                input_borders_file_path,
                columnarPoolFormatParams,
                emptyIntVec,
                EObjectsOrder_Undefined,
                prep_params.tree,
                block_size,
                TQuantizedFeaturesInfoPtr(),
                thread_count,
                False
            )
        else:
            self.__pool = ReadDataset(
                TMaybe[ETaskType](),
                pool_file_path,
                pairs_file_path,
                TPathWithScheme(),
                TPathWithScheme(),
                TPathWithScheme(),
                feature_names_file_path,
                TPathWithScheme(),
                columnarPoolFormatParams,
                emptyIntVec,
                EObjectsOrder_Undefined,
                thread_count,
                False,
                False
            )
        self.__data_holders = None # free previously used resources
        self.target_type = str


    cdef _init_features_order_layout_pool(
        self,
        data,
        embedding_features_data,
        feature_names,
        const TDataMetaInfo& data_meta_info,
        label,
        pairs,
        weight,
        group_id,
        group_weight,
        subgroup_id,
        pairs_weight,
        baseline,
        timestamp,
        thread_count):

        cdef TFeaturesLayout* features_layout = data_meta_info.FeaturesLayout.Get()
        cdef Py_FeaturesOrderBuilderVisitor py_builder_visitor = Py_FeaturesOrderBuilderVisitor(thread_count)
        cdef IRawFeaturesOrderDataVisitor* builder_visitor = py_builder_visitor.builder_visitor
        py_builder_visitor.set_features_layout(data_meta_info.FeaturesLayout.Get())

        cdef TVector[ui32] main_data_feature_idx_to_dst_feature_idx
        cdef TVector[bool_t] cat_features_mask # used only if data is np.ndarray

        cdef TVector[TIntrusivePtr[IResourceHolder]] resource_holders
        builder_visitor[0].Start(
            data_meta_info,
            _get_object_count(data),
            EObjectsOrder_Undefined,
            resource_holders
        )

        new_data_holders = None
        if isinstance(data, FeaturesData):
            new_data_holders = data

            _set_features_order_data_features_data(
                data.num_feature_data,
                data.cat_feature_data,
                py_builder_visitor)

            # prevent inadvent modification of pool data
            if data.num_feature_data is not None:
                data.num_feature_data.setflags(write=0)
            if data.cat_feature_data is not None:
                data.cat_feature_data.setflags(write=0)
        elif isinstance(data, pd.DataFrame):
            new_data_holders = _set_features_order_data_pd_data_frame(
                data,
                embedding_features_data is not None,
                features_layout,
                builder_visitor
            )
        elif isinstance(data, scipy.sparse.spmatrix):
            new_data_holders = _set_features_order_data_scipy_sparse_matrix(
                data,
                embedding_features_data is not None,
                features_layout,
                py_builder_visitor
            )
        elif isinstance(data, np.ndarray):
            if (data_meta_info.FeaturesLayout.Get()[0].GetFloatFeatureCount() or
                (data_meta_info.FeaturesLayout.Get()[0].GetEmbeddingFeatureCount() and (embedding_features_data is None))):
                new_data_holders = data

            main_data_feature_idx_to_dst_feature_idx = _get_main_data_feature_idx_to_dst_feature_idx(
                features_layout,
                embedding_features_data is not None
            )

            cat_features_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
            text_features_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)
            embedding_features_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Embedding)

            _set_features_order_data_ndarray(
                data,
                <ui32[:main_data_feature_idx_to_dst_feature_idx.size()]>main_data_feature_idx_to_dst_feature_idx.data(),
                <bool_t[:features_layout[0].GetExternalFeatureCount()]>cat_features_mask.data(),
                <bool_t[:features_layout[0].GetExternalFeatureCount()]>text_features_mask.data(),
                <bool_t[:features_layout[0].GetExternalFeatureCount()]>embedding_features_mask.data(),
                py_builder_visitor
            )

            # prevent inadvent modification of pool data
            data.setflags(write=0)
        else:
            raise CatBoostError(
                '[Internal error] wrong data type for _init_features_order_layout_pool: ' + type(data)
            )

        if embedding_features_data is not None:
            embedding_data_holders = _set_features_order_embedding_features_data(embedding_features_data, feature_names, features_layout, builder_visitor)
            if new_data_holders is None:
                new_data_holders = embedding_data_holders
            else:
                new_data_holders = [new_data_holders, embedding_data_holders]

        if label is not None:
            self._set_label_features_order(label, builder_visitor)
        if pairs is not None:
            _set_pairs(pairs, pairs_weight, builder_visitor)
        elif pairs_weight is not None:
            raise CatBoostError('"pairs_weight" is specified but "pairs" is not')
        if baseline is not None:
            _set_baseline_features_order(baseline, builder_visitor)
        if weight is not None:
            _set_weight_features_order(weight, builder_visitor)
        if group_id is not None:
            _set_group_id(group_id, builder_visitor)
        if group_weight is not None:
            _set_group_weight_features_order(group_weight, builder_visitor)
        if subgroup_id is not None:
            _set_subgroup_id(subgroup_id, builder_visitor)
        if timestamp is not None:
            _set_timestamp(timestamp, builder_visitor)

        builder_visitor[0].Finish()

        self.__pool = py_builder_visitor.data_provider_builder.Get()[0].GetResult()
        self.__data_holders = new_data_holders


    cdef _init_objects_order_layout_pool(
        self,
        data,
        embedding_features_data,
        feature_names,
        const TDataMetaInfo& data_meta_info,
        label,
        pairs,
        weight,
        group_id,
        group_weight,
        subgroup_id,
        pairs_weight,
        baseline,
        timestamp,
        thread_count):

        cdef Py_ObjectsOrderBuilderVisitor py_builder_visitor = Py_ObjectsOrderBuilderVisitor(thread_count)
        cdef IRawObjectsOrderDataVisitor* builder_visitor = py_builder_visitor.builder_visitor
        py_builder_visitor.set_features_layout(data_meta_info.FeaturesLayout.Get())

        cdef TVector[TIntrusivePtr[IResourceHolder]] resource_holders
        builder_visitor[0].Start(
            False,
            data_meta_info,
            False,
            _get_object_count(data),
            EObjectsOrder_Undefined,
            resource_holders
        )
        builder_visitor[0].StartNextBlock(_get_object_count(data))

        new_data_holders = _set_data(data, embedding_features_data, feature_names, data_meta_info.FeaturesLayout.Get(), py_builder_visitor)

        if label is not None:
            self._set_label_objects_order(label, py_builder_visitor)
        if pairs is not None:
            _set_pairs(pairs, pairs_weight, builder_visitor)
        elif pairs_weight is not None:
            raise CatBoostError('"pairs_weight" is specified but "pairs" is not')
        if baseline is not None:
            _set_baseline(baseline, builder_visitor)
        if weight is not None:
            _set_weight(weight, builder_visitor)
        if group_id is not None:
            _set_group_id(group_id, builder_visitor)
        if group_weight is not None:
            _set_group_weight(group_weight, builder_visitor)
        if subgroup_id is not None:
            _set_subgroup_id(subgroup_id, builder_visitor)
        if timestamp is not None:
            _set_timestamp(timestamp, builder_visitor)

        builder_visitor[0].Finish()

        self.__pool = py_builder_visitor.data_provider_builder.Get()[0].GetResult()
        self.__data_holders = new_data_holders


    cpdef _init_pool(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, weight,
                     group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags,
                     thread_count):
        if group_weight is not None and weight is not None:
            raise CatBoostError('Pool must have either weight or group_weight.')

        thread_count = UpdateThreadCount(thread_count)

        cdef TDataMetaInfo data_meta_info
        if label is not None:
            data_meta_info.TargetCount = <ui32>len(label[0])
            if data_meta_info.TargetCount:
                data_meta_info.TargetType = _py_target_type_to_raw_target_data(type(label[0][0]))

        data_meta_info.BaselineCount = len(baseline[0]) if baseline is not None else 0
        data_meta_info.HasGroupId = group_id is not None
        data_meta_info.HasGroupWeight = group_weight is not None
        data_meta_info.HasSubgroupIds = subgroup_id is not None
        data_meta_info.HasWeights = weight is not None
        data_meta_info.HasTimestamp = timestamp is not None
        data_meta_info.HasPairs = pairs is not None

        data_meta_info.FeaturesLayout = _init_features_layout(
            data,
            embedding_features_data,
            cat_features,
            text_features,
            embedding_features,
            feature_names,
            feature_tags
        )

        do_use_raw_data_in_features_order = False
        if isinstance(data, FeaturesData):
            if ((data.num_feature_data is not None) and
                data.num_feature_data.flags.aligned and
                data.num_feature_data.flags.f_contiguous and
                (len(data.num_feature_data) != 0)
               ):
                do_use_raw_data_in_features_order = True
        elif isinstance(data, pd.DataFrame):
            do_use_raw_data_in_features_order = True
        elif isinstance(data, scipy.sparse.csc_matrix):
            do_use_raw_data_in_features_order = True
        else:
            if isinstance(data, np.ndarray) and (data.dtype in numpy_num_dtype_list):
                if data.flags.aligned and data.flags.f_contiguous and (len(data) != 0):
                    do_use_raw_data_in_features_order = True

        if do_use_raw_data_in_features_order:
            self._init_features_order_layout_pool(
                data,
                embedding_features_data,
                feature_names,
                data_meta_info,
                label,
                pairs,
                weight,
                group_id,
                group_weight,
                subgroup_id,
                pairs_weight,
                baseline,
                timestamp,
                thread_count
            )
        else:
            self._init_objects_order_layout_pool(
                data,
                embedding_features_data,
                feature_names,
                data_meta_info,
                label,
                pairs,
                weight,
                group_id,
                group_weight,
                subgroup_id,
                pairs_weight,
                baseline,
                timestamp,
                thread_count
            )

    cpdef _save(self, fname):
        cdef TString file_name = to_arcadia_string(fspath(fname))
        SaveQuantizedPool(self.__pool, file_name)


    cpdef _set_pairs(self, pairs):
        cdef TVector[TPair] pairs_vector = _make_pairs_vector(pairs)
        self.__pool.Get()[0].SetPairs(TConstArrayRef[TPair](pairs_vector.data(), pairs_vector.size()))

    cpdef _set_weight(self, weight):
        cdef TVector[float] weight_vector
        for value in weight:
            weight_vector.push_back(value)
        self.__pool.Get()[0].SetWeights(
            TConstArrayRef[float](weight_vector.data(), weight_vector.size())
        )

    cpdef _set_group_id(self, group_id):
        rows = self.num_row()
        cdef TVector[TGroupId] group_id_vector
        group_id_vector.reserve(rows)

        for i in range(rows):
            group_id_vector.push_back(_calc_group_id_for(i, group_id))

        self.__pool.Get()[0].SetGroupIds(
            TConstArrayRef[TGroupId](group_id_vector.data(), group_id_vector.size())
        )

    cpdef _set_group_weight(self, group_weight):
        cdef TVector[float] group_weight_vector
        for value in group_weight:
            group_weight_vector.push_back(value)
        self.__pool.Get()[0].SetGroupWeights(
            TConstArrayRef[float](group_weight_vector.data(), group_weight_vector.size())
        )

    cpdef _set_subgroup_id(self, subgroup_id):
        rows = self.num_row()
        cdef TVector[TSubgroupId] subgroup_id_vector
        subgroup_id_vector.reserve(rows)

        for i in range(rows):
            subgroup_id_vector.push_back(_calc_subgroup_id_for(i, subgroup_id))

        self.__pool.Get()[0].SetSubgroupIds(
            TConstArrayRef[TSubgroupId](subgroup_id_vector.data(), subgroup_id_vector.size())
        )

    cpdef _set_pairs_weight(self, pairs_weight):
        cdef TConstArrayRef[TPair] old_pairs = GetUngroupedPairs(self.__pool.Get()[0])
        cdef TVector[TPair] new_pairs
        for i in range(old_pairs.size()):
            new_pairs.push_back(TPair(old_pairs[i].WinnerId, old_pairs[i].LoserId, pairs_weight[i]))
        self.__pool.Get()[0].SetPairs(TConstArrayRef[TPair](new_pairs.data(), new_pairs.size()))

    cpdef _set_baseline(self, baseline):
        rows = self.num_row()
        approx_dimension = len(baseline[0])

        cdef TVector[TVector[float]] baseline_matrix # [approxIdx][objectIdx]
        cdef TVector[TConstArrayRef[float]] baseline_matrix_view # [approxIdx][objectIdx]
        baseline_matrix.resize(approx_dimension)
        baseline_matrix_view.resize(approx_dimension)
        for j in range(approx_dimension):
            baseline_matrix[j].resize(rows)
            baseline_matrix_view[j] = TConstArrayRef[float](
                baseline_matrix[j].data(),
                baseline_matrix[j].size()
            )

        for i in range(rows):
            for j, value in enumerate(baseline[i]):
                baseline_matrix[j][i] = float(value)

        self.__pool.Get()[0].SetBaseline(
            TBaselineArrayRef(baseline_matrix_view.data(), baseline_matrix_view.size())
        )

    cpdef _set_timestamp(self, timestamp):
        cdef TVector[ui64] timestamp_vector
        for value in timestamp:
            timestamp_vector.push_back(<ui64>value)
        self.__pool.Get()[0].SetTimestamps(
            TConstArrayRef[ui64](timestamp_vector.data(), timestamp_vector.size())
        )

    cpdef _set_feature_names(self, feature_names):
        cdef TVector[TString] feature_names_vector
        for value in feature_names:
            feature_names_vector.push_back(to_arcadia_string(str(value)))
        self.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()[0].SetExternalFeatureIds(
            TConstArrayRef[TString](feature_names_vector.data(), feature_names_vector.size())
        )

    cpdef _quantize(self, dict params):
        _input_borders = params.pop("input_borders", None)
        prep_params = _PreprocessParams(params)
        cdef TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
        cdef TQuantizedObjectsDataProviderPtr quantizedObjects

        if _input_borders:
            quantizedFeaturesInfo = _init_quantized_feature_info(self.__pool, _input_borders)

        with nogil:
            SetPythonInterruptHandler()
            try:
                quantizedObjects = ConstructQuantizedPoolFromRawPool(self.__pool, prep_params.tree, quantizedFeaturesInfo)
            finally:
                ResetPythonInterruptHandler()

        self.__pool.Get()[0].ObjectsData = quantizedObjects
        self.__pool.Get()[0].MetaInfo.FeaturesLayout = quantizedObjects.Get()[0].GetFeaturesLayout()

    cpdef get_feature_names(self):
        feature_names = []
        cdef bytes pystr
        cdef TConstArrayRef[TFeatureMetaInfo] features_meta_info = (
            self.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()[0].GetExternalFeaturesMetaInfo()
        )
        for meta_info in features_meta_info:
            pystr = meta_info.Name.c_str()
            feature_names.append(to_native_str(pystr))
        return feature_names

    cpdef num_row(self):
        """
        Get the number of rows in the Pool.

        Returns
        -------
        number of rows : int
        """
        return self.__pool.Get()[0].GetObjectCount()

    cpdef num_col(self):
        """
        Get the number of columns in the Pool.

        Returns
        -------
        number of cols : int
        """
        return self.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()[0].GetExternalFeatureCount()

    cpdef num_pairs(self):
        """
        Get the number of pairs in the Pool.

        Returns
        -------
        number of pairs : int
        """
        return GetNumPairs(self.__pool.Get()[0])

    @property
    def shape(self):
        """
        Get the shape of the Pool.

        Returns
        -------
        shape : (int, int)
            (rows, cols)
        """
        return tuple([self.num_row(), self.num_col()])


    cpdef is_quantized(self):
        cdef TQuantizedObjectsDataProvider* quantized_objects_data_provider = dynamic_cast_to_TQuantizedObjectsDataProvider(
            self.__pool.Get()[0].ObjectsData.Get()
        )
        if not quantized_objects_data_provider:
            return False
        return True

    cdef _get_feature(
        self,
        TRawObjectsDataProvider* raw_objects_data_provider,
        factor_idx,
        ILocalExecutor* local_executor,
        dst_data):

        cdef TMaybeData[const TFloatValuesHolder*] maybe_factor_data = raw_objects_data_provider[0].GetFloatFeature(factor_idx)
        cdef TMaybeOwningArrayHolder[float] factor_data

        if maybe_factor_data.Defined():
            factor_data = maybe_factor_data.GetRef()[0].ExtractValues(local_executor)
            for doc in range(self.num_row()):
                dst_data[doc, factor_idx] = factor_data[doc]
        else:
            for doc in range(self.num_row()):
                dst_data[doc, factor_idx] = np.float32(0)


    cpdef get_features(self):
        """
        Get feature matrix from Pool.

        Returns
        -------
        feature matrix : np.ndarray of shape (object_count, feature_count)
        """
        cdef int thread_count = UpdateThreadCount(-1)
        cdef TAtomicSharedPtr[TTbbLocalExecutor] local_executor = GetCachedLocalExecutor(thread_count)
        cdef TFeaturesLayout* features_layout =self.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()
        cdef TRawObjectsDataProvider* raw_objects_data_provider = dynamic_cast_to_TRawObjectsDataProvider(
            self.__pool.Get()[0].ObjectsData.Get()
        )
        if not raw_objects_data_provider:
            raise CatBoostError('Pool does not have raw features data, only quantized')
        if features_layout[0].GetExternalFeatureCount() != features_layout[0].GetFloatFeatureCount():
            raise CatBoostError('Pool has non-numeric features, get_features supports only numeric features')

        data = np.empty(self.shape, dtype=np.float32)

        for factor in range(self.num_col()):
            self._get_feature(raw_objects_data_provider, factor, <ILocalExecutor*>local_executor.Get(), data)

        return data


    cpdef has_label(self):
        """
        Returns
        -------
        True if Pool has label data
        """
        return self.__pool.Get()[0].MetaInfo.TargetCount > 0

    cpdef get_label(self):
        """
        Get labels from Pool.

        Returns
        -------
        labels : list if labels are one-dimensional, np.ndarray of shape (object_count, labels_count) otherwise
        """
        cdef ERawTargetType raw_target_type
        cdef TVector[TArrayRef[float]] num_target_references
        cdef TVector[TConstArrayRef[TString]] string_target_references
        cdef ui32 target_count = self.__pool.Get()[0].MetaInfo.TargetCount
        cdef ui32 object_count = self.__pool.Get()[0].GetObjectCount()
        cdef np.ndarray[np.float32_t, ndim=1] num_target_1d
        cdef np.ndarray[np.float32_t, ndim=2] num_target_2d
        cdef ui32 target_idx

        if self.__target_data_holders:
            if len(self.__target_data_holders) == 1:
                return self.__target_data_holders[0]
            else:
                return np.array(self.__target_data_holders).T
        else:
            raw_target_type = self.__pool.Get()[0].RawTargetData.GetTargetType()
            if (raw_target_type == ERawTargetType_Integer) or (raw_target_type == ERawTargetType_Float):
                num_target_references.resize(target_count)
                if target_count == 1:
                    num_target_1d = np.empty(object_count, dtype=np.float32)
                    num_target_references[0] = TArrayRef[float](&num_target_1d[0], object_count)
                    self.__pool.Get()[0].RawTargetData.GetNumericTarget(
                        <TArrayRef[TArrayRef[float]]>num_target_references
                    )
                    return num_target_1d.astype(self.target_type)
                else:
                    num_target_2d = np.empty((object_count, target_count), dtype=np.float32, order='F')
                    for target_idx in range(target_count):
                        num_target_references[target_idx] = TArrayRef[float](
                            &num_target_2d[0, target_idx],
                            object_count
                        )
                    self.__pool.Get()[0].RawTargetData.GetNumericTarget(
                        <TArrayRef[TArrayRef[float]]>num_target_references
                    )
                    return num_target_2d.astype(self.target_type)
            elif raw_target_type == ERawTargetType_String:
                string_target_references.resize(target_count)
                self.__pool.Get()[0].RawTargetData.GetStringTargetRef(&string_target_references)
                labels = [
                    [
                        self.target_type(to_native_str(target_string))
                        for target_string in target
                    ]
                    for target in string_target_references
                ]

                if target_count == 1:
                    return labels[0]
                else:
                    return np.array(labels).T
            else:
                return None

    cpdef get_cat_feature_indices(self):
        """
        Get cat_feature indices from Pool.

        Returns
        -------
        cat_feature_indices : list
        """
        cdef TFeaturesLayout* featuresLayout = dereference(self.__pool.Get()).MetaInfo.FeaturesLayout.Get()
        return [int(i) for i in featuresLayout[0].GetCatFeatureInternalIdxToExternalIdx()]

    cpdef get_text_feature_indices(self):
        """
        Get text_feature indices from Pool.

        Returns
        -------
        text_feature_indices : list
        """
        cdef TFeaturesLayout* featuresLayout = dereference(self.__pool.Get()).MetaInfo.FeaturesLayout.Get()
        return [int(i) for i in featuresLayout[0].GetTextFeatureInternalIdxToExternalIdx()]

    cpdef get_embedding_feature_indices(self):
        """
        Get embedding_feature indices from Pool.

        Returns
        -------
        embedding_feature_indices : list
        """
        cdef TFeaturesLayout* featuresLayout = dereference(self.__pool.Get()).MetaInfo.FeaturesLayout.Get()
        return [int(i) for i in featuresLayout[0].GetEmbeddingFeatureInternalIdxToExternalIdx()]

    cpdef get_weight(self):
        """
        Get weight for each instance.

        Returns
        -------
        weight : list
        """
        cdef const TWeights[float]* weights = &(self.__pool.Get()[0].RawTargetData.GetWeights())
        cdef TConstArrayRef[float] non_trivial_data
        if weights.IsTrivial():
            return [1.0]*weights.GetSize()
        else:
            non_trivial_data = weights.GetNonTrivialData()
            return [weight for weight in non_trivial_data]


    cpdef get_group_id_hash(self):
        """
        Get hashes generated from group_id.

        Returns
        -------
        group_id : np.array with dtype==np.uint64 if group_id was defined or None otherwise.
        """
        cdef TMaybeData[TConstArrayRef[TGroupId]] arr_group_ids = self.__pool.Get()[0].ObjectsData.Get()[0].GetGroupIds()
        cdef const TGroupId* groupIdsPtr
        if arr_group_ids.Defined():
            result_group_ids = np.empty(arr_group_ids.GetRef().size(), dtype=np.uint64)
            groupIdsPtr = arr_group_ids.GetRef().data()
            for i in xrange(arr_group_ids.GetRef().size()):
                result_group_ids[i] = groupIdsPtr[i]
            return result_group_ids
        return None


    cpdef get_baseline(self):
        """
        Get baseline from Pool.

        Returns
        -------
        baseline : np.ndarray of shape (object_count, baseline_count)
        """
        cdef TMaybeData[TBaselineArrayRef] maybe_baseline = self.__pool.Get()[0].RawTargetData.GetBaseline()
        cdef TBaselineArrayRef baseline
        if maybe_baseline.Defined():
            baseline = maybe_baseline.GetRef()
            result = np.empty((self.num_row(), baseline.size()), dtype=np.float32)
            for baseline_idx in range(baseline.size()):
                for object_idx in range(self.num_row()):
                    result[object_idx, baseline_idx] = baseline[baseline_idx][object_idx]
            return result
        else:
            return np.empty((self.num_row(), 0), dtype=np.float32)

    cpdef _take_slice(self, _PoolBase pool, row_indices):
        cdef TVector[ui32] rowIndices
        for index in row_indices:
            rowIndices.push_back(index)

        thread_count = UpdateThreadCount(-1)
        self.__pool = pool.__pool.Get()[0].GetSubset(
            GetGroupingSubsetFromObjectsSubset(
                pool.__pool.Get()[0].ObjectsGrouping,
                rowIndices,
                EObjectsOrder_Undefined
            ),
            TotalMemorySize(),
            thread_count
        )
        self.target_type = pool.target_type
        
    cpdef _train_eval_split(self, _PoolBase train_pool, _PoolBase eval_pool, has_time, is_classification, eval_fraction, save_eval_pool):
        cdef TTrainTestSplitParams split_params
        split_params.Shuffle = not has_time
        split_params.Stratified = is_classification
        
        if (eval_fraction <= 0.0) or (eval_fraction >= 1.0):
            raise CatBoostError("eval_fraction must be in (0,1) range") 
        
        split_params.TrainPart = 1.0 - eval_fraction
    
        TrainEvalSplit(
            self.__pool.Get()[0],
            &train_pool.__pool,
            &eval_pool.__pool,
            split_params,
            save_eval_pool,
            UpdateThreadCount(-1),
            TotalMemorySize()
        )
        train_pool.target_type = self.target_type
        train_pool.__data_holders = self.__data_holders
        if save_eval_pool:
            eval_pool.target_type = self.target_type
            eval_pool.__data_holders = self.__data_holders


    cpdef save_quantization_borders(self, output_file):
        """
        Save file with borders used in numeric features quantization.
        File format is described here: https://catboost.ai/docs/concepts/input-data_custom-borders.html

        Parameters
        ----------
        output_file : string or pathlib.Path
            Output file name.

        Examples
        --------
        >>> train.quantize()
        >>> train.save_quantization_borders("borders.dat")
        >>> test.quantize(input_borders="borders.dat")
        """
        cdef TQuantizedObjectsDataProvider* quantized_objects_data_provider = dynamic_cast_to_TQuantizedObjectsDataProvider(
            self.__pool.Get()[0].ObjectsData.Get()
        )
        if not quantized_objects_data_provider:
            raise CatBoostError("Pool is not quantized")

        cdef TString fname = to_arcadia_string(fspath(output_file))
        cdef TQuantizedFeaturesInfoPtr quantized_features_info = quantized_objects_data_provider[0].GetQuantizedFeaturesInfo()

        with nogil:
            SaveBordersAndNanModesToFileInMatrixnetFormat(fname, quantized_features_info.Get()[0])


    @property
    def is_empty_(self):
        """
        Check if Pool is empty (contains no objects).

        Returns
        -------
        is_empty_ : bool
        """
        return self.num_row() == 0


cpdef _have_equal_features(_PoolBase pool1, _PoolBase pool2, bool_t ignore_sparsity=False):
    """
        ignoreSparsity means don't take into account whether columns are marked as either sparse or dense
          - only compare values
    """
    return pool1.__pool.Get()[0].ObjectsData.Get()[0].EqualTo(
        pool2.__pool.Get()[0].ObjectsData.Get()[0],
        ignore_sparsity
    )


cdef pair[int, int] _check_and_get_interaction_indices(_PoolBase pool, interaction_indices):
    cdef pair[int, int] pair_of_features
    if not isinstance(interaction_indices, list):
        raise CatBoostError(
            "interaction_indices is not a list type")
    if len(interaction_indices) != 2:
        raise CatBoostError(
            "interaction_indices must contain two numbers or string")
    if isinstance(interaction_indices[0], str):
        feature_names = pool.get_feature_names()
        if not isinstance(interaction_indices[1], str):
            raise CatBoostError(
                "interaction_indices must have one type")
        for idx in range(0, len(feature_names)):
            if interaction_indices[0] == feature_names[idx]:
                pair_of_features.first = idx
            if interaction_indices[1] == feature_names[idx]:
                pair_of_features.second = idx
        return pair_of_features

    if not isinstance(interaction_indices[0], int) or not isinstance(interaction_indices[1], int):
        raise CatBoostError(
            "interaction_indices must have either string or int type")
    pair_of_features.first = interaction_indices[0]
    pair_of_features.second = interaction_indices[1]
    return pair_of_features


cdef TQuantizedFeaturesInfoPtr _init_quantized_feature_info(TDataProviderPtr pool, _input_borders) except *:
    cdef TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
    quantizedFeaturesInfo = new TQuantizedFeaturesInfo(
        dereference(dereference(pool.Get()).MetaInfo.FeaturesLayout.Get()),
        TConstArrayRef[ui32](),
        TBinarizationOptions()
    )
    input_borders_str = to_arcadia_string(fspath(_input_borders))
    with nogil:
        LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
            input_borders_str,
            quantizedFeaturesInfo.Get())
    return quantizedFeaturesInfo


cdef _get_model_class_labels(const TFullModel& model):
    cdef TVector[TJsonValue] jsonLabels = model.GetModelClassLabels()
    if jsonLabels.empty():
        return np.empty(0, object)

    cdef size_t classCount = jsonLabels.size()
    cdef EJsonValueType labelType = jsonLabels[0].GetType()
    cdef size_t classIdx

    if labelType == JSON_INTEGER:
        labels = np.empty(classCount, np.int64)
        for classIdx in range(classCount):
            labels[classIdx] = jsonLabels[classIdx].GetInteger()
    elif labelType == JSON_DOUBLE:
        labels = np.empty(classCount, np.float64)
        for classIdx in range(classCount):
            labels[classIdx] = jsonLabels[classIdx].GetDouble()
    elif labelType == JSON_STRING:
        labels = np.empty(classCount, object)
        for classIdx in range(classCount):
            labels[classIdx] = to_native_str(bytes(jsonLabels[classIdx].GetString()))
    else:
        raise CatBoostError('[Internal error] unexpected model class labels type')

    return labels


cdef _get_loss_function_name(const TFullModel& model):
    return to_native_str(model.GetLossFunctionName())


cdef class _CatBoost:
    cdef TFullModel* __model
    cdef TVector[TEvalResult*] __test_evals
    cdef TMetricsAndTimeLeftHistory __metrics_history
    cdef THolder[TLearnProgress] __cached_learn_progress
    cdef size_t __n_features_in
    cdef object model_blob

    def __cinit__(self):
        self.__model = new TFullModel()
        self.__n_features_in = 0

    def __dealloc__(self):
        del self.__model

        cdef int i
        for i in xrange(self.__test_evals.size()):
            del self.__test_evals[i]

    def __eq__(self, _CatBoost other):
        return dereference(self.__model) == dereference(other.__model)

    def __ne__(self, _CatBoost other):
        return dereference(self.__model) != dereference(other.__model)

    cpdef _reserve_test_evals(self, num_tests):
        self.__test_evals.resize(num_tests)
        for i in range(num_tests):
            if self.__test_evals[i] == NULL:
                self.__test_evals[i] = new TEvalResult()

    cpdef _clear_test_evals(self):
        for i in range(self.__test_evals.size()):
            dereference(self.__test_evals[i]).ClearRawValues()

    cpdef _train(self, _PoolBase train_pool, test_pools, dict params, allow_clear_pool, maybe_init_model):
        self.model_blob = None
        _input_borders = params.pop("input_borders", None)
        prep_params = _PreprocessParams(params)
        cdef int thread_count = params.get("thread_count", 1)
        cdef TDataProviders dataProviders
        dataProviders.Learn = train_pool.__pool
        cdef _PoolBase test_pool
        cdef TVector[ui32] ignored_features
        cdef TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
        cdef TString input_borders_str
        cdef _CatBoost init_model
        cdef TMaybe[TFullModel*] init_model_param
        cdef THolder[TLearnProgress]* init_learn_progress_param
        cdef THolder[TLearnProgress]* dst_learn_progress_param

        cdef size_t n_features_in = train_pool.__pool.Get().MetaInfo.GetFeatureCount()
        self.__n_features_in = max(self.__n_features_in, n_features_in)
        task_type = params.get('task_type', 'CPU')

        if isinstance(test_pools, list):
            if task_type == 'GPU' and len(test_pools) > 1:
                raise CatBoostError('Multiple eval sets are not supported on GPU')
            for test_pool in test_pools:
                dataProviders.Test.push_back(test_pool.__pool)
        else:
            test_pool = test_pools
            dataProviders.Test.push_back(test_pool.__pool)
        if maybe_init_model is not None:
            if not isinstance(maybe_init_model, _CatBoost):
                raise CatBoostError('init_model is not an instance of _CatBoost class')
            init_model = maybe_init_model
            init_model_param = init_model.__model
            init_learn_progress_param = &(init_model.__cached_learn_progress)
        else:
            init_learn_progress_param = <THolder[TLearnProgress]*>nullptr
        self._reserve_test_evals(dataProviders.Test.size())
        self._clear_test_evals()

        if (_input_borders):
            quantizedFeaturesInfo = _init_quantized_feature_info(dataProviders.Learn, _input_borders)
        if task_type == 'CPU':
            dst_learn_progress_param = &self.__cached_learn_progress
        else:
            dst_learn_progress_param = <THolder[TLearnProgress]*>nullptr

        with nogil:
            SetPythonInterruptHandler()
            try:
                TrainModel(
                    prep_params.tree,
                    quantizedFeaturesInfo,
                    prep_params.customObjectiveDescriptor,
                    prep_params.customMetricDescriptor,
                    prep_params.customCallbackDescriptor,
                    dataProviders,
                    init_model_param,
                    init_learn_progress_param,
                    TString(<const char*>""),
                    self.__model,
                    self.__test_evals,
                    &self.__metrics_history,
                    dst_learn_progress_param
                )
            finally:
                ResetPythonInterruptHandler()

    cpdef _set_test_evals(self, test_evals):
        cdef TVector[double] vector
        num_tests = len(test_evals)
        self._reserve_test_evals(num_tests)
        self._clear_test_evals()
        for test_no in range(num_tests):
            for row in test_evals[test_no]:
                for value in row:
                    vector.push_back(float(value))
                dereference(self.__test_evals[test_no]).GetRawValuesRef()[0].push_back(vector)
                vector.clear()

    cpdef _get_test_evals(self):
        test_evals = []
        num_tests = self.__test_evals.size()
        for test_no in range(num_tests):
            test_eval = []
            for i in range(self.__test_evals[test_no].GetRawValuesRef()[0].size()):
                test_eval.append([value for value in dereference(self.__test_evals[test_no]).GetRawValuesRef()[0][i]])
            test_evals.append(test_eval)
        return test_evals

    cpdef _get_metrics_evals(self):
        return _get_metrics_evals_pydict(self.__metrics_history)

    cpdef _get_best_score(self):
        if self.__metrics_history.LearnBestError.empty():
            return {}
        best_scores = {}
        best_scores["learn"] = {}
        for metric, best_error in self.__metrics_history.LearnBestError:
            best_scores["learn"][to_native_str(metric)] = best_error
        for testIdx in range(self.__metrics_history.TestBestError.size()):
            eval_set_name = "validation"
            if self.__metrics_history.TestBestError.size() > 1:
                eval_set_name += "_" + str(testIdx)
            best_scores[eval_set_name] = {}
            for metric, best_error in self.__metrics_history.TestBestError[testIdx]:
                best_scores[eval_set_name][to_native_str(metric)] = best_error
        return best_scores

    cpdef _get_best_iteration(self):
        if self.__metrics_history.BestIteration.Defined():
            return self.__metrics_history.BestIteration.GetRef()
        return None

    cpdef _has_leaf_weights_in_model(self):
        return not self.__model.ModelTrees.Get().GetModelTreeData().Get().GetLeafWeights().empty()

    cpdef _get_cat_feature_indices(self):
        cdef TConstArrayRef[TCatFeature] arrayView = self.__model.ModelTrees.Get().GetCatFeatures()
        return [feature.Position.FlatIndex for feature in arrayView]

    cpdef _get_text_feature_indices(self):
        cdef TConstArrayRef[TTextFeature] arrayView = self.__model.ModelTrees.Get().GetTextFeatures()
        return [feature.Position.FlatIndex for feature in arrayView]

    cpdef _get_embedding_feature_indices(self):
        # TODO(akhropov): embedding features support in model
        return []

    cpdef _get_float_feature_indices(self):
        cdef TConstArrayRef[TFloatFeature] arrayView = self.__model.ModelTrees.Get().GetFloatFeatures()
        return [feature.Position.FlatIndex for feature in arrayView]

    cpdef _get_borders(self):
        cdef TConstArrayRef[TFloatFeature] arrayView = self.__model.ModelTrees.Get().GetFloatFeatures()
        return dict([(feature.Position.FlatIndex, feature.Borders) for feature in arrayView])

    cpdef _base_predict(self, _PoolBase pool, str prediction_type, int ntree_start, int ntree_end, int thread_count, bool_t verbose, str task_type):
        cdef TVector[TVector[double]] pred
        cdef EPredictionType predictionType = string_to_prediction_type(prediction_type)
        cdef EFormulaEvaluatorType formulaEvaluatorType = EFormulaEvaluatorType_GPU if task_type == 'GPU' else EFormulaEvaluatorType_CPU
        thread_count = UpdateThreadCount(thread_count);
        dereference(self.__model).SetEvaluatorType(formulaEvaluatorType);
        with nogil:
            pred = ApplyModelMulti(
                dereference(self.__model),
                dereference(pool.__pool.Get()),
                verbose,
                predictionType,
                ntree_start,
                ntree_end,
                thread_count
            )

        return transform_predictions(pred, predictionType, thread_count, self.__model)

    cpdef _base_virtual_ensembles_predict(self, _PoolBase pool, str prediction_type, int ntree_end, int virtual_ensembles_count, int thread_count, bool_t verbose):
            cdef TVector[TVector[double]] pred
            cdef EPredictionType predictionType = string_to_prediction_type(prediction_type)
            thread_count = UpdateThreadCount(thread_count);
            with nogil:
                pred = ApplyUncertaintyPredictions(
                    dereference(self.__model),
                    dereference(pool.__pool.Get()),
                    verbose,
                    predictionType,
                    ntree_end,
                    virtual_ensembles_count,
                    thread_count
                )
            return np.transpose(_2d_vector_of_double_to_np_array(pred))

    cpdef _staged_predict_iterator(self, _PoolBase pool, str prediction_type, int ntree_start, int ntree_end, int eval_period, int thread_count, verbose):
        thread_count = UpdateThreadCount(thread_count);
        stagedPredictIterator = _StagedPredictIterator(prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)
        stagedPredictIterator._initialize_model_calcer(self.__model, pool)
        return stagedPredictIterator

    cpdef _base_calc_leaf_indexes(self, _PoolBase pool, int ntree_start, int ntree_end,
                                  int thread_count, bool_t verbose):
        cdef int tree_count = ntree_end - ntree_start
        cdef int object_count = pool.__pool.Get()[0].ObjectsData.Get()[0].GetObjectCount()
        cdef TVector[ui32] flat_leaf_indexes
        thread_count = UpdateThreadCount(thread_count);
        with nogil:
            flat_leaf_indexes = CalcLeafIndexesMulti(
                dereference(self.__model),
                pool.__pool.Get()[0].ObjectsData,
                verbose,
                ntree_start,
                ntree_end,
                thread_count
            )
        return _vector_of_uints_to_2d_np_array(flat_leaf_indexes, object_count, tree_count)

    cpdef _leaf_indexes_iterator(self, _PoolBase pool, int ntree_start, int ntree_end):
        leafIndexIterator = _LeafIndexIterator()
        leafIndexIterator._initialize(self.__model, pool, ntree_start, ntree_end)
        return leafIndexIterator

    cpdef _base_eval_metrics(self, _PoolBase pool, metric_descriptions, int ntree_start, int ntree_end, int eval_period, int thread_count, result_dir, tmp_dir):
        thread_count = UpdateThreadCount(thread_count);
        cdef TVector[TString] metricDescriptions
        for metric_description in metric_descriptions:
            metricDescriptions.push_back(to_arcadia_string(metric_description))

        cdef TVector[TVector[double]] metrics
        metrics = EvalMetrics(
            dereference(self.__model),
            pool.__pool.Get()[0],
            metricDescriptions,
            ntree_start,
            ntree_end,
            eval_period,
            thread_count,
            to_arcadia_string(fspath(result_dir)),
            to_arcadia_string(fspath(tmp_dir))
        )
        cdef TVector[TString] metric_names = GetMetricNames(dereference(self.__model), metricDescriptions)
        return metrics, [to_native_str(name) for name in metric_names]

    cpdef _get_loss_function_name(self):
        return _get_loss_function_name(dereference(self.__model))

    cpdef _calc_partial_dependence(self, _PoolBase pool, features, int thread_count):
        thread_count = UpdateThreadCount(thread_count);
        cdef TVector[double] fstr
        cdef TDataProviderPtr dataProviderPtr
        if pool:
            dataProviderPtr = pool.__pool

        fstr = GetPartialDependence(
            dereference(self.__model),
            features,
            dataProviderPtr,
            thread_count
        )
        return _vector_of_double_to_np_array(fstr)

    cpdef _calc_fstr(self, type_name, _PoolBase pool, _PoolBase reference_data, int thread_count, int verbose,
                     model_output_name, shap_mode_name, interaction_indices, shap_calc_type, int sage_n_samples,
                     int sage_batch_size, bool_t sage_detect_convergence):
        thread_count = UpdateThreadCount(thread_count);
        cdef TVector[TString] feature_ids = GetMaybeGeneratedModelFeatureIds(
            dereference(self.__model),
            pool.__pool if pool else TDataProviderPtr(),
        )
        native_feature_ids = [to_native_str(s) for s in feature_ids]

        cdef TVector[TVector[double]] fstr
        cdef TVector[TVector[TVector[double]]] fstr_multi
        cdef TDataProviderPtr dataProviderPtr
        if pool:
            dataProviderPtr = pool.__pool

        cdef TDataProviderPtr referenceDataProviderPtr
        cdef EFstrType fstr_type = string_to_fstr_type(type_name)
        cdef EPreCalcShapValues shap_mode = string_to_shap_mode(shap_mode_name)
        cdef EExplainableModelOutput model_output = string_to_model_output(model_output_name)
        cdef TMaybe[pair[int, int]] pair_of_features

        if shap_calc_type == 'Exact':
            assert dereference(self.__model).IsOblivious(), "'Exact' calculation type is supported only for symmetric trees."
        cdef ECalcTypeShapValues calc_type = string_to_calc_type(shap_calc_type)
        if reference_data:
            referenceDataProviderPtr = reference_data.__pool
            TryFromString[ECalcTypeShapValues](to_arcadia_string("Independent"), calc_type)

        if type_name == 'ShapValues' and dereference(self.__model).GetDimensionsCount() > 1:
            with nogil:
                fstr_multi = GetFeatureImportancesMulti(
                    fstr_type,
                    dereference(self.__model),
                    dataProviderPtr,
                    referenceDataProviderPtr,
                    thread_count,
                    shap_mode,
                    verbose,
                    calc_type,
                    model_output
                )
            return _3d_vector_of_double_to_np_array(fstr_multi), native_feature_ids
        elif type_name == 'ShapInteractionValues':
            # TODO: Ensure sensible results of non-'Regular' calculation types for ShapInteractionValues
            assert shap_calc_type == "Regular", "Only 'Regular' calculation type is supported for ShapInteractionValues"
            if interaction_indices is not None:
                pair_of_features = _check_and_get_interaction_indices(pool, interaction_indices)
            with nogil:
                fstr_4d = CalcShapFeatureInteractionMulti(
                    fstr_type,
                    dereference(self.__model),
                    dataProviderPtr,
                    pair_of_features,
                    thread_count,
                    shap_mode,
                    verbose,
                    calc_type
                )
            if dereference(self.__model).GetDimensionsCount() > 1:
                return _reorder_axes_for_python_4d_shap_values(fstr_4d), native_feature_ids
            else:
                return _reorder_axes_for_python_3d_shap_values(fstr_4d), native_feature_ids
        else:
            with nogil:
                fstr = GetFeatureImportances(
                    fstr_type,
                    dereference(self.__model),
                    dataProviderPtr,
                    referenceDataProviderPtr,
                    thread_count,
                    shap_mode,
                    verbose,
                    calc_type,
                    model_output,
                    sage_n_samples,
                    sage_batch_size,
                    sage_detect_convergence
                )
            return _2d_vector_of_double_to_np_array(fstr), native_feature_ids

    cpdef _calc_ostr(self, _PoolBase train_pool, _PoolBase test_pool, int top_size, ostr_type, update_method, importance_values_sign, int thread_count, int verbose):
        thread_count = UpdateThreadCount(thread_count);
        cdef TDStrResult ostr = GetDocumentImportances(
            dereference(self.__model),
            train_pool.__pool.Get()[0],
            test_pool.__pool.Get()[0],
            to_arcadia_string(ostr_type),
            top_size,
            to_arcadia_string(update_method),
            to_arcadia_string(importance_values_sign),
            thread_count,
            verbose
        )
        indices = [[int(value) for value in ostr.Indices[i]] for i in range(ostr.Indices.size())]
        scores = [[float(value) for value in ostr.Scores[i]] for i in range(ostr.Scores.size())]
        if to_arcadia_string(ostr_type) == to_arcadia_string('Average'):
            indices = indices[0]
            scores = scores[0]
        return indices, scores

    cpdef _base_shrink(self, int ntree_start, int ntree_end):
        self.__model.Truncate(ntree_start, ntree_end)

    cpdef _get_scale_and_bias(self):
        cdef TScaleAndBias scale_and_bias = dereference(self.__model).GetScaleAndBias()
        bias = scale_and_bias.GetBiasRef()
        if len(bias) == 0:
            bias = 0
        elif len(bias) == 1:
            bias = bias[0]
        return scale_and_bias.Scale, bias

    cpdef _set_scale_and_bias(self, scale, list bias):
        cdef TScaleAndBias scale_and_bias = TScaleAndBias(scale, bias)
        dereference(self.__model).SetScaleAndBias(scale_and_bias)

    cpdef _is_oblivious(self):
        return self.__model.IsOblivious()

    cpdef _base_drop_unused_features(self):
        self.__model.ModelTrees.GetMutable().DropUnusedFeatures()

    cpdef _load_from_stream(self, stream) except +ProcessException:
        cdef THolder[TPythonStreamWrapper] wrapper = MakeHolder[TPythonStreamWrapper](python_stream_read_func, <PyObject*>stream)
        cdef TFullModel tmp_model
        tmp_model.Load(wrapper.Get())
        self.model_blob = None
        self.__model.Swap(tmp_model)

    cpdef _load_model(self, model_file, format):
        cdef TFullModel tmp_model
        cdef EModelType modelType = string_to_model_type(format)
        tmp_model = ReadModel(to_arcadia_string(fspath(model_file)), modelType)
        self.model_blob = None
        self.__model.Swap(tmp_model)

    cpdef _save_model(self, output_file, format, export_parameters, _PoolBase pool):
        cdef EModelType modelType = string_to_model_type(format)

        cdef TVector[TString] feature_id
        if pool:
            self._check_model_and_dataset_compatibility(pool)
            feature_id = pool.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()[0].GetExternalFeatureIds()

        cdef THashMap[ui32, TString] cat_features_hash_to_string
        if pool:
            cat_features_hash_to_string = MergeCatFeaturesHashToString(pool.__pool.Get()[0].ObjectsData.Get()[0])

        ExportModel(
            dereference(self.__model),
            to_arcadia_string(fspath(output_file)),
            modelType,
            to_arcadia_string(export_parameters),
            False,
            &feature_id if pool else <TVector[TString]*>nullptr,
            &cat_features_hash_to_string if pool else <THashMap[ui32, TString]*>nullptr
        )

    cpdef _serialize_model(self):
        cdef TString tstr = SerializeModel(dereference(self.__model))
        cdef const char* c_serialized_model_string = tstr.c_str()
        cdef bytes py_serialized_model_str = c_serialized_model_string[:tstr.size()]
        return py_serialized_model_str

    cpdef _deserialize_model(self, serialized_model_str):
        self.model_blob = serialized_model_str
        cdef TFullModel tmp_model = ReadZeroCopyModel(<char*>serialized_model_str, len(serialized_model_str))
        self.__model.Swap(tmp_model)

    cpdef _get_params(self):
        try:
            params_json = to_native_str(self.__model.ModelInfo["params"])
            params_dict = loads(params_json)
            flat_params = params_dict["flat_params"]
            params = {str(key): value for key, value in iteritems(flat_params)}
            return params
        except Exception as e:
            return {}

    cpdef _get_plain_params(self):
        return loads(to_native_str(WriteTJsonValue(GetPlainJsonWithAllOptions(dereference(self.__model)))))

    def _get_tree_count(self):
        return self.__model.GetTreeCount()

    def _get_random_seed(self):
        if not self.__model.ModelInfo.contains("params"):
            return 0
        cdef const char* c_params_json = self.__model.ModelInfo["params"].c_str()
        cdef bytes py_params_json = c_params_json
        params_json = to_native_str(py_params_json)
        if params_json:
            return loads(params_json).get('random_seed', 0)
        return 0

    def _get_learning_rate(self):
        if not self.__model.ModelInfo.contains("params"):
            return {}
        cdef const char* c_params_json = self.__model.ModelInfo["params"].c_str()
        cdef bytes py_params_json = c_params_json
        params_json = to_native_str(py_params_json)
        if params_json:
            params = loads(params_json)
            if 'boosting_options' in params:
                return params['boosting_options'].get('learning_rate', None)
        return None

    def _get_n_features_in(self):
        return self.__n_features_in

    def _get_metadata_wrapper(self):
        return _MetadataHashProxy(self)

    def _get_feature_names(self):
        return [to_native_str(s) for s in GetModelUsedFeaturesNames(dereference(self.__model))]

    def _get_class_labels(self):
        return _get_model_class_labels(self.__model[0])

    cpdef _sum_models(self, models, weights, ctr_merge_policy):
        cdef TVector[TFullModel_const_ptr] models_vector
        cdef TVector[double] weights_vector
        cdef TVector[TString] model_prefix_vector
        cdef ECtrTableMergePolicy merge_policy
        if not TryFromString[ECtrTableMergePolicy](to_arcadia_string(ctr_merge_policy), merge_policy):
            raise CatBoostError("Unknown ctr table merge policy {}".format(ctr_merge_policy))
        assert(len(models) == len(weights))
        for model_id in range(len(models)):
            models_vector.push_back((<_CatBoost>models[model_id]).__model)
            weights_vector.push_back(weights[model_id])
        cdef TFullModel tmp_model = SumModels(models_vector, weights_vector, model_prefix_vector, merge_policy)
        self.model_blob = None
        self.__model.Swap(tmp_model)

    cpdef _save_borders(self, output_file):
        SaveModelBorders(to_arcadia_string(fspath(output_file)), dereference(self.__model))

    cpdef _check_model_and_dataset_compatibility(self, _PoolBase pool):
        if pool:
            CheckModelAndDatasetCompatibility(
                dereference(self.__model),
                dereference(pool.__pool.Get()[0].ObjectsData.Get())
            )

    cpdef _get_tree_splits(self, size_t tree_idx, _PoolBase pool):
        cdef TVector[TString] splits = GetTreeSplitsDescriptions(
            dereference(self.__model),
            tree_idx,
            pool.__pool if pool else TDataProviderPtr(),
        )

        node_descriptions = [to_native_str(s) for s in splits]
        return node_descriptions

    cpdef _get_tree_leaf_values(self, tree_idx):
        leaf_values = GetTreeLeafValuesDescriptions(dereference(self.__model), tree_idx)
        return [to_native_str(value) for value in leaf_values]

    cpdef _get_tree_step_nodes(self, tree_idx):
        step_nodes = GetTreeStepNodes(dereference(self.__model), tree_idx)
        return [(node.LeftSubtreeDiff, node.RightSubtreeDiff) for node in step_nodes]

    cpdef _get_tree_node_to_leaf(self, tree_idx):
        return list(GetTreeNodeToLeaf(dereference(self.__model), tree_idx))

    cpdef _tune_hyperparams(self, list grids_list, _PoolBase train_pool, dict params, int n_iter,
                          int fold_count, int partition_random_seed, bool_t shuffle, bool_t stratified,
                          double train_size, bool_t choose_by_train_test_split, bool_t return_cv_results,
                          custom_folds, int verbose):

        prep_params = _PreprocessParams(params)
        prep_grids = _PreprocessGrids(grids_list)

        self._reserve_test_evals(1)
        self._clear_test_evals()

        cdef TCrossValidationParams cvParams
        cvParams.FoldCount = fold_count
        cvParams.PartitionRandSeed = partition_random_seed
        cvParams.Shuffle = shuffle
        cvParams.Stratified = stratified
        cvParams.Type = ECrossValidation_Classical
        cvParams.IsCalledFromSearchHyperparameters = True;

        cdef TMaybe[TCustomTrainTestSubsets] custom_train_test_subset
        if custom_folds is not None:
            custom_train_test_subset = _make_train_test_subsets(train_pool, custom_folds)
            cvParams.FoldCount = custom_train_test_subset.GetRef().first.size()
            cvParams.customTrainSubsets = custom_train_test_subset.GetRef().first
            cvParams.customTestSubsets = custom_train_test_subset.GetRef().second

        cdef TTrainTestSplitParams ttParams
        ttParams.PartitionRandSeed = partition_random_seed
        ttParams.Shuffle = shuffle
        ttParams.Stratified = False
        ttParams.TrainPart = train_size

        cdef TBestOptionValuesWithCvResult results
        cdef TMetricsAndTimeLeftHistory trainTestResults
        with nogil:
            SetPythonInterruptHandler()
            try:
                if n_iter == -1:
                    GridSearch(
                        prep_grids.tree,
                        prep_params.tree,
                        ttParams,
                        cvParams,
                        prep_params.customObjectiveDescriptor,
                        prep_params.customMetricDescriptor,
                        train_pool.__pool,
                        &results,
                        &trainTestResults,
                        choose_by_train_test_split,
                        return_cv_results,
                        verbose
                    )
                else:
                    RandomizedSearch(
                        n_iter,
                        prep_grids.custom_rnd_dist_gens,
                        prep_grids.tree,
                        prep_params.tree,
                        ttParams,
                        cvParams,
                        prep_params.customObjectiveDescriptor,
                        prep_params.customMetricDescriptor,
                        train_pool.__pool,
                        &results,
                        &trainTestResults,
                        choose_by_train_test_split,
                        return_cv_results,
                        verbose
                    )
            finally:
                ResetPythonInterruptHandler()
        cv_results = defaultdict(list)
        result_metrics = set()
        cdef THashMap[TString, double] metric_result
        if choose_by_train_test_split:
            self.__metrics_history = trainTestResults
        for metric_idx in xrange(results.CvResult.size()):
            name = to_native_str(results.CvResult[metric_idx].Metric)
            if name in result_metrics:
                continue
            _prepare_cv_result(
                name,
                results.CvResult[metric_idx].Iterations,
                results.CvResult[metric_idx].AverageTrain,
                results.CvResult[metric_idx].StdDevTrain,
                results.CvResult[metric_idx].AverageTest,
                results.CvResult[metric_idx].StdDevTest,
                cv_results
            )
            result_metrics.add(name)

        best_params = {}
        for key, value in results.BoolOptions:
            best_params[to_native_str(key)] = value
        for key, value in results.IntOptions:
            best_params[to_native_str(key)] = value
        for key, value in results.UIntOptions:
            best_params[to_native_str(key)] = value
        for key, value in results.DoubleOptions:
            best_params[to_native_str(key)] = value
        for key, value in results.StringOptions:
            best_params[to_native_str(key)] = to_native_str(value)
        for key, value in results.ListOfDoublesOptions:
            best_params[to_native_str(key)] = [float(elem) for elem in value]
        search_result = {}
        search_result["params"] = best_params
        if return_cv_results:
            search_result["cv_results"] = cv_results
        return search_result

    cpdef _select_features(self, _PoolBase train_pool, _PoolBase test_pool, dict params):
        prep_params = _PreprocessParams(params)

        cdef TDataProviders dataProviders
        dataProviders.Learn = train_pool.__pool
        if test_pool:
            dataProviders.Test.push_back(test_pool.__pool)
        self._reserve_test_evals(dataProviders.Test.size())
        self._clear_test_evals()

        cdef TJsonValue summary_json
        with nogil:
            SetPythonInterruptHandler()
            try:
                summary_json = SelectFeatures(
                    prep_params.tree,
                    prep_params.customMetricDescriptor,
                    dataProviders,
                    self.__model,
                    self.__test_evals,
                    &self.__metrics_history,
                )
            finally:
                ResetPythonInterruptHandler()
        return loads(to_native_str(WriteTJsonValue(summary_json)))

    cpdef _get_binarized_statistics(self, _PoolBase pool, catFeaturesNums, floatFeaturesNums, predictionType, int thread_count):
        thread_count = UpdateThreadCount(thread_count)
        cdef TVector[TBinarizedFeatureStatistics] statistics
        cdef TVector[size_t] catFeaturesNumsVec
        cdef TVector[size_t] floatFeaturesNumsVec
        for num in catFeaturesNums:
            catFeaturesNumsVec.push_back(num)
        for num in floatFeaturesNums:
            floatFeaturesNumsVec.push_back(num)
        statistics_vec = GetBinarizedStatistics(
            dereference(self.__model),
            dereference(pool.__pool.Get()),
            catFeaturesNumsVec,
            floatFeaturesNumsVec,
            string_to_prediction_type(predictionType),
            thread_count
        )
        statistics_list = []
        for stat in statistics_vec:
            statistics_list.append(
                {
                    'borders': _vector_of_floats_to_np_array(stat.Borders),
                    'binarized_feature': _vector_of_ints_to_np_array(stat.BinarizedFeature),
                    'mean_target': _vector_of_floats_to_np_array(stat.MeanTarget),
                    'mean_weighted_target': _vector_of_floats_to_np_array(stat.MeanWeightedTarget),
                    'mean_prediction': _vector_of_floats_to_np_array(stat.MeanPrediction),
                    'objects_per_bin': _vector_of_size_t_to_np_array(stat.ObjectsPerBin),
                    'predictions_on_varying_feature': _vector_of_double_to_np_array(stat.PredictionsOnVaryingFeature)
                }
            )
        return statistics_list

    cpdef _calc_cat_feature_perfect_hash(self, value, size_t featureNum):
        return GetCatFeaturePerfectHash(dereference(self.__model), to_arcadia_string(value), featureNum)

    cpdef _get_feature_type_and_internal_index(self, int flatFeatureIndex):
        cdef TFeatureTypeAndInternalIndex typeAndIndex = GetFeatureTypeAndInternalIndex(
            dereference(self.__model), flatFeatureIndex)
        if typeAndIndex.Type == EFeatureType_Float:
            return 'float', typeAndIndex.Index
        elif typeAndIndex.Type == EFeatureType_Categorical:
            return 'categorical', typeAndIndex.Index
        else:
            return 'unknown', -1

    cpdef _get_cat_feature_values(self, _PoolBase pool, size_t flatFeatureIndex):
        cdef TVector[TString] values = GetCatFeatureValues(
            dereference(pool.__pool.Get()),
            flatFeatureIndex)
        res = {to_native_str(val) for val in values}
        return res

    cpdef _get_leaf_values(self):
        return _constarrayref_of_double_to_np_array(self.__model.ModelTrees.Get().GetModelTreeData().Get().GetLeafValues())

    cpdef _get_leaf_weights(self):
        result = np.empty(self.__model.ModelTrees.Get().GetModelTreeData().Get().GetLeafValues().size(), dtype=_npfloat64)
        cdef size_t curr_index = 0
        cdef TConstArrayRef[double] arrayView = self.__model.ModelTrees.Get().GetModelTreeData().Get().GetLeafWeights()
        for val in arrayView:
            result[curr_index] = val
            curr_index += 1
        assert curr_index == 0 or curr_index == self.__model.ModelTrees.Get().GetModelTreeData().Get().GetLeafValues().size(), (
            "wrong number of leaf weights")
        return result

    cpdef _get_tree_leaf_counts(self):
        return _vector_of_uints_to_np_array(self.__model.ModelTrees.Get().GetTreeLeafCounts())

    cpdef _set_leaf_values(self, new_leaf_values):
        assert isinstance(new_leaf_values, np.ndarray), "expected numpy.ndarray."
        assert new_leaf_values.dtype == np.float64, "leaf values should have type np.float64 (double)."
        assert len(new_leaf_values.shape) == 1, "leaf values should be a 1d-vector."
        assert new_leaf_values.shape[0] == self.__model.ModelTrees.Get().GetModelTreeData().Get().GetLeafValues().size(), (
            "count of leaf values should be equal to the leaf count.")
        cdef TVector[double] model_leafs = new_leaf_values
        self.__model.ModelTrees.GetMutable().GetModelTreeData().Get().SetLeafValues(model_leafs)

    cpdef _set_feature_names(self, feature_names):
            cdef TVector[TString] feature_names_vector
            for value in feature_names:
                feature_names_vector.push_back(to_arcadia_string(str(value)))
            SetModelExternalFeatureNames(feature_names_vector, self.__model)

    cpdef _convert_oblivious_to_asymmetric(self):
        self.__model.ModelTrees.GetMutable().ConvertObliviousToAsymmetric()

    cpdef _get_nan_treatments(self):
        cdef THashMap[int, ENanValueTreatment] nanTreatmentsMap = GetNanTreatments(dereference(self.__model))
        nanTreatments = {}
        for pair in nanTreatmentsMap:
            if pair.second == ENanValueTreatment_AsIs:
                nanTreatments[pair.first] = 'AsIs'
            elif pair.second == ENanValueTreatment_AsFalse:
                nanTreatments[pair.first] = 'AsFalse'
            else:
                nanTreatments[pair.first] = 'AsTrue'
        return nanTreatments

    cpdef _get_binclass_probability_threshold(self):
        return self.__model.GetBinClassProbabilityThreshold()


cdef class _MetadataHashProxy:
    cdef _CatBoost _catboost
    def __init__(self, catboost):
        self._catboost = catboost  # here we store reference to _Catboost class to increment object ref count

    def __getitem__(self, key):
        if not isinstance(key, string_types):
            raise CatBoostError('only string keys allowed')
        cdef TString key_str = to_arcadia_string(key)
        if not self._catboost.__model.ModelInfo.contains(key_str):
            raise KeyError
        return to_native_str(self._catboost.__model.ModelInfo.at(key_str))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if not isinstance(key, string_types):
            raise CatBoostError('only string keys allowed')
        if not isinstance(value, string_types):
            raise CatBoostError('only string values allowed')
        self._catboost.__model.ModelInfo[to_arcadia_string(key)] = to_arcadia_string(value)

    def __delitem__(self, key):
        if not isinstance(key, string_types):
            raise CatBoostError('only string keys allowed')
        cdef TString key_str = to_arcadia_string(key)
        if not self._catboost.__model.ModelInfo.contains(key_str):
            raise KeyError
        self._catboost.__model.ModelInfo.erase(key_str)

    def __len__(self):
        return self._catboost.__model.ModelInfo.size()

    def keys(self):
        return [to_native_str(kv.first) for kv in self._catboost.__model.ModelInfo]

    def iterkeys(self):
        return (to_native_str(kv.first) for kv in self._catboost.__model.ModelInfo)

    def __iter__(self):
        return self.iterkeys()

    def items(self):
        return [(to_native_str(kv.first), to_native_str(kv.second)) for kv in self._catboost.__model.ModelInfo]

    def iteritems(self):
        return ((to_native_str(kv.first), to_native_str(kv.second)) for kv in self._catboost.__model.ModelInfo)


cdef TCustomTrainTestSubsets _make_train_test_subsets(_PoolBase pool, folds) except *:
    num_data = pool.num_row()

    if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
        raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                             "or scikit-learn splitter object with split method")

    cdef TMaybeData[TConstArrayRef[TGroupId]] arr_group_ids = pool.__pool.Get()[0].ObjectsData.Get()[0].GetGroupIds()

    if hasattr(folds, 'split'):
        if arr_group_ids.Defined():
            flatted_group = _CreateNumpyUI64ArrayView(arr_group_ids.GetRef().data(), arr_group_ids.GetRef().size())
        else:
            flatted_group = np.zeros(num_data, dtype=int)
        folds = folds.split(X=np.zeros(num_data), y=pool.get_label(), groups=flatted_group)

    cdef TVector[TVector[ui32]] custom_train_subsets
    cdef TVector[TVector[ui32]] custom_test_subsets

    cdef THashSet[ui64] train_group_ids
    cdef THashMap[TGroupId, ui64] map_group_id_to_group_number
    cdef ui64 current_num
    cdef const TGroupId* group_id_ptr
    cdef TGroupId current_group

    if not arr_group_ids.Defined():
        for train_test in folds:
            train = train_test[0]
            test = train_test[1]

            custom_train_subsets.emplace_back()
            for subset in train:
                custom_train_subsets.back().push_back(subset)

            custom_test_subsets.emplace_back()
            for subset in test:
                custom_test_subsets.back().push_back(subset)
    else:
        current_num = 0
        group_id_ptr = arr_group_ids.GetRef().data()
        for idx in range(arr_group_ids.GetRef().size()):
            if idx == 0 or group_id_ptr[idx] != group_id_ptr[idx - 1]:
                map_group_id_to_group_number[group_id_ptr[idx]] = current_num
                current_num = current_num + 1

        for train_test in folds:
            train = train_test[0]
            test = train_test[1]
            train_group_ids.clear()

            custom_train_subsets.emplace_back()

            for idx in range(len(train)):
                current_group = group_id_ptr[train[idx]]
                if idx == 0 or current_group != group_id_ptr[train[idx - 1]]:
                    custom_train_subsets.back().push_back(map_group_id_to_group_number[current_group])
                    train_group_ids.insert(map_group_id_to_group_number[current_group])

            custom_test_subsets.emplace_back()

            for idx in range(len(test)):
                current_group = group_id_ptr[test[idx]]

                if train_group_ids.contains(map_group_id_to_group_number[current_group]):
                    raise CatBoostError('Objects with the same group id must be in the same fold.')

                if idx == 0 or current_group != group_id_ptr[test[idx - 1]]:
                    custom_test_subsets.back().push_back(map_group_id_to_group_number[current_group])

    cdef TCustomTrainTestSubsets result
    result.first = custom_train_subsets
    result.second = custom_test_subsets

    return result


cpdef _cv(dict params, _PoolBase pool, int fold_count, bool_t inverted, int partition_random_seed,
          bool_t shuffle, bool_t stratified, float metric_update_interval, bool_t as_pandas, folds,
          type, bool_t return_models):
    prep_params = _PreprocessParams(params)
    cdef TCrossValidationParams cvParams
    cdef TVector[TCVResult] results
    cdef TVector[TFullModel] cvFullModels

    cvParams.FoldCount = fold_count
    cvParams.PartitionRandSeed = partition_random_seed
    cvParams.Shuffle = shuffle
    cvParams.Stratified = stratified
    cvParams.MetricUpdateInterval = metric_update_interval
    cvParams.ReturnModels = return_models

    if type == 'Classical':
        cvParams.Type = ECrossValidation_Classical
    elif type == 'Inverted':
        cvParams.Type = ECrossValidation_Inverted
    else:
        cvParams.Type = ECrossValidation_TimeSeries

    cdef TMaybe[TCustomTrainTestSubsets] custom_train_test_subset
    if folds is not None:
        custom_train_test_subset = _make_train_test_subsets(pool, folds)
        cvParams.FoldCount = custom_train_test_subset.GetRef().first.size()
        cvParams.customTrainSubsets = custom_train_test_subset.GetRef().first
        cvParams.customTestSubsets = custom_train_test_subset.GetRef().second

    with nogil:
        SetPythonInterruptHandler()
        try:
            CrossValidate(
                prep_params.tree,
                TQuantizedFeaturesInfoPtr(<TQuantizedFeaturesInfo*>nullptr),
                prep_params.customObjectiveDescriptor,
                prep_params.customMetricDescriptor,
                pool.__pool,
                cvParams,
                &results)
        finally:
            ResetPythonInterruptHandler()

    cv_results = defaultdict(list)
    result_metrics = set()
    for metric_idx in xrange(results.size()):
        name = to_native_str(results[metric_idx].Metric)
        if name in result_metrics:
            continue
        _prepare_cv_result(
            name,
            results[metric_idx].Iterations,
            results[metric_idx].AverageTrain,
            results[metric_idx].StdDevTrain,
            results[metric_idx].AverageTest,
            results[metric_idx].StdDevTest,
            cv_results
        )
        result_metrics.add(name)
    if as_pandas:
        results_output = pd.DataFrame.from_dict(cv_results)
    else:
        results_output = cv_results
    if return_models:
        cv_models = []
        cvFullModels = results.front().CVFullModels
        for i in range(<int>cvFullModels.size()):
            catboost_model = _CatBoost()
            catboost_model.__model.Swap(cvFullModels[i])
            cv_models.append(catboost_model)
        return results_output, cv_models
    return results_output


cdef _convert_to_visible_labels(EPredictionType predictionType, TVector[TVector[double]] raws, int thread_count, TFullModel* model):
    cdef size_t objectCount
    cdef size_t objectIdx
    cdef size_t dim
    cdef size_t dimIdx
    cdef TConstArrayRef[double] raws1d

    if predictionType == string_to_prediction_type('Class'):
        loss_function = _get_loss_function_name(model[0])
        if loss_function in ('MultiLogloss', 'MultiCrossEntropy'):
            return _2d_vector_of_double_to_np_array(raws).astype(int)

        assert (raws.size() == 1)
        raws1d = TConstArrayRef[double](raws[0])
        objectCount = raws1d.size()

        model_class_labels = _get_model_class_labels(model[0])

        class_label_type = type(model_class_labels[0])
        result = np.empty((1, objectCount), object if class_label_type == str else class_label_type)
        for objectIdx in range(objectCount):
            result[0][objectIdx] = model_class_labels[<ui32>raws1d[objectIdx]]
        return result

    return _2d_vector_of_double_to_np_array(raws)


cdef _get_metrics_evals_pydict(TMetricsAndTimeLeftHistory history):
    metrics_evals = defaultdict(functools.partial(defaultdict, list))

    iteration_count = history.LearnMetricsHistory.size()
    for iteration_num in range(iteration_count):
        for metric, value in history.LearnMetricsHistory[iteration_num]:
            metrics_evals["learn"][to_native_str(metric)].append(value)

    if not history.TestMetricsHistory.empty():
        test_count = 0
        for i in range(iteration_count):
            test_count = max(test_count, history.TestMetricsHistory[i].size())
        for iteration_num in range(iteration_count):
            for test_index in range(history.TestMetricsHistory[iteration_num].size()):
                eval_set_name = "validation"
                if test_count > 1:
                    eval_set_name += "_" + str(test_index)
                for metric, value in history.TestMetricsHistory[iteration_num][test_index]:
                    metrics_evals[eval_set_name][to_native_str(metric)].append(value)
    return {k: dict(v) for k, v in iteritems(metrics_evals)}



cdef class _StagedPredictIterator:
    cdef TVector[double] __flatApprox
    cdef TVector[TVector[double]] __approx
    cdef TVector[TVector[double]] __pred
    cdef TFullModel* __model
    cdef TAtomicSharedPtr[TTbbLocalExecutor] __executor
    cdef TModelCalcerOnPool* __modelCalcerOnPool
    cdef EPredictionType predictionType
    cdef int ntree_start, ntree_end, eval_period, thread_count
    cdef bool_t verbose

    def __cinit__(self, str prediction_type, int ntree_start, int ntree_end, int eval_period, int thread_count, verbose):
        self.predictionType = string_to_prediction_type(prediction_type)
        self.ntree_start = ntree_start
        self.ntree_end = ntree_end
        self.eval_period = eval_period
        self.thread_count = UpdateThreadCount(thread_count)
        self.verbose = verbose
        self.__executor = GetCachedLocalExecutor(thread_count)

    cdef _initialize_model_calcer(self, TFullModel* model, _PoolBase pool):
        self.__model = model
        self.__modelCalcerOnPool = new TModelCalcerOnPool(
            dereference(self.__model),
            pool.__pool.Get()[0].ObjectsData,
            <ILocalExecutor*>self.__executor.Get()
        )
        cdef TMaybeData[TBaselineArrayRef] maybe_baseline = pool.__pool.Get()[0].RawTargetData.GetBaseline()
        cdef TBaselineArrayRef baseline
        if maybe_baseline.Defined():
            baseline = maybe_baseline.GetRef()
            for baseline_idx in range(baseline.size()):
                for object_idx in range(pool.num_row()):
                    self.__approx[object_idx][baseline_idx] = baseline[baseline_idx][object_idx]

    def __dealloc__(self):
        del self.__modelCalcerOnPool

    def __deepcopy__(self, _):
        raise CatBoostError('Can\'t deepcopy _StagedPredictIterator object')

    def __next__(self):
        if self.ntree_start >= self.ntree_end:
            raise StopIteration

        dereference(self.__modelCalcerOnPool).ApplyModelMulti(
            string_to_prediction_type('RawFormulaVal'),
            self.ntree_start,
            min(self.ntree_start + self.eval_period, self.ntree_end),
            &self.__flatApprox,
            &self.__pred
        )

        if self.__approx.empty():
            self.__approx.swap(self.__pred)
        else:
            for i in range(self.__approx.size()):
                for j in range(self.__approx[0].size()):
                    self.__approx[i][j] += self.__pred[i][j]

        self.ntree_start += self.eval_period
        self.__pred = PrepareEvalForInternalApprox(self.predictionType, dereference(self.__model), self.__approx, self.thread_count)

        return transform_predictions(self.__pred, self.predictionType, self.thread_count, self.__model)

    def __iter__(self):
        return self

cdef class _LeafIndexIterator:
    cdef TLeafIndexCalcerOnPool* __leafIndexCalcer

    cdef _initialize(self, TFullModel* model, _PoolBase pool, int ntree_start, int ntree_end):
        self.__leafIndexCalcer = new TLeafIndexCalcerOnPool(
            dereference(model),
            pool.__pool.Get()[0].ObjectsData,
            ntree_start,
            ntree_end
        )

    def __dealloc__(self):
        del self.__leafIndexCalcer

    def __deepcopy__(self, _):
        raise CatBoostError('Can\'t deepcopy _LeafIndexIterator object')

    def __next__(self):
        if not dereference(self.__leafIndexCalcer).CanGet():
            raise StopIteration
        result = _vector_of_uints_to_np_array(dereference(self.__leafIndexCalcer).Get())
        dereference(self.__leafIndexCalcer).Next()
        return result

    def __iter__(self):
        return self

class MetricDescription:

    def __init__(self, metric_name, is_max_optimal):
        self._metric_description = metric_name
        self._is_max_optimal = is_max_optimal

    def get_metric_description(self):
        return self._metric_description

    def is_max_optimal(self):
        return self._is_max_optimal

    def __str__(self):
        return self._metric_description

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self._metric_description == other._metric_description and self._is_max_optimal == other._is_max_optimal

    def __hash__(self):
        return hash((self._metric_description, self._is_max_optimal))


def _metric_description_or_str_to_str(metric_description):
    key = None
    if isinstance(metric_description, MetricDescription):
        key = metric_description.get_metric_description()
    else:
        key = metric_description
    return key


class EvalMetricsResult:

    def __init__(self, plots, metrics_description):
        self._plots = dict()
        self._metric_descriptions = dict()

        for (plot, metric) in zip(plots, metrics_description):
            key = _metric_description_or_str_to_str(metric)
            self._metric_descriptions[key] = metrics_description
            self._plots[key] = plot


    def has_metric(self, metric_description):
        key = _metric_description_or_str_to_str(metric_description)
        return key in self._metric_descriptions

    def get_metric(self, metric_description):
        key = _metric_description_or_str_to_str(metric_description)
        return self._metric_descriptions[key]

    def get_result(self, metric_description):
        key = _metric_description_or_str_to_str(metric_description)
        return self._plots[key]


cdef class _MetricCalcerBase:
    cdef TMetricsPlotCalcerPythonWrapper*__calcer
    cdef _CatBoost __catboost

    cpdef _create_calcer(self, metrics_description, int ntree_start, int ntree_end, int eval_period, int thread_count,
                         tmp_dir, bool_t delete_temp_dir_on_exit):
        thread_count=UpdateThreadCount(thread_count);
        cdef TVector[TString] metricsDescription
        for metric_description in metrics_description:
            metricsDescription.push_back(to_arcadia_string(metric_description))

        self.__calcer = new TMetricsPlotCalcerPythonWrapper(metricsDescription, dereference(self.__catboost.__model),
                                                            ntree_start, ntree_end, eval_period, thread_count,
                                                            to_arcadia_string(fspath(tmp_dir)), delete_temp_dir_on_exit)

        self._metric_descriptions = list()

        cdef TVector[const IMetric*] metrics = self.__calcer.GetMetricRawPtrs()

        for metric_idx in xrange(metrics.size()):
            metric = metrics[metric_idx]
            name = to_native_str(metric.GetDescription().c_str())
            flag = IsMaxOptimal(dereference(metric))
            self._metric_descriptions.append(MetricDescription(name, flag))

    def __init__(self, catboost_model, *args, **kwargs):
        self.__catboost = catboost_model
        self._metric_descriptions = list()

    def __dealloc__(self):
        del self.__calcer

    def metric_descriptions(self):
        return self._metric_descriptions

    def eval_metrics(self):
        cdef TVector[TVector[double]] plots = self.__calcer.ComputeScores()
        return EvalMetricsResult([[value for value in plot] for plot in plots],
                                 self._metric_descriptions)

    cpdef add(self, _PoolBase pool):
        self.__calcer.AddPool(pool.__pool.Get()[0])

    def __deepcopy__(self):
        raise CatBoostError('Can\'t deepcopy _MetricCalcerBase object')


cdef to_tvector(np.ndarray[double, ndim=1, mode="c"] x):
    cdef TVector[double] result
    result.assign(<double *>x.data, <double *>x.data + x.shape[0])
    return result


cpdef _eval_metric_util(
    label_param, approx_param, metric, weight_param, group_id_param,
    group_weight_param, subgroup_id_param, pairs_param, thread_count
):
    if (len(label_param[0]) != len(approx_param[0])):
        raise CatBoostError('Label and approx should have same sizes.')
    doc_count = len(label_param[0]);

    cdef TVector[TVector[float]] label
    for labelIdx in range(len(label_param)):
        label.push_back(to_tvector(np.array(label_param[labelIdx], dtype='double').ravel()))

    cdef TVector[TVector[double]] approx
    for i in range(len(approx_param)):
        approx.push_back(to_tvector(np.array(approx_param[i], dtype='double').ravel()))

    cdef TVector[float] weight
    if weight_param is not None:
        if (len(weight_param) != doc_count):
            raise CatBoostError('Label and weight should have same sizes.')
        weight = to_tvector(np.array(weight_param, dtype='double').ravel())

    cdef TString group_id_strbuf

    cdef TVector[TGroupId] group_id;
    if group_id_param is not None:
        if (len(group_id_param) != doc_count):
            raise CatBoostError('Label and group_id should have same sizes.')
        group_id.resize(doc_count)
        for i in range(doc_count):
            get_id_object_bytes_string_representation(group_id_param[i], &group_id_strbuf)
            group_id[i] = CalcGroupIdFor(<TStringBuf>group_id_strbuf)

    cdef TVector[float] group_weight
    if group_weight_param is not None:
        if (len(group_weight_param) != doc_count):
            raise CatBoostError('Label and group weight should have same sizes.')
        group_weight = to_tvector(np.array(group_weight_param, dtype='double').ravel())

    cdef TString subgroup_id_strbuf

    cdef TVector[TSubgroupId] subgroup_id;
    if subgroup_id_param is not None:
        if (len(subgroup_id_param) != doc_count):
            raise CatBoostError('Label and subgroup_id should have same sizes.')
        subgroup_id.resize(doc_count)
        for i in range(doc_count):
            get_id_object_bytes_string_representation(subgroup_id_param[i], &subgroup_id_strbuf)
            subgroup_id[i] = CalcSubgroupIdFor(<TStringBuf>subgroup_id_strbuf)

    cdef TVector[TPair] pairs;
    if pairs_param is not None:
        pairs.resize(len(pairs_param))
        for i in range(len(pairs_param)):
            pairs[i] = TPair(pairs_param[i][0], pairs_param[i][1], 1)

    thread_count = UpdateThreadCount(thread_count);

    return EvalMetricsForUtils(
        <TConstArrayRef[TVector[float]]>(label),
        approx,
        to_arcadia_string(metric),
        weight,
        group_id,
        group_weight,
        subgroup_id,
        pairs,
        thread_count
    )


cpdef _get_confusion_matrix(model, pool, thread_count):
    thread_count = UpdateThreadCount(thread_count)
    cdef TVector[double] cm = MakeConfusionMatrix(
        dereference((<_CatBoost>model).__model), (<_PoolBase>pool).__pool, thread_count
    )
    n_classes = int(np.sqrt(cm.size()))
    return np.array([counter for counter in cm]).reshape((n_classes, n_classes))


cpdef _get_roc_curve(model, pools_list, thread_count):
    thread_count = UpdateThreadCount(thread_count)
    cdef TVector[TDataProviderPtr] pools
    for pool in pools_list:
        pools.push_back((<_PoolBase>pool).__pool)
    cdef TVector[TRocPoint] curve = TRocCurve(
        dereference((<_CatBoost>model).__model), pools, thread_count
    ).GetCurvePoints()
    tpr = np.array([1 - point.FalseNegativeRate for point in curve])
    fpr = np.array([point.FalsePositiveRate for point in curve])
    thresholds = np.array([point.Boundary for point in curve])
    return fpr, tpr, thresholds


cpdef _select_threshold(model, data, curve, FPR, FNR, thread_count):
    if FPR is not None and FNR is not None:
        raise CatBoostError('Only one of the parameters FPR, FNR should be initialized.')

    thread_count = UpdateThreadCount(thread_count)

    cdef TRocCurve rocCurve
    cdef TVector[TRocPoint] points
    cdef TVector[TDataProviderPtr] pools

    if data is not None:
        for pool in data:
            pools.push_back((<_PoolBase>pool).__pool)
        rocCurve = TRocCurve(dereference((<_CatBoost>model).__model), pools, thread_count)
    else:
        size = len(curve[2])
        for i in range(size):
            points.push_back(TRocPoint(curve[2][i], 1 - curve[1][i], curve[0][i]))
        rocCurve = TRocCurve(points)

    if FPR is not None:
        return rocCurve.SelectDecisionBoundaryByFalsePositiveRate(FPR)
    if FNR is not None:
        return rocCurve.SelectDecisionBoundaryByFalseNegativeRate(FNR)
    return rocCurve.SelectDecisionBoundaryByIntersection()


cdef void _WriteLog(const char* str, size_t len, void* targetObject) except * with gil:
    cdef streamLikeObject = <object> targetObject
    cdef bytes bytes_str = str[:len]
    streamLikeObject.write(to_native_str(bytes_str))


cpdef _set_logger(cout, cerr):
    SetCustomLoggingFunction(&_WriteLog, &_WriteLog, <void*>cout, <void*>cerr)


cpdef _reset_logger():
    RestoreOriginalLogger()


cpdef _configure_malloc():
    ConfigureMalloc()


cpdef _library_init():
    LibraryInit()


cdef size_t python_stream_read_func(char* whereToWrite, size_t bufLen, PyObject* stream, TString* errorMsg):
    BUF_SIZE = 64 * 1024
    cdef size_t total_read = 0
    while bufLen > 0:
        curr_read_size = min(BUF_SIZE, bufLen)
        try:
            tmp_str = (<object>stream).read(curr_read_size)
        except BaseException as e:
            errorMsg[0] = to_arcadia_string(str(e))
            return -1
        total_read += len(tmp_str)
        if len(tmp_str) == 0:
            return total_read
        memcpy(whereToWrite, <char*>tmp_str, len(tmp_str))
        whereToWrite += len(tmp_str)
        bufLen -= len(tmp_str)
    return total_read


cpdef compute_wx_test(baseline, test):
    cdef TVector[double] baselineVec
    cdef TVector[double] testVec
    for x in baseline:
        baselineVec.push_back(x)
    for x in test:
        testVec.push_back(x)
    result=WxTest(baselineVec, testVec)
    return {"pvalue" : result.PValue, "wplus":result.WPlus, "wminus":result.WMinus}


cpdef is_classification_objective(loss_name):
    return IsClassificationObjective(to_arcadia_string(loss_name))


cpdef is_cv_stratified_objective(loss_name):
    return IsCvStratifiedObjective(to_arcadia_string(loss_name))


cpdef is_regression_objective(loss_name):
    return IsRegressionObjective(to_arcadia_string(loss_name))


cpdef is_multiregression_objective(loss_name):
    return IsMultiRegressionObjective(to_arcadia_string(loss_name))


cpdef is_multitarget_objective(loss_name):
    return IsMultiTargetObjective(to_arcadia_string(loss_name))


cpdef is_survivalregression_objective(loss_name):
    return IsSurvivalRegressionObjective(to_arcadia_string(loss_name))


cpdef is_groupwise_metric(metric_name):
    return IsGroupwiseMetric(to_arcadia_string(metric_name))


cpdef is_multiclass_metric(metric_name):
    return IsMultiClassCompatibleMetric(to_arcadia_string(metric_name))


cpdef is_pairwise_metric(metric_name):
    return IsPairwiseMetric(to_arcadia_string(metric_name))


cpdef is_ranking_metric(metric_name):
    return IsRankingMetric(to_arcadia_string(metric_name))


cpdef is_minimizable_metric(metric_name):
    return IsMinOptimal(to_arcadia_string(metric_name))


cpdef is_maximizable_metric(metric_name):
    return IsMaxOptimal(to_arcadia_string(metric_name))


cpdef is_user_defined_metric(metric_name):
    return IsUserDefined(to_arcadia_string(metric_name))


cpdef has_gpu_implementation_metric(metric_name):
    return HasGpuImplementation(to_arcadia_string(metric_name))


cpdef get_experiment_name(ui32 feature_set_idx, ui32 fold_idx):
    cdef TString experiment_name = GetExperimentName(feature_set_idx, fold_idx)
    cdef const char* c_experiment_name_string = experiment_name.c_str()
    cdef bytes py_experiment_name_str = c_experiment_name_string[:experiment_name.size()]
    return py_experiment_name_str


cpdef convert_features_to_indices(indices_or_names, cd_path, pool_metainfo_path):
    cdef TJsonValue indices_or_names_as_json = ReadTJsonValue(
        to_arcadia_string(
            dumps(indices_or_names, cls=_NumpyAwareEncoder)
        )
    )
    cdef TPathWithScheme cd_path_with_scheme
    if cd_path is not None:
        cd_path_with_scheme = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(cd_path)), TStringBuf(<char*>'dsv'))

    cdef TPathWithScheme pool_metainfo_path_with_scheme
    if pool_metainfo_path is not None:
        pool_metainfo_path_with_scheme = TPathWithScheme(<TStringBuf>to_arcadia_string(fspath(pool_metainfo_path)), TStringBuf(<char*>''))

    ConvertFeaturesFromStringToIndices(cd_path_with_scheme, pool_metainfo_path_with_scheme, &indices_or_names_as_json)
    return loads(to_native_str(WriteTJsonValue(indices_or_names_as_json)))


cpdef _check_train_params(dict params):
    params_to_check = params.copy()
    if 'cat_features' in params_to_check:
        del params_to_check['cat_features']
    if 'input_borders' in params_to_check:
        del params_to_check['input_borders']
    if 'ignored_features' in params_to_check:
        del params_to_check['ignored_features']
    if 'monotone_constraints' in params_to_check:
        del params_to_check['monotone_constraints']
    if 'feature_weights' in params_to_check:
        del params_to_check['feature_weights']
    if 'first_feature_use_penalties' in params_to_check:
        del params_to_check['first_feature_use_penalties']
    if 'per_object_feature_penalties' in params_to_check:
        del params_to_check['per_object_feature_penalties']


    prep_params = _PreprocessParams(params_to_check)
    CheckFitParams(
        prep_params.tree,
        prep_params.customObjectiveDescriptor.Get(),
        prep_params.customMetricDescriptor.Get())


cpdef _get_gpu_device_count():
    return GetGpuDeviceCount()


cpdef _reset_trace_backend(file):
    ResetTraceBackend(to_arcadia_string(fspath(file)))


@cython.embedsignature(True)
cdef class TargetStats:
    cdef TTargetStats TargetStats

    def __init__(self, float min_value, float max_value):
        self.TargetStats.MinValue = min_value
        self.TargetStats.MaxValue = max_value


@cython.embedsignature(True)
cdef class DataMetaInfo:
    cdef TDataMetaInfo DataMetaInfo

    def __init__(
        self,
        ui64 object_count,
        ui32 feature_count,
        ui64 max_cat_features_uniq_values_on_learn,
        TargetStats target_stats,
        bool_t has_pairs
    ):
        self.DataMetaInfo.ObjectCount = object_count
        self.DataMetaInfo.MaxCatFeaturesUniqValuesOnLearn = max_cat_features_uniq_values_on_learn
        if target_stats is not None:
            self.DataMetaInfo.TargetStats = target_stats.TargetStats
        self.DataMetaInfo.HasPairs = has_pairs
        self.DataMetaInfo.FeaturesLayout = MakeHolder[TFeaturesLayout](feature_count).Release()


@cython.embedsignature(True)
cpdef compute_training_options(dict options, DataMetaInfo train_meta_info, DataMetaInfo test_meta_info=None):
    cdef TMaybe[TDataMetaInfo] testMetaInfo
    if test_meta_info is not None:
        testMetaInfo = test_meta_info.DataMetaInfo
    cdef TJsonValue trainingOptions = GetTrainingOptions(
        _PreprocessParams(options).tree,
        train_meta_info.DataMetaInfo,
        testMetaInfo
    )
    return loads(to_native_str(WriteTJsonValue(trainingOptions)))


cpdef _get_onnx_model(model, export_parameters):
    if not model._is_oblivious():
        raise CatBoostError(
            "ONNX-ML export is available only for models on oblivious trees ")

    cdef TString result = ConvertTreeToOnnxProto(
        dereference((<_CatBoost>model).__model),
        to_arcadia_string(export_parameters),
    )
    cdef const char* result_ptr = result.c_str()
    cdef size_t result_len = result.size()
    return bytes(result_ptr[:result_len])

include "_grid_creator.pxi"
include "_monoforest.pxi"
include "_text_processing.pxi"
