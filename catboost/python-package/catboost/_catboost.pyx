# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

import atexit
import six
from six import iteritems, string_types, PY3
from six.moves import range
from json import dumps, loads, JSONEncoder
from copy import deepcopy
from collections import defaultdict
import functools
import traceback
import numbers

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
from cython.operator cimport dereference, preincrement

from libc.math cimport isnan, modf
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool as bool_t
from libcpp cimport nullptr
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from util.generic.hash cimport THashMap
from util.generic.maybe cimport TMaybe
from util.generic.ptr cimport THolder, TIntrusivePtr, MakeHolder
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui8, ui16, ui32, ui64, i32, i64
from util.string.cast cimport StrToD, TryFromString, ToString

ctypedef const np.float32_t const_float32_t
ctypedef const np.uint32_t const_ui32_t
ctypedef const TString const_TString

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


@cython.embedsignature(True)
class MultiRegressionCustomMetric:
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
class MultiRegressionCustomObjective:
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


cdef public object PyCatboostExceptionType = <object>CatBoostError


cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef void ProcessException()
    cdef void SetPythonInterruptHandler() nogil
    cdef void ResetPythonInterruptHandler() nogil
    cdef void ThrowCppExceptionWithMessage(const TString&) nogil


cdef extern from "library/cpp/threading/local_executor/local_executor.h" namespace "NPar":
    cdef cppclass TLocalExecutor:
        TLocalExecutor() nogil
        void RunAdditionalThreads(int threadCount) nogil except +ProcessException


cdef extern from "catboost/libs/logging/logging.h":
    cdef void SetCustomLoggingFunction(void(*func)(const char*, size_t len) except * with gil, void(*func)(const char*, size_t len) except * with gil)
    cdef void RestoreOriginalLogger()
    cdef void ResetTraceBackend(const TString&)


cdef extern from "catboost/libs/cat_feature/cat_feature.h":
    cdef ui32 CalcCatFeatureHash(TStringBuf feature) except +ProcessException
    cdef float ConvertCatFeatureHashToFloat(ui32 hashVal) except +ProcessException


cdef extern from "catboost/libs/helpers/resource_holder.h" namespace "NCB":
    cdef cppclass IResourceHolder:
        pass

    cdef cppclass TVectorHolder[T](IResourceHolder):
        TVector[T] Data


cdef extern from "catboost/libs/helpers/maybe_owning_array_holder.h" namespace "NCB":
    cdef cppclass TMaybeOwningArrayHolder[T]:
        @staticmethod
        TMaybeOwningArrayHolder[T] CreateNonOwning(TArrayRef[T] arrayRef)

        @staticmethod
        TMaybeOwningArrayHolder[T] CreateOwning(
            TArrayRef[T] arrayRef,
            TIntrusivePtr[IResourceHolder] resourceHolder
        ) except +ProcessException

        T operator[](size_t idx) except +ProcessException

    cdef cppclass TMaybeOwningConstArrayHolder[T]:
        @staticmethod
        TMaybeOwningConstArrayHolder[T] CreateNonOwning(TConstArrayRef[T] arrayRef)

        @staticmethod
        TMaybeOwningConstArrayHolder[T] CreateOwning(
            TConstArrayRef[T] arrayRef,
            TIntrusivePtr[IResourceHolder] resourceHolder
        ) except +ProcessException

        @staticmethod
        TMaybeOwningConstArrayHolder[T] CreateOwningMovedFrom[T2](TVector[T2]& data) except +ProcessException

    cdef TMaybeOwningConstArrayHolder[TDst] CreateConstOwningWithMaybeTypeCast[TDst, TSrc](
        TMaybeOwningArrayHolder[TSrc] src
    ) except +ProcessException

cdef extern from "catboost/libs/helpers/polymorphic_type_containers.h" namespace "NCB":
    cdef cppclass ITypedSequencePtr[T]:
        pass

    cdef cppclass TTypeCastArrayHolder[TInterfaceValue, TStoredValue](ITypedSequencePtr[TInterfaceValue]):
        TTypeCastArrayHolder(TMaybeOwningConstArrayHolder[TStoredValue] values) except +ProcessException

    cdef ITypedSequencePtr[TInterfaceValue] MakeTypeCastArrayHolder[TInterfaceValue, TStoredValue](
        TMaybeOwningConstArrayHolder[TStoredValue] values
    ) except +ProcessException

    cdef ITypedSequencePtr[TInterfaceValue] MakeNonOwningTypeCastArrayHolder[TInterfaceValue, TStoredValue](
        const TStoredValue* begin,
        const TStoredValue* end
    ) except +ProcessException

    cdef ITypedSequencePtr[TInterfaceValue] MakeTypeCastArrayHolderFromVector[TInterfaceValue, TStoredValue](
        TVector[TStoredValue]& values
    ) except +ProcessException


cdef class Py_ITypedSequencePtr:
    cdef ITypedSequencePtr[np.float32_t] result

    def __cinit__(self):
        pass

    cdef set_result(self, ITypedSequencePtr[np.float32_t] result):
        self.result = result

    cdef get_result(self, ITypedSequencePtr[np.float32_t]* result):
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

    cdef Py_ITypedSequencePtr py_result = Py_ITypedSequencePtr()
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


cdef extern from "catboost/libs/helpers/sparse_array.h" namespace "NCB":
    cdef cppclass TSparseArrayIndexingPtr[TSize]:
        pass

    cdef cppclass TConstPolymorphicValuesSparseArray[TValue, TSize]:
        pass

    cdef TSparseArrayIndexingPtr[TSize] MakeSparseArrayIndexing[TSize](
        TSize size,
        TMaybeOwningConstArrayHolder[TSize] indices,
    ) except +ProcessException

    cdef TSparseArrayIndexingPtr[TSize] MakeSparseBlockIndexing[TSize](
        TSize size,
        TMaybeOwningConstArrayHolder[TSize] blockStarts,
        TMaybeOwningConstArrayHolder[TSize] blockLengths
    ) except +ProcessException

    cdef TConstPolymorphicValuesSparseArray[TDstValue, TSize] MakeConstPolymorphicValuesSparseArrayGeneric[TDstValue, TSize](
        TSparseArrayIndexingPtr[TSize] indexing,
        ITypedSequencePtr[TDstValue] nonDefaultValues,
        TDstValue defaultValue
    ) except +ProcessException

    cdef TConstPolymorphicValuesSparseArray[TDstValue, TSize] MakeConstPolymorphicValuesSparseArray[TDstValue, TSrcValue, TSize](
        TSparseArrayIndexingPtr[TSize] indexing,
        TMaybeOwningConstArrayHolder[TSrcValue] nonDefaultValues,
        TDstValue defaultValue
    ) except +ProcessException

    cdef TConstPolymorphicValuesSparseArray[TDstValue, TSize] MakeConstPolymorphicValuesSparseArrayWithArrayIndexGeneric[TDstValue, TSize](
        TSize size,
        TMaybeOwningConstArrayHolder[TSize] indexing,
        ITypedSequencePtr[TDstValue] nonDefaultValues,
        bool_t ordered,
        TDstValue defaultValue
    ) except +ProcessException

    cdef TConstPolymorphicValuesSparseArray[TDstValue, TSize] MakeConstPolymorphicValuesSparseArrayWithArrayIndex[TDstValue, TSrcValue, TSize](
        TSize size,
        TMaybeOwningConstArrayHolder[TSize] indexing,
        TMaybeOwningConstArrayHolder[TSrcValue] nonDefaultValues,
        bool_t ordered,
        TDstValue defaultValue
    ) except +ProcessException

cdef extern from "catboost/private/libs/options/binarization_options.h" namespace "NCatboostOptions" nogil:
    cdef cppclass TBinarizationOptions:
        TBinarizationOptions(...)


cdef extern from "catboost/private/libs/options/enums.h":
    cdef cppclass EFeatureType:
        bool_t operator==(EFeatureType)

    cdef EFeatureType EFeatureType_Float "EFeatureType::Float"
    cdef EFeatureType EFeatureType_Categorical "EFeatureType::Categorical"
    cdef EFeatureType EFeatureType_Text "EFeatureType::Text"


    cdef cppclass EPredictionType:
        bool_t operator==(EPredictionType)

    cdef EPredictionType EPredictionType_Class "EPredictionType::Class"
    cdef EPredictionType EPredictionType_Probability "EPredictionType::Probability"
    cdef EPredictionType EPredictionType_LogProbability "EPredictionType::LogProbability"
    cdef EPredictionType EPredictionType_RawFormulaVal "EPredictionType::RawFormulaVal"
    cdef EPredictionType EPredictionType_Exponent "EPredictionType::Exponent"

    cdef cppclass EFstrType:
        pass

    cdef cppclass EPreCalcShapValues:
        pass

    cdef cppclass ECalcTypeShapValues:
        pass

    cdef cppclass ECrossValidation:
        pass

    cdef ECrossValidation ECrossValidation_TimeSeries "ECrossValidation::TimeSeries"
    cdef ECrossValidation ECrossValidation_Classical "ECrossValidation::Classical"
    cdef ECrossValidation ECrossValidation_Inverted "ECrossValidation::Inverted"

    cdef cppclass ETaskType:
        pass


cdef extern from "catboost/private/libs/options/enums.h" namespace "NCB":
    cdef cppclass ERawTargetType:
        bool_t operator==(ERawTargetType)

    cdef ERawTargetType ERawTargetType_Integer "NCB::ERawTargetType::Integer"
    cdef ERawTargetType ERawTargetType_Float "NCB::ERawTargetType::Float"
    cdef ERawTargetType ERawTargetType_String "NCB::ERawTargetType::String"
    cdef ERawTargetType ERawTargetType_None "NCB::ERawTargetType::None"

cdef extern from "catboost/private/libs/options/json_helper.h":
    cdef TString WriteTJsonValue(const TJsonValue& jsonValue) except +ProcessException

cdef extern from "catboost/private/libs/options/model_based_eval_options.h" namespace "NCatboostOptions" nogil:
    cdef TString GetExperimentName(ui32 featureSetIdx, ui32 foldIdx) except +ProcessException


cdef extern from "catboost/private/libs/quantization_schema/schema.h" namespace "NCB":
    cdef cppclass TPoolQuantizationSchema:
        pass


cdef extern from "catboost/libs/data/features_layout.h" namespace "NCB":
    cdef cppclass TFeatureMetaInfo:
        EFeatureType Type
        TString Name
        bool_t IsSparse
        bool_t IsIgnored
        bool_t IsAvailable

    cdef cppclass TFeaturesLayout:
        TFeaturesLayout() except +ProcessException
        TFeaturesLayout(const ui32 featureCount) except +ProcessException
        TFeaturesLayout(
            const ui32 featureCount,
            const TVector[ui32]& catFeatureIndices,
            const TVector[ui32]& textFeatureIndices,
            const TVector[TString]& featureId,
            bool_t allFeaturesAreSparse
        ) except +ProcessException

        TConstArrayRef[TFeatureMetaInfo] GetExternalFeaturesMetaInfo() except +ProcessException
        TVector[TString] GetExternalFeatureIds() except +ProcessException
        void SetExternalFeatureIds(TConstArrayRef[TString] featureIds) except +ProcessException
        EFeatureType GetExternalFeatureType(ui32 externalFeatureIdx) except +ProcessException
        ui32 GetFloatFeatureCount() except +ProcessException
        ui32 GetCatFeatureCount() except +ProcessException
        ui32 GetExternalFeatureCount() except +ProcessException
        TConstArrayRef[ui32] GetCatFeatureInternalIdxToExternalIdx() except +ProcessException
        TConstArrayRef[ui32] GetTextFeatureInternalIdxToExternalIdx() except +ProcessException

    ctypedef TIntrusivePtr[TFeaturesLayout] TFeaturesLayoutPtr


cdef extern from "catboost/libs/data/meta_info.h" namespace "NCB":
    cdef cppclass TTargetStats:
        float MinValue
        float MaxValue

    cdef cppclass TDataMetaInfo:
        ui64 ObjectCount

        TIntrusivePtr[TFeaturesLayout] FeaturesLayout
        ui64 MaxCatFeaturesUniqValuesOnLearn
        TMaybe[TTargetStats] TargetStats

        ERawTargetType TargetType
        ui32 TargetCount
        ui32 BaselineCount
        bool_t HasGroupId
        bool_t HasGroupWeight
        bool_t HasSubgroupIds
        bool_t HasWeights
        bool_t HasTimestamp
        bool_t HasPairs

        # ColumnsInfo is not here because it is not used for now

        ui32 GetFeatureCount() except +ProcessException

cdef extern from "catboost/libs/data/order.h" namespace "NCB":
    cdef cppclass EObjectsOrder:
        pass

    cdef EObjectsOrder EObjectsOrder_Ordered "NCB::EObjectsOrder::Ordered"
    cdef EObjectsOrder EObjectsOrder_RandomShuffled "NCB::EObjectsOrder::RandomShuffled"
    cdef EObjectsOrder EObjectsOrder_Undefined "NCB::EObjectsOrder::Undefined"


cdef extern from "catboost/private/libs/data_types/pair.h":
    cdef cppclass TPair:
        ui32 WinnerId
        ui32 LoserId
        float Weight
        TPair(ui32 winnerId, ui32 loserId, float weight) nogil except +ProcessException

cdef extern from "catboost/private/libs/data_types/groupid.h":
    ctypedef ui64 TGroupId
    ctypedef ui32 TSubgroupId
    cdef TGroupId CalcGroupIdFor(const TStringBuf& token) except +ProcessException
    cdef TSubgroupId CalcSubgroupIdFor(const TStringBuf& token) except +ProcessException


cdef extern from "catboost/libs/data/util.h" namespace "NCB":
    cdef cppclass TMaybeData[T]:
        TMaybeData(...) except +

        TMaybeData& operator=(...) except +

        void ConstructInPlace(...) except +
        void Clear() except +

        bint Defined()
        bint Empty()

        void CheckDefined() except +

        T* Get() except +
        T& GetRef() except +

        T GetOrElse(T&) except +
        TMaybeData OrElse(TMaybeData&) except +


cdef extern from "catboost/libs/data/quantized_features_info.h" namespace "NCB":
    cdef cppclass TQuantizedFeaturesInfo:
        TQuantizedFeaturesInfo(...)

    ctypedef TIntrusivePtr[TQuantizedFeaturesInfo] TQuantizedFeaturesInfoPtr


cdef extern from "catboost/libs/data/objects_grouping.h" namespace "NCB":
    cdef cppclass TObjectsGrouping:
        pass

    ctypedef TIntrusivePtr[TObjectsGrouping] TObjectsGroupingPtr

    cdef cppclass TObjectsGroupingSubset:
        pass

    cdef TObjectsGroupingSubset GetGroupingSubsetFromObjectsSubset(
        TObjectsGroupingPtr objectsGrouping,
        TVector[ui32]& objectsSubset,
        EObjectsOrder subsetOrder
    ) except +ProcessException

cdef extern from "catboost/libs/data/columns.h" namespace "NCB":
    cdef cppclass TFloatValuesHolder:
        TMaybeOwningArrayHolder[float] ExtractValues(TLocalExecutor* localExecutor) except +ProcessException

cdef extern from "catboost/libs/data/objects.h":
    cdef void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData) except +ProcessException

cdef extern from "catboost/libs/data/objects.h" namespace "NCB":
    cdef cppclass TObjectsDataProvider:
        ui32 GetObjectCount() except +ProcessException
        bool_t EqualTo(const TObjectsDataProvider& rhs, bool_t ignoreSparsity) except +ProcessException
        TMaybeData[TConstArrayRef[TGroupId]] GetGroupIds() except +ProcessException
        TMaybeData[TConstArrayRef[TSubgroupId]] GetSubgroupIds() except +ProcessException
        TMaybeData[TConstArrayRef[ui64]] GetTimestamp() except +ProcessException
        const THashMap[ui32, TString]& GetCatFeaturesHashToString(ui32 catFeatureIdx) except +ProcessException
        TFeaturesLayoutPtr GetFeaturesLayout() except +ProcessException

    cdef cppclass TRawObjectsDataProvider(TObjectsDataProvider):
        void SetGroupIds(TConstArrayRef[TStringBuf] groupStringIds) except +ProcessException
        void SetSubgroupIds(TConstArrayRef[TStringBuf] subgroupStringIds) except +ProcessException
        TMaybeData[const TFloatValuesHolder*] GetFloatFeature(ui32 floatFeatureIdx) except +ProcessException

    cdef cppclass TQuantizedObjectsDataProvider(TObjectsDataProvider):
        TQuantizedFeaturesInfoPtr GetQuantizedFeaturesInfo() except +ProcessException

    cdef THashMap[ui32, TString] MergeCatFeaturesHashToString(const TObjectsDataProvider& objectsData) except +ProcessException

cdef extern from *:
    TRawObjectsDataProvider* dynamic_cast_to_TRawObjectsDataProvider "dynamic_cast<NCB::TRawObjectsDataProvider*>" (TObjectsDataProvider*)
    TQuantizedObjectsDataProvider* dynamic_cast_to_TQuantizedObjectsDataProvider "dynamic_cast<NCB::TQuantizedObjectsDataProvider*>" (TObjectsDataProvider*)


cdef extern from "catboost/libs/data/weights.h" namespace "NCB":
    cdef cppclass TWeights[T]:
        T operator[](ui32 idx) except +ProcessException
        ui32 GetSize() except +ProcessException
        bool_t IsTrivial() except +ProcessException
        TConstArrayRef[T] GetNonTrivialData() except +ProcessException


ctypedef TConstArrayRef[TConstArrayRef[float]] TBaselineArrayRef


cdef extern from "catboost/libs/data/target.h" namespace "NCB":
    cdef cppclass TRawTargetDataProvider:
        ERawTargetType GetTargetType() except +ProcessException
        void GetNumericTarget(TArrayRef[TArrayRef[float]] dst) except +ProcessException
        void GetStringTargetRef(TVector[TConstArrayRef[TString]]* dst) except +ProcessException
        TMaybeData[TBaselineArrayRef] GetBaseline() except +ProcessException
        const TWeights[float]& GetWeights() except +ProcessException
        const TWeights[float]& GetGroupWeights() except +ProcessException
        TConstArrayRef[TPair] GetPairs() except +ProcessException

    cdef cppclass ETargetType:
        pass

    cdef cppclass TTargetDataSpecification:
        ETargetType Type
        TString Description

    cdef cppclass TTargetDataProvider:
        pass

ctypedef TIntrusivePtr[TTargetDataProvider] TTargetDataProviderPtr
ctypedef TIntrusivePtr[TQuantizedObjectsDataProvider] TQuantizedObjectsDataProviderPtr

cdef extern from "catboost/libs/data/data_provider.h" namespace "NCB":
    cdef cppclass TDataProviderTemplate[TTObjectsDataProvider]:
        TDataMetaInfo MetaInfo
        TIntrusivePtr[TTObjectsDataProvider] ObjectsData
        TObjectsGroupingPtr ObjectsGrouping
        TRawTargetDataProvider RawTargetData

        bool_t operator==(const TDataProviderTemplate& rhs)  except +ProcessException
        TIntrusivePtr[TDataProviderTemplate[TTObjectsDataProvider]] GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            ui64 cpuRamLimit,
            int threadCount
        ) except +ProcessException
        ui32 GetObjectCount() except +ProcessException

        void SetBaseline(TBaselineArrayRef baseline) except +ProcessException
        void SetGroupIds(TConstArrayRef[TGroupId] groupIds) except +ProcessException
        void SetGroupWeights(TConstArrayRef[float] groupWeights) except +ProcessException
        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException
        void SetSubgroupIds(TConstArrayRef[TSubgroupId] subgroupIds) except +ProcessException
        void SetWeights(TConstArrayRef[float] weights) except +ProcessException

    ctypedef TDataProviderTemplate[TQuantizedObjectsDataProvider] TQuantizedDataProvider

    ctypedef TDataProviderTemplate[TObjectsDataProvider] TDataProvider
    ctypedef TIntrusivePtr[TDataProvider] TDataProviderPtr

    cdef cppclass TDataProvidersTemplate[TTObjectsDataProvider]:
        TIntrusivePtr[TDataProviderTemplate[TObjectsDataProvider]] Learn
        TVector[TIntrusivePtr[TDataProviderTemplate[TObjectsDataProvider]]] Test

    ctypedef TDataProvidersTemplate[TObjectsDataProvider] TDataProviders


    cdef cppclass TProcessedDataProviderTemplate[TTObjectsDataProvider]:
        TDataMetaInfo MetaInfo
        TObjectsGroupingPtr ObjectsGrouping
        TIntrusivePtr[TTObjectsDataProvider] ObjectsData
        TTargetDataProviderPtr TargetData


    ctypedef TProcessedDataProviderTemplate[TObjectsDataProvider] TProcessedDataProvider
    ctypedef TIntrusivePtr[TProcessedDataProvider] TProcessedDataProviderPtr


    cdef cppclass TTrainingDataProvidersTemplate[TTObjectsDataProvider]:
        TIntrusivePtr[TProcessedDataProviderTemplate[TObjectsDataProvider]] Learn
        TVector[TIntrusivePtr[TProcessedDataProviderTemplate[TObjectsDataProvider]]] Test

    ctypedef TTrainingDataProvidersTemplate[TObjectsDataProvider] TTrainingDataProviders


cdef extern from "catboost/private/libs/quantized_pool/serialization.h" namespace "NCB":
    cdef void SaveQuantizedPool(const TDataProviderPtr& dataProvider, TString fileName) except +ProcessException


cdef extern from "catboost/private/libs/data_util/path_with_scheme.h" namespace "NCB":
    cdef cppclass TPathWithScheme:
        TString Scheme
        TString Path
        TPathWithScheme() except +ProcessException
        TPathWithScheme(const TStringBuf& pathWithScheme, const TStringBuf& defaultScheme) except +ProcessException
        bool_t Inited() except +ProcessException

cdef extern from "catboost/private/libs/data_util/line_data_reader.h" namespace "NCB":
    cdef cppclass TDsvFormatOptions:
        bool_t HasHeader
        char Delimiter

cdef extern from "catboost/private/libs/options/load_options.h" namespace "NCatboostOptions":
    cdef cppclass TColumnarPoolFormatParams:
        TDsvFormatOptions DsvFormat
        TPathWithScheme CdFilePath


cdef extern from "catboost/libs/data/visitor.h" namespace "NCB":
    cdef cppclass IRawObjectsOrderDataVisitor:
        void Start(
            bool_t inBlock,
            const TDataMetaInfo& metaInfo,
            bool_t haveUnknownNumberOfSparseFeatures,
            ui32 objectCount,
            EObjectsOrder objectsOrder,
            TVector[TIntrusivePtr[IResourceHolder]] resourceHolders
        ) except +ProcessException

        void StartNextBlock(ui32 blockSize) except +ProcessException

        void AddGroupId(ui32 localObjectIdx, TGroupId value) except +ProcessException
        void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) except +ProcessException
        void AddTimestamp(ui32 localObjectIdx, ui64 value) except +ProcessException

        void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) except +ProcessException
        void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef[float] features) except +ProcessException

        ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) except +ProcessException
        void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) except +ProcessException
        void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef[ui32] features) except +ProcessException
        void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) except +ProcessException

        void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) except +ProcessException
        void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef[ui32] features) except +ProcessException
        void AddTextFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) except +ProcessException

        void AddTarget(ui32 localObjectIdx, const TString& value) except +ProcessException
        void AddTarget(ui32 localObjectIdx, float value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) except +ProcessException
        void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) except +ProcessException
        void AddWeight(ui32 localObjectIdx, float value) except +ProcessException
        void AddGroupWeight(ui32 localObjectIdx, float value) except +ProcessException

        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException

        void Finish() except +ProcessException

    cdef cppclass IRawFeaturesOrderDataVisitor:
        void Start(
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,
            TVector[TIntrusivePtr[IResourceHolder]] resourceHolders
        )

        void AddGroupId(ui32 objectIdx, TGroupId value) except +ProcessException
        void AddSubgroupId(ui32 objectIdx, TSubgroupId value) except +ProcessException
        void AddTimestamp(ui32 objectIdx, ui64 value) except +ProcessException

        void AddFloatFeature(ui32 flatFeatureIdx, ITypedSequencePtr[float] features) except +ProcessException
        void AddFloatFeature(ui32 flatFeatureIdx, TConstPolymorphicValuesSparseArray[float, ui32] features) except +ProcessException

        ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) except +ProcessException
        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef[TString] feature) except +ProcessException
        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef[TStringBuf] feature) except +ProcessException

        void AddCatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder[ui32] features) except +ProcessException
        void AddCatFeature(ui32 flatFeatureIdx, TConstPolymorphicValuesSparseArray[TString, ui32] features) except +ProcessException

        void AddTextFeature(ui32 flatFeatureIdx, TConstArrayRef[TString] feature) except +ProcessException
        void AddTextFeature(ui32 flatFeatureIdx, TConstArrayRef[TStringBuf] feature) except +ProcessException

        void AddTarget(TConstArrayRef[TString] value) except +ProcessException
        void AddTarget(ITypedSequencePtr[float] value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, TConstArrayRef[TString] value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, ITypedSequencePtr[float] value) except +ProcessException
        void AddBaseline(ui32 baselineIdx, TConstArrayRef[float] value) except +ProcessException
        void AddWeights(TConstArrayRef[float] value) except +ProcessException
        void AddGroupWeights(TConstArrayRef[float] value) except +ProcessException

        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException

        void Finish() except +ProcessException


cdef extern from "catboost/libs/data/data_provider_builders.h" namespace "NCB":
    cdef cppclass IDataProviderBuilder:
        TDataProviderPtr GetResult() except +ProcessException

    cdef cppclass TDataProviderBuilderOptions:
        pass

    cdef void CreateDataProviderBuilderAndVisitor[IVisitor](
        const TDataProviderBuilderOptions& options,
        TLocalExecutor* localExecutor,
        THolder[IDataProviderBuilder]* dataProviderBuilder,
        IVisitor** loader
    ) except +ProcessException


cdef class Py_ObjectsOrderBuilderVisitor:
    cdef TDataProviderBuilderOptions options
    cdef TLocalExecutor local_executor
    cdef THolder[IDataProviderBuilder] data_provider_builder
    cdef IRawObjectsOrderDataVisitor* builder_visitor
    cdef const TFeaturesLayout* features_layout

    def __cinit__(self, thread_count):
        self.local_executor.RunAdditionalThreads(thread_count - 1)
        CreateDataProviderBuilderAndVisitor(
            self.options,
            &self.local_executor,
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
    cdef TLocalExecutor local_executor
    cdef THolder[IDataProviderBuilder] data_provider_builder
    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    cdef const TFeaturesLayout* features_layout

    def __cinit__(self, thread_count):
        self.local_executor.RunAdditionalThreads(thread_count - 1)
        CreateDataProviderBuilderAndVisitor(
            self.options,
            &self.local_executor,
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
        const TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector[ui32]& ignoredFeatures,
        EObjectsOrder objectsOrder,
        int threadCount,
        bool_t verbose
    ) nogil except +ProcessException


cdef extern from "catboost/libs/data/load_and_quantize_data.h" namespace "NCB":
    cdef TDataProviderPtr ReadAndQuantizeDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const TPathWithScheme& timestampsFilePath,
        const TPathWithScheme& baselineFilePath,
        const TPathWithScheme& featureNamesPath,
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


cdef extern from "catboost/private/libs/algo_helpers/hessian.h":
    cdef cppclass THessianInfo:
        TVector[double] Data

cdef extern from "catboost/private/libs/algo/learn_context.h":
    cdef cppclass TLearnProgress:
        pass


cdef extern from "catboost/libs/model/ctr_provider.h":
    cdef cppclass ECtrTableMergePolicy:
        pass

cdef extern from "catboost/libs/model/scale_and_bias.h":
    cdef cppclass TScaleAndBias:
        double Scale
        double Bias

cdef extern from "catboost/libs/model/model.h":
    cdef cppclass TFeaturePosition:
        int Index
        int FlatIndex

    cdef cppclass TCatFeature:
        TFeaturePosition Position
        TString FeatureId

    cdef cppclass TFloatFeature:
        bool_t HasNans
        TFeaturePosition Position
        TVector[float] Borders
        TString FeatureId

    cdef cppclass TTextFeature:
        TFeaturePosition Position
        TString FeatureId

    cdef cppclass TNonSymmetricTreeStepNode:
        ui16 LeftSubtreeDiff
        ui16 RightSubtreeDiff

    cdef cppclass TModelTrees:
        int GetDimensionCount() except +ProcessException
        TConstArrayRef[double] GetLeafValues() except +ProcessException
        TConstArrayRef[double] GetLeafWeights() except +ProcessException
        TConstArrayRef[TCatFeature] GetCatFeatures() except +ProcessException
        TConstArrayRef[TTextFeature] GetTextFeatures() except +ProcessException
        TConstArrayRef[TFloatFeature] GetFloatFeatures() except +ProcessException
        void SetLeafValues(const TVector[double]& leafValues) except +ProcessException
        void DropUnusedFeatures() except +ProcessException
        TVector[ui32] GetTreeLeafCounts() except +ProcessException

        void ConvertObliviousToAsymmetric() except +ProcessException

    cdef cppclass TCOWTreeWrapper:
        const TModelTrees& operator*() except +ProcessException
        const TModelTrees* Get() except +ProcessException
        TModelTrees* GetMutable() except +ProcessException

    cdef cppclass TFullModel:
        TCOWTreeWrapper ModelTrees
        THashMap[TString, TString] ModelInfo

        bool_t operator==(const TFullModel& other) except +ProcessException
        bool_t operator!=(const TFullModel& other) except +ProcessException

        void Swap(TFullModel& other) except +ProcessException
        size_t GetTreeCount() nogil except +ProcessException
        size_t GetDimensionsCount() nogil except +ProcessException
        void Truncate(size_t begin, size_t end) except +ProcessException
        bool_t IsOblivious() except +ProcessException
        TString GetLossFunctionName() except +ProcessException
        TVector[TJsonValue] GetModelClassLabels() except +ProcessException
        TScaleAndBias GetScaleAndBias() except +ProcessException
        void SetScaleAndBias(const TScaleAndBias&) except +ProcessException

    cdef cppclass EModelType:
        pass

    cdef TFullModel ReadModel(const TString& modelFile, EModelType format) nogil except +ProcessException
    cdef TString SerializeModel(const TFullModel& model) except +ProcessException
    cdef TFullModel DeserializeModel(const TString& serializeModelString) nogil except +ProcessException
    cdef TVector[TString] GetModelUsedFeaturesNames(const TFullModel& model) except +ProcessException
    void SetModelExternalFeatureNames(const TVector[TString]& featureNames, TFullModel* model) nogil except +ProcessException
    cdef void SaveModelBorders(const TString& file, const TFullModel& model) nogil except +ProcessException

ctypedef const TFullModel* TFullModel_const_ptr

cdef extern from "catboost/libs/model/model.h":
    cdef TFullModel SumModels(TVector[TFullModel_const_ptr], TVector[double], ECtrTableMergePolicy) nogil except +ProcessException

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

cdef extern from "library/cpp/json/writer/json_value.h" namespace "NJson":
    cdef enum EJsonValueType:
        JSON_UNDEFINED,
        JSON_NULL,
        JSON_BOOLEAN,
        JSON_INTEGER,
        JSON_DOUBLE,
        JSON_STRING,
        JSON_MAP,
        JSON_ARRAY,
        JSON_UINTEGER

    cdef cppclass TJsonValue:
        EJsonValueType GetType()
        i64 GetInteger() except +ProcessException
        double GetDouble() except +ProcessException
        const TString& GetString() except +ProcessException


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

cdef extern from "catboost/private/libs/algo_helpers/ders_holder.h":
    cdef cppclass TDers:
        double Der1
        double Der2


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


cdef extern from "catboost/private/libs/options/enum_helpers.h":
    cdef bool_t IsClassificationObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsCvStratifiedObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsRegressionObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsMultiRegressionObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsGroupwiseMetric(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsMultiClassCompatibleMetric(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsPairwiseMetric(const TString& metricName) nogil except +ProcessException

cdef extern from "catboost/libs/metrics/metric.h":
    cdef cppclass TCustomMetricDescriptor:
        void* CustomData

        ctypedef TMetricHolder (*TEvalFuncPtr)(
            const TVector[TVector[double]]& approx,
            const TConstArrayRef[float] target,
            const TConstArrayRef[float] weight,
            int begin, int end, void* customData) with gil

        ctypedef TMetricHolder (*TEvalMultiregressionFuncPtr)(
            const TConstArrayRef[TVector[double]] approx,
            const TConstArrayRef[TConstArrayRef[float]] target,
            const TConstArrayRef[float] weight,
            int begin, int end, void* customData) with gil

        TMaybe[TEvalFuncPtr] EvalFunc
        TMaybe[TEvalMultiregressionFuncPtr] EvalMultiregressionFunc

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

        void (*CalcDersMultiRegression)(
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
        double MaxTimeSpentOnFixedCostRatio
        ui32 DevMaxIterationsBatchSize
        bool_t IsCalledFromSearchHyperparameters

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
        TDataProviders pools,
        TMaybe[TFullModel*] initModel,
        THolder[TLearnProgress]* initLearnProgress,
        const TString& outputModelPath,
        TFullModel* dstModel,
        const TVector[TEvalResult*]& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        THolder[TLearnProgress]* dstLearnProgress
    ) nogil except +ProcessException

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
            TLocalExecutor* executor
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

cdef extern from "catboost/libs/eval_result/eval_helpers.h":
    cdef TVector[TVector[double]] PrepareEval(
        const EPredictionType predictionType,
        const TMaybe[TString]& lossFunctionName,
        const TVector[TVector[double]]& approx,
        int threadCount
    ) nogil except +ProcessException

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
        int threadCount,
        EPreCalcShapValues mode,
        int logPeriod,
        ECalcTypeShapValues calcType
    ) nogil except +ProcessException

    cdef TVector[TVector[TVector[double]]] GetFeatureImportancesMulti(
        const EFstrType type,
        const TFullModel& model,
        const TDataProviderPtr dataset,
        int threadCount,
        EPreCalcShapValues mode,
        int logPeriod,
        ECalcTypeShapValues calcType
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

cdef extern from "catboost/libs/helpers/wx_test.h" nogil:
    cdef cppclass TWxTestResult:
        double WPlus
        double WMinus
        double PValue
    cdef TWxTestResult WxTest(const TVector[double]& baseline, const TVector[double]& test) nogil except +ProcessException

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

    cdef TJsonValue GetPlainJsonWithAllOptions(
        const TFullModel& model,
        bool_t hasCatFeatures,
        bool_t hasTextFeatures
    ) nogil except +ProcessException

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


cpdef run_atexit_finalizers():
    ManualRunAtExitFinalizers()


if not getattr(sys, "is_standalone_binary", False) and platform.system() == 'Windows':
    atexit.register(run_atexit_finalizers)


cdef inline float _FloatOrNan(object obj) except *:
    try:
        return float(obj)
    except:
        pass

    cdef float res
    if obj is None:
        res = _FLOAT_NAN
    elif isinstance(obj, string_types + (np.string_,)):
        res = _FloatOrNanFromString(to_arcadia_string(obj))
    else:
        raise TypeError("Cannot convert obj {} to float".format(str(obj)))
    return res

cdef TString _MetricGetDescription(void* customData) except * with gil:
    cdef metricObject = <object>customData
    name = metricObject.__class__.__name__
    if PY3:
        name = name.encode()
    return TString(<const char*>name)

cdef bool_t _MetricIsMaxOptimal(void* customData) except * with gil:
    cdef metricObject = <object>customData
    return metricObject.is_max_optimal()

cdef double _MetricGetFinalError(const TMetricHolder& error, void *customData) except * with gil:
    # TODO(nikitxskv): use error.Stats for custom metrics.
    cdef metricObject = <object>customData
    return metricObject.get_final_error(error.Stats[0], error.Stats[1])


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

cdef TMetricHolder _MetricEval(
    const TVector[TVector[double]]& approx,
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

    approxes = [_DoubleArrayWrapper.create(approx[i].data() + begin, end - begin) for i in xrange(approx.size())]
    targets = _FloatArrayWrapper.create(target.data() + begin, end - begin)

    if weight.size() == 0:
        weights = None
    else:
        weights = _FloatArrayWrapper.create(weight.data() + begin, end - begin)

    try:
        error, weight_ = metricObject.evaluate(approxes, targets, weights)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    holder.Stats[0] = error
    holder.Stats[1] = weight_
    return holder

cdef TMetricHolder _MultiregressionMetricEval(
    TConstArrayRef[TVector[double]] approx,
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

    approxes = [_DoubleArrayWrapper.create(approx[i].data() + begin, end - begin) for i in xrange(approx.size())]
    targets = [_FloatArrayWrapper.create(target[i].data() + begin, end - begin) for i in xrange(target.size())]

    if weight.size() == 0:
        weights = None
    else:
        weights = _FloatArrayWrapper.create(weight.data() + begin, end - begin)

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

    approx = _DoubleArrayWrapper.create(approxes, count)
    target = _FloatArrayWrapper.create(targets, count)

    if weights:
        weight = _FloatArrayWrapper.create(weights, count)
    else:
        weight = None

    try:
        result = objectiveObject.calc_ders_range(approx, target, weight)
    except:
        errorMessage = to_arcadia_string(traceback.format_exc())
        with nogil:
            ThrowCppExceptionWithMessage(errorMessage)

    index = 0
    for der1, der2 in result:
        ders[index].Der1 = der1
        ders[index].Der2 = der2
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

    approxes = _DoubleArrayWrapper.create(approx.data(), approx.size())

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

cdef void _ObjectiveCalcDersMultiRegression(
    TConstArrayRef[double] approx,
    TConstArrayRef[float] target,
    float weight,
    TVector[double]* ders,
    THessianInfo* der2,
    void* customData
) with gil:
    cdef objectiveObject = <object>(customData)
    cdef TString errorMessage

    approxes = _DoubleArrayWrapper.create(approx.data(), approx.size())
    targetes = _FloatArrayWrapper.create(target.data(), target.size())

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


# customGenerator should have method rvs()
cdef TCustomRandomDistributionGenerator _BuildCustomRandomDistributionGenerator(object customGenerator):
    cdef TCustomRandomDistributionGenerator descriptor
    descriptor.CustomData = <void*>customGenerator
    descriptor.EvalFunc = &_RandomDistGen
    return descriptor

cdef TCustomMetricDescriptor _BuildCustomMetricDescriptor(object metricObject):
    cdef TCustomMetricDescriptor descriptor
    descriptor.CustomData = <void*>metricObject
    if (issubclass(metricObject.__class__, MultiRegressionCustomMetric)):
        descriptor.EvalMultiregressionFunc = &_MultiregressionMetricEval
    else:
        descriptor.EvalFunc = &_MetricEval
    descriptor.GetDescriptionFunc = &_MetricGetDescription
    descriptor.IsMaxOptimalFunc = &_MetricIsMaxOptimal
    descriptor.GetFinalErrorFunc = &_MetricGetFinalError
    return descriptor

cdef TCustomObjectiveDescriptor _BuildCustomObjectiveDescriptor(object objectiveObject):
    cdef TCustomObjectiveDescriptor descriptor
    descriptor.CustomData = <void*>objectiveObject
    descriptor.CalcDersRange = &_ObjectiveCalcDersRange
    descriptor.CalcDersMultiRegression = &_ObjectiveCalcDersMultiRegression
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


cdef class _PreprocessParams:
    cdef TJsonValue tree
    cdef TMaybe[TCustomObjectiveDescriptor] customObjectiveDescriptor
    cdef TMaybe[TCustomMetricDescriptor] customMetricDescriptor
    def __init__(self, dict params):
        eval_metric = params.get("eval_metric")
        objective = params.get("loss_function")

        is_custom_eval_metric = eval_metric is not None and not isinstance(eval_metric, string_types)
        is_custom_objective = objective is not None and not isinstance(objective, string_types)

        devices = params.get('devices')
        if devices is not None and isinstance(devices, list):
            params['devices'] = ':'.join(map(str, devices))

        if 'verbose' in params:
            params['verbose'] = int(params['verbose'])

        params_to_json = params

        if is_custom_objective or is_custom_eval_metric:
            if params.get("task_type") == "GPU":
                raise CatBoostError("User defined loss functions and metrics are not supported for GPU")
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
                params_to_json[k] = "PythonUserDefinedPerObject"

        if params_to_json.get("loss_function") == "PythonUserDefinedPerObject":
            self.customObjectiveDescriptor = _BuildCustomObjectiveDescriptor(params["loss_function"])
            if (issubclass(params["loss_function"].__class__, MultiRegressionCustomObjective)):
                params_to_json["loss_function"] = "PythonUserDefinedMultiRegression"

        if params_to_json.get("eval_metric") == "PythonUserDefinedPerObject":
            self.customMetricDescriptor = _BuildCustomMetricDescriptor(params["eval_metric"])
            if (issubclass(params["eval_metric"].__class__, MultiRegressionCustomMetric)):
                params_to_json["eval_metric"] = "PythonUserDefinedMultiRegression"

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
    cdef type s_type = type(s)
    if len(s) == 0:
        return TString()
    if s_type is unicode:
        # Fast path for most common case(s).
        tmp = (<unicode>s).encode('utf8')
        return TString(<const char*>tmp, len(tmp))
    elif s_type is bytes:
        return TString(<const char*>s, len(s))

    if PY3 and hasattr(s, 'encode'):
        # encode to the specific encoding used inside of the module
        bytes_s = s.encode()
    else:
        bytes_s = s
    return TString(<const char*>&bytes_s[0], len(bytes_s))

cdef to_native_str(binary):
    if PY3 and hasattr(binary, 'decode'):
        return binary.decode()
    return binary

cdef all_string_types_plus_bytes = string_types + (bytes,)

cdef _npstring_ = np.string_
cdef _npint32 = np.int32
cdef _npint64 = np.int64
cdef _npuint32 = np.uint32
cdef _npuint64 = np.uint64
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
    if obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npstring_:
        bytes_string_buf_representation[0] = to_arcadia_string(id_object)
    elif obj_type is int or obj_type is long or obj_type is _npint32 or obj_type is _npint64:
        bytes_string_buf_representation[0] = ToString[i64](<i64>id_object)
    elif obj_type is _npuint32 or obj_type is _npuint64:
        bytes_string_buf_representation[0] = ToString[ui64](<ui64>id_object)
    elif obj_type is float or obj_type is _npfloat32 or obj_type is _npfloat64:
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


cdef TFeaturesLayout* _init_features_layout(data, cat_features, text_features, feature_names) except*:
    cdef TVector[ui32] cat_features_vector
    cdef TVector[ui32] text_features_vector
    cdef TVector[TString] feature_names_vector
    cdef bool_t all_features_are_sparse

    if isinstance(data, FeaturesData):
        feature_count = data.get_feature_count()
        cat_features = [i for i in range(data.get_num_feature_count(), feature_count)]
        feature_names = data.get_feature_names()
    else:
        feature_count = np.shape(data)[1]

    list_to_vector(cat_features, &cat_features_vector)
    list_to_vector(text_features, &text_features_vector)

    if feature_names is not None:
        for feature_name in feature_names:
            feature_names_vector.push_back(to_arcadia_string(str(feature_name)))

    all_features_are_sparse = False
    if isinstance(data, SPARSE_MATRIX_TYPES):
        all_features_are_sparse = True

    return new TFeaturesLayout(
        <ui32>feature_count,
        cat_features_vector,
        text_features_vector,
        feature_names_vector,
        all_features_are_sparse)

cdef TVector[bool_t] _get_is_feature_type_mask(const TFeaturesLayout* featuresLayout, EFeatureType featureType) except *:
    cdef TVector[bool_t] mask
    mask.resize(featuresLayout.GetExternalFeatureCount(), False)

    cdef ui32 idx
    for idx in range(featuresLayout.GetExternalFeatureCount()):
        if featuresLayout[0].GetExternalFeatureType(idx) == featureType:
            mask[idx] = True

    return mask

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

    cdef Py_ITypedSequencePtr py_num_factor_data
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
    bool_t [:] is_cat_feature_mask,
    bool_t [:] is_text_feature_mask,
    Py_FeaturesOrderBuilderVisitor py_builder_visitor
):

    """
        older buffer interface is used instead of memory views because of
        https://github.com/cython/cython/issues/1772, https://github.com/cython/cython/issues/2485
    """

    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    py_builder_visitor.get_raw_features_order_data_visitor(&builder_visitor)

    cdef ui32 doc_count = <ui32>(feature_values.shape[0])
    cdef ui32 feature_count = <ui32>(feature_values.shape[1])

    cdef Py_ITypedSequencePtr py_num_factor_data
    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TVector[TString] string_factor_data
    cdef ui32 doc_idx

    cdef ui32 flat_feature_idx

    string_factor_data.reserve(doc_count)

    for flat_feature_idx in range(feature_count):
        if is_cat_feature_mask[flat_feature_idx] or is_text_feature_mask[flat_feature_idx]:
            string_factor_data.clear()
            for doc_idx in range(doc_count):
                string_factor_data.push_back(ToString(feature_values[doc_idx, flat_feature_idx]))
            if is_cat_feature_mask[flat_feature_idx]:
                builder_visitor[0].AddCatFeature(flat_feature_idx, <TConstArrayRef[TString]>string_factor_data)
            else:
                builder_visitor[0].AddTextFeature(flat_feature_idx, <TConstArrayRef[TString]>string_factor_data)
        else:
            py_num_factor_data = make_non_owning_type_cast_array_holder(feature_values[:,flat_feature_idx])
            py_num_factor_data.get_result(&num_factor_data)
            builder_visitor[0].AddFloatFeature(flat_feature_idx, num_factor_data)


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

    cdef Py_ITypedSequencePtr py_num_factor_data

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
    if obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npstring_:
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
    const TFeaturesLayout* features_layout,
    IRawFeaturesOrderDataVisitor* builder_visitor
):
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
    cdef TVector[bool_t] is_text_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)
    cdef ui32 doc_count = data_frame.shape[0]

    cdef TString factor_string

    cdef ITypedSequencePtr[np.float32_t] num_factor_data

    cdef TVector[TString] string_factor_data

    # this buffer for categorical processing is here to avoid reallocations in
    # _set_features_order_data_pd_data_frame_categorical_column

    # array of [dst_value_for_cateory0, dst_value_for_category1 ...]
    cdef TVector[ui32] categories_as_hashed_cat_values

    cdef ui32 doc_idx
    cdef ui32 flat_feature_idx
    cdef np.ndarray column_values # for columns that are not Sparse or Categorical

    string_factor_data.reserve(doc_count)

    new_data_holders = []
    for flat_feature_idx, (column_name, column_data) in enumerate(data_frame.iteritems()):
        if isinstance(column_data.dtype, pd.SparseDtype):
            new_data_holders += _set_features_order_data_pd_data_frame_sparse_column(
                doc_count,
                flat_feature_idx,
                is_cat_feature_mask[flat_feature_idx],
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
            column_values = column_data.values
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
    IRawObjectsOrderDataVisitor* builder_visitor
):
    if (num_feature_values is None) and (cat_feature_values is None):
        raise CatBoostError('both num_feature_values and cat_feature_values are empty')

    cdef ui32 doc_count = <ui32>(
        num_feature_values.shape[0] if num_feature_values is not None else cat_feature_values.shape[0]
    )

    cdef ui32 num_feature_count = <ui32>(num_feature_values.shape[1] if num_feature_values is not None else 0)
    cdef ui32 cat_feature_count = <ui32>(cat_feature_values.shape[1] if cat_feature_values is not None else 0)

    cdef ui32 doc_idx
    cdef ui32 num_feature_idx
    cdef ui32 cat_feature_idx

    cdef ui32 dst_feature_idx
    for doc_idx in range(doc_count):
        dst_feature_idx = <ui32>0
        for num_feature_idx in range(num_feature_count):
            builder_visitor[0].AddFloatFeature(
                doc_idx,
                dst_feature_idx,
                num_feature_values[doc_idx, num_feature_idx]
            )
            dst_feature_idx += 1
        for cat_feature_idx in range(cat_feature_count):
            builder_visitor[0].AddCatFeature(
                doc_idx,
                dst_feature_idx,
                <TStringBuf>to_arcadia_string(cat_feature_values[doc_idx, cat_feature_idx])
            )
            dst_feature_idx += 1

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
            'Invalid value for cat_feature[{doc_idx},{feature_idx)]={value}'
            +' cat_features must be integer or string, real number values and NaN values'
            +' should be converted to string'.format(doc_idx=doc_idx, feature_idx=feature_idx, value=value))
    else:
        factor_string_buf[0] = ToString[i64](<i64>value)

cdef _add_single_feature_value_from_scipy_sparse(
    int doc_idx,
    int feature_idx,
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
                        feature_block_start_idx + feature_in_block_idx,
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
        feature_idx = col[nonzero_idx]
        value = data[nonzero_idx]
        _add_single_feature_value_from_scipy_sparse(
            doc_idx,
            feature_idx,
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
    Py_ObjectsOrderBuilderVisitor py_builder_visitor
):
    cdef int doc_count = indptr.shape[0] - 1
    cdef IRawObjectsOrderDataVisitor * builder_visitor = py_builder_visitor.builder_visitor

    if doc_count == 0:
        return

    cdef TString factor_string_buf
    cdef int nonzero_elements_idx
    cdef int doc_idx
    cdef int feature_idx
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(py_builder_visitor.features_layout, EFeatureType_Categorical)

    cdef bool_t is_float_value = False

    if (numpy_num_dtype is np.float32_t) or (numpy_num_dtype is np.float64_t):
        is_float_value = True

    cdef int nonzero_begin = 0
    cdef int nonzero_end = 0
    for doc_idx in xrange(doc_count):
        nonzero_begin = indptr[doc_idx]
        nonzero_end = indptr[doc_idx + 1]
        for nonzero_elements_idx in xrange(nonzero_begin, nonzero_end, 1):
            feature_idx = indices[nonzero_elements_idx]
            value = data[nonzero_elements_idx]
            _add_single_feature_value_from_scipy_sparse(
                doc_idx,
                feature_idx,
                value,
                is_float_value,
                <TConstArrayRef[bool_t]>is_cat_feature_mask,
                & factor_string_buf,
                builder_visitor
            )

cdef _set_data_from_scipy_lil_sparse(
    data,
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
            feature_idx = row_indices[nonzero_column_idx]
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
    const TFeaturesLayout * features_layout,
    Py_ObjectsOrderBuilderVisitor py_builder_visitor
):
    cdef IRawObjectsOrderDataVisitor * builder_visitor = py_builder_visitor.builder_visitor
    _set_cat_features_default_values_for_scipy_sparse(features_layout, builder_visitor)

    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
    cdef TVector[bool_t] is_text_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)
    if np.any(is_text_feature_mask):
        raise CatBoostError('Text features reading is not supported in sparse matrix format')

    if isinstance(data, scipy.sparse.bsr_matrix):
        _set_data_from_scipy_bsr_sparse(
            data,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )
    elif isinstance(data, scipy.sparse.coo_matrix):
        _set_data_from_scipy_coo_sparse(
            data.data,
            data.row,
            data.col,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )
    elif isinstance(data, scipy.sparse.csr_matrix):
        _set_data_from_scipy_csr_sparse(
            data.data,
            data.indices,
            data.indptr,
            py_builder_visitor
        )
    elif isinstance(data, scipy.sparse.dok_matrix):
        coo_matrix = data.tocoo()
        _set_data_from_scipy_coo_sparse(
            coo_matrix.data,
            coo_matrix.row,
            coo_matrix.col,
            <TConstArrayRef[bool_t]>is_cat_feature_mask,
            builder_visitor
        )
    elif isinstance(data, scipy.sparse.lil_matrix):
        _set_data_from_scipy_lil_sparse(
            data,
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
    Py_FeaturesOrderBuilderVisitor py_builder_visitor
):
    cdef IRawFeaturesOrderDataVisitor* builder_visitor
    py_builder_visitor.get_raw_features_order_data_visitor(&builder_visitor)

    cdef const TFeaturesLayout* features_layout
    py_builder_visitor.get_features_layout(&features_layout)

    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)

    cdef np.float32_t float_default_value = 0.0
    cdef TString cat_default_value = "0"

    cdef ui32 feature_count = indptr.shape[0] - 1
    cdef ui32 feature_idx
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

    for feature_idx in xrange(feature_count):
        feature_nonzero_count = indptr[feature_idx + 1] - indptr[feature_idx]
        new_data_holders += get_canonical_type_indexing_array(
            np.asarray(indices[indptr[feature_idx]:indptr[feature_idx + 1]]),
            &feature_indices_holder
        )

        if is_cat_feature_mask[feature_idx]:
            cat_feature_values.clear()
            indptr_begin = indptr[feature_idx]
            indptr_end = indptr[feature_idx + 1]
            for data_idx in range(indptr_begin, indptr_end, 1):
                value = data[data_idx]
                _get_categorical_feature_value_from_scipy_sparse(
                    indices[data_idx],
                    feature_idx,
                    value,
                    is_float_value,
                    &factor_string_buf
                )

                cat_feature_values.push_back(factor_string_buf)

            builder_visitor[0].AddCatFeature(
                feature_idx,
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
                feature_idx,
                np.asarray(data[indptr[feature_idx]:indptr[feature_idx + 1]]),
                &num_factor_data
            )
            builder_visitor[0].AddFloatFeature(
                feature_idx,
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
            py_builder_visitor
        )

    return new_data_holders

cdef _set_data_from_generic_matrix(
    data,
    const TFeaturesLayout* features_layout,
    IRawObjectsOrderDataVisitor* builder_visitor
):
    data_shape = np.shape(data)
    cdef int doc_count = data_shape[0]
    cdef int feature_count = data_shape[1]

    if doc_count == 0:
        return

    cdef TString factor_strbuf
    cdef int doc_idx
    cdef int feature_idx
    cdef int cat_feature_idx

    cdef TVector[bool_t] is_cat_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
    cdef TVector[bool_t] is_text_feature_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)

    for doc_idx in xrange(doc_count):
        doc_data = data[doc_idx]
        for feature_idx in xrange(feature_count):
            factor = doc_data[feature_idx]
            if is_cat_feature_mask[feature_idx]:
                get_cat_factor_bytes_representation(
                    doc_idx,
                    feature_idx,
                    factor,
                    &factor_strbuf
                )
                builder_visitor[0].AddCatFeature(doc_idx, feature_idx, <TStringBuf>factor_strbuf)
            elif is_text_feature_mask[feature_idx]:
                get_text_factor_bytes_representation(
                    doc_idx,
                    feature_idx,
                    factor,
                    &factor_strbuf
                )
                builder_visitor[0].AddTextFeature(doc_idx, feature_idx, <TStringBuf>factor_strbuf)
            else:
                builder_visitor[0].AddFloatFeature(
                    doc_idx,
                    feature_idx,
                    get_float_feature(doc_idx, feature_idx, factor)
                )

cdef _set_data(data, const TFeaturesLayout* features_layout, Py_ObjectsOrderBuilderVisitor py_builder_visitor):
    if isinstance(data, FeaturesData):
        _set_data_np(data.num_feature_data, data.cat_feature_data, py_builder_visitor.builder_visitor)
    elif isinstance(data, np.ndarray) and data.dtype == np.float32:
        _set_data_np(data, None, py_builder_visitor.builder_visitor)
    elif isinstance(data, SPARSE_MATRIX_TYPES):
        _set_objects_order_data_scipy_sparse_matrix(data, features_layout, py_builder_visitor)
    else:
        _set_data_from_generic_matrix(data, features_layout, py_builder_visitor.builder_visitor)


cdef TString obj_to_arcadia_string(obj) except *:
    INT64_MIN = -9223372036854775808
    INT64_MAX =  9223372036854775807
    cdef type obj_type = type(obj)

    if obj_type is float or obj_type is _npfloat32 or obj_type is _npfloat64:
        return ToString[double](<double>obj)
    elif ((obj_type is int or obj_type is long) and (INT64_MIN <= obj <= INT64_MAX)) or obj_type is _npint32 or obj_type is _npint64:
        return ToString[i64](<i64>obj)
    elif obj_type is str or obj_type is unicode or obj_type is bytes or obj_type is _npstring_:
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
        if isinstance(label[0][0], numbers.Number):
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
        cdef Py_ITypedSequencePtr py_num_target_data
        cdef ITypedSequencePtr[np.float32_t] num_target_data
        cdef np.ndarray target_array
        cdef TVector[TString] string_target_data
        cdef ui32 object_count = len(label)
        cdef ui32 target_count = len(label[0])
        cdef ui32 target_idx
        cdef ui32 object_idx

        self.target_type = type(label[0][0])
        if isinstance(label[0][0], numbers.Number):
            self.__target_data_holders = []
            for target_idx in range(target_count):
                if isinstance(label, np.ndarray):
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


    cpdef _read_pool(self, pool_file, cd_file, pairs_file, feature_names_file, delimiter, bool_t has_header, int thread_count, dict quantization_params):
        cdef TPathWithScheme pool_file_path
        pool_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(pool_file), TStringBuf(<char*>'dsv'))

        cdef TPathWithScheme pairs_file_path
        if len(pairs_file):
            pairs_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(pairs_file), TStringBuf(<char*>'dsv'))

        cdef TPathWithScheme feature_names_file_path
        if len(feature_names_file):
            feature_names_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(feature_names_file), TStringBuf(<char*>'dsv'))

        cdef TColumnarPoolFormatParams columnarPoolFormatParams
        columnarPoolFormatParams.DsvFormat.HasHeader = has_header
        columnarPoolFormatParams.DsvFormat.Delimiter = ord(delimiter)
        if len(cd_file):
            columnarPoolFormatParams.CdFilePath = TPathWithScheme(<TStringBuf>to_arcadia_string(cd_file), TStringBuf(<char*>'dsv'))

        thread_count = UpdateThreadCount(thread_count)

        cdef TVector[ui32] emptyIntVec
        cdef TPathWithScheme input_borders_file_path
        if quantization_params is not None:
            input_borders = quantization_params.pop("input_borders", None)
            block_size = quantization_params.pop("dev_block_size", None)
            prep_params = _PreprocessParams(quantization_params)
            if input_borders:
                input_borders_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(input_borders), TStringBuf(<char*>'dsv'))
            self.__pool = ReadAndQuantizeDataset(
                pool_file_path,
                pairs_file_path,
                TPathWithScheme(),
                TPathWithScheme(),
                TPathWithScheme(),
                feature_names_file_path,
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
                columnarPoolFormatParams,
                emptyIntVec,
                EObjectsOrder_Undefined,
                thread_count,
                False
            )
        self.__data_holders = None # free previously used resources
        self.target_type = str


    cdef _init_features_order_layout_pool(
        self,
        data,
        const TDataMetaInfo& data_meta_info,
        label,
        pairs,
        weight,
        group_id,
        group_weight,
        subgroup_id,
        pairs_weight,
        baseline,
        thread_count):

        cdef TFeaturesLayout* features_layout = data_meta_info.FeaturesLayout.Get()
        cdef Py_FeaturesOrderBuilderVisitor py_builder_visitor = Py_FeaturesOrderBuilderVisitor(thread_count)
        cdef IRawFeaturesOrderDataVisitor* builder_visitor = py_builder_visitor.builder_visitor
        py_builder_visitor.set_features_layout(data_meta_info.FeaturesLayout.Get())

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
                features_layout,
                builder_visitor
            )
        elif isinstance(data, scipy.sparse.spmatrix):
            new_data_holders = _set_features_order_data_scipy_sparse_matrix(
                data,
                features_layout,
                py_builder_visitor
            )
        elif isinstance(data, np.ndarray):
            if data_meta_info.FeaturesLayout.Get()[0].GetFloatFeatureCount():
                new_data_holders = data

            cat_features_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Categorical)
            text_features_mask = _get_is_feature_type_mask(features_layout, EFeatureType_Text)

            _set_features_order_data_ndarray(
                data,
                <bool_t[:features_layout[0].GetExternalFeatureCount()]>cat_features_mask.data(),
                <bool_t[:features_layout[0].GetExternalFeatureCount()]>text_features_mask.data(),
                py_builder_visitor
            )

            # prevent inadvent modification of pool data
            data.setflags(write=0)
        else:
            raise CatBoostError(
                '[Internal error] wrong data type for _init_features_order_layout_pool: ' + type(data)
            )

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

        builder_visitor[0].Finish()

        self.__pool = py_builder_visitor.data_provider_builder.Get()[0].GetResult()
        self.__data_holders = new_data_holders


    cdef _init_objects_order_layout_pool(
        self,
        data,
        const TDataMetaInfo& data_meta_info,
        label,
        pairs,
        weight,
        group_id,
        group_weight,
        subgroup_id,
        pairs_weight,
        baseline,
        thread_count):

        cdef Py_ObjectsOrderBuilderVisitor py_builder_visitor = Py_ObjectsOrderBuilderVisitor(thread_count)
        cdef IRawObjectsOrderDataVisitor* builder_visitor = py_builder_visitor.builder_visitor
        py_builder_visitor.set_features_layout(data_meta_info.FeaturesLayout.Get())

        self.__data_holder = None # free previously used resources

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

        _set_data(data, data_meta_info.FeaturesLayout.Get(), py_builder_visitor)

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

        builder_visitor[0].Finish()

        self.__pool = py_builder_visitor.data_provider_builder.Get()[0].GetResult()


    cpdef _init_pool(self, data, label, cat_features, text_features, pairs, weight, group_id, group_weight,
                     subgroup_id, pairs_weight, baseline, feature_names, thread_count):
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
        data_meta_info.HasTimestamp = False
        data_meta_info.HasPairs = pairs is not None

        data_meta_info.FeaturesLayout = _init_features_layout(data, cat_features, text_features, feature_names)

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
                data_meta_info,
                label,
                pairs,
                weight,
                group_id,
                group_weight,
                subgroup_id,
                pairs_weight,
                baseline,
                thread_count
            )
        else:
            self._init_objects_order_layout_pool(
                data,
                data_meta_info,
                label,
                pairs,
                weight,
                group_id,
                group_weight,
                subgroup_id,
                pairs_weight,
                baseline,
                thread_count
            )

    cpdef _save(self, fname):
        cdef TString file_name = to_arcadia_string(fname)
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
        cdef TConstArrayRef[TPair] old_pairs = self.__pool.Get()[0].RawTargetData.GetPairs()
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

        if (_input_borders):
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
        return self.__pool.Get()[0].RawTargetData.GetPairs().size()

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
        TLocalExecutor* local_executor,
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
        cdef TLocalExecutor local_executor
        cdef TFeaturesLayout* features_layout =self.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()
        cdef TRawObjectsDataProvider* raw_objects_data_provider = dynamic_cast_to_TRawObjectsDataProvider(
            self.__pool.Get()[0].ObjectsData.Get()
        )
        if not raw_objects_data_provider:
            raise CatBoostError('Pool does not have raw features data, only quantized')
        if features_layout[0].GetExternalFeatureCount() != features_layout[0].GetFloatFeatureCount():
            raise CatBoostError('Pool has non-numeric features, get_features supports only numeric features')

        local_executor.RunAdditionalThreads(thread_count - 1)

        data = np.empty(self.shape, dtype=np.float32)

        for factor in range(self.num_col()):
            self._get_feature(raw_objects_data_provider, factor, &local_executor, data)

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


    cpdef save_quantization_borders(self, output_file):
        """
        Save file with borders used in numeric features quantization.
        File format is described here: https://catboost.ai/docs/concepts/input-data_custom-borders.html

        Parameters
        ----------
        output_file : string
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

        cdef TString fname = to_arcadia_string(output_file)
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
    input_borders_str = to_arcadia_string(_input_borders)
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


cdef class _CatBoost:
    cdef TFullModel* __model
    cdef TVector[TEvalResult*] __test_evals
    cdef TMetricsAndTimeLeftHistory __metrics_history
    cdef THolder[TLearnProgress] __cached_learn_progress

    def __cinit__(self):
        self.__model = new TFullModel()

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
        metrics_evals = defaultdict(functools.partial(defaultdict, list))
        iteration_count = self.__metrics_history.LearnMetricsHistory.size()
        for iteration_num in range(iteration_count):
            for metric, value in self.__metrics_history.LearnMetricsHistory[iteration_num]:
                metrics_evals["learn"][to_native_str(metric)].append(value)

        if not self.__metrics_history.TestMetricsHistory.empty():
            test_count = 0
            for i in range(iteration_count):
                test_count = max(test_count, self.__metrics_history.TestMetricsHistory[i].size())
            for iteration_num in range(iteration_count):
                for test_index in range(self.__metrics_history.TestMetricsHistory[iteration_num].size()):
                    eval_set_name = "validation"
                    if test_count > 1:
                        eval_set_name += "_" + str(test_index)
                    for metric, value in self.__metrics_history.TestMetricsHistory[iteration_num][test_index]:
                        metrics_evals[eval_set_name][to_native_str(metric)].append(value)
        return {k: dict(v) for k, v in iteritems(metrics_evals)}

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
        return not self.__model.ModelTrees.Get().GetLeafWeights().empty()

    cpdef _get_cat_feature_indices(self):
        cdef TConstArrayRef[TCatFeature] arrayView = self.__model.ModelTrees.Get().GetCatFeatures()
        return [feature.Position.FlatIndex for feature in arrayView]

    cpdef _get_text_feature_indices(self):
        cdef TConstArrayRef[TTextFeature] arrayView = self.__model.ModelTrees.Get().GetTextFeatures()
        return [feature.Position.FlatIndex for feature in arrayView]

    cpdef _get_float_feature_indices(self):
        cdef TConstArrayRef[TFloatFeature] arrayView = self.__model.ModelTrees.Get().GetFloatFeatures()
        return [feature.Position.FlatIndex for feature in arrayView]

    cpdef _get_borders(self):
        cdef TConstArrayRef[TFloatFeature] arrayView = self.__model.ModelTrees.Get().GetFloatFeatures()
        return dict([(feature.Position.FlatIndex, feature.Borders) for feature in arrayView])

    cpdef _base_predict(self, _PoolBase pool, str prediction_type, int ntree_start, int ntree_end, int thread_count, bool_t verbose):
        cdef TVector[TVector[double]] pred
        cdef EPredictionType predictionType = string_to_prediction_type(prediction_type)
        thread_count = UpdateThreadCount(thread_count);
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
            to_arcadia_string(result_dir),
            to_arcadia_string(tmp_dir)
        )
        cdef TVector[TString] metric_names = GetMetricNames(dereference(self.__model), metricDescriptions)
        return metrics, [to_native_str(name) for name in metric_names]

    cpdef _get_loss_function_name(self):
        return self.__model.GetLossFunctionName()

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

    cpdef _calc_fstr(self, type_name, _PoolBase pool, int thread_count, int verbose, shap_mode_name, interaction_indices,
                     shap_calc_type):
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

        cdef EFstrType fstr_type = string_to_fstr_type(type_name)
        cdef EPreCalcShapValues shap_mode = string_to_shap_mode(shap_mode_name)
        cdef TMaybe[pair[int, int]] pair_of_features

        if shap_calc_type == 'Exact':
            assert dereference(self.__model).IsOblivious(), "'Exact' calculation type is supported only for symmetric trees."
        cdef ECalcTypeShapValues calc_type = string_to_calc_type(shap_calc_type)

        if type_name == 'ShapValues' and dereference(self.__model).GetDimensionsCount() > 1:
            with nogil:
                fstr_multi = GetFeatureImportancesMulti(
                    fstr_type,
                    dereference(self.__model),
                    dataProviderPtr,
                    thread_count,
                    shap_mode,
                    verbose,
                    calc_type
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
                    thread_count,
                    shap_mode,
                    verbose,
                    calc_type
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
        scores = _2d_vector_of_double_to_np_array(ostr.Scores)
        if to_arcadia_string(ostr_type) == to_arcadia_string('Average'):
            indices = indices[0]
            scores = scores[0]
        return indices, scores

    cpdef _base_shrink(self, int ntree_start, int ntree_end):
        self.__model.Truncate(ntree_start, ntree_end)

    cpdef _get_scale_and_bias(self):
        cdef TScaleAndBias scale_and_bias = dereference(self.__model).GetScaleAndBias()
        return scale_and_bias.Scale, scale_and_bias.Bias

    cpdef _set_scale_and_bias(self, scale, bias):
        cdef TScaleAndBias scale_and_bias
        scale_and_bias.Scale = scale
        scale_and_bias.Bias = bias
        dereference(self.__model).SetScaleAndBias(scale_and_bias)

    cpdef _is_oblivious(self):
        return self.__model.IsOblivious()

    cpdef _base_drop_unused_features(self):
        self.__model.ModelTrees.GetMutable().DropUnusedFeatures()

    cpdef _load_model(self, model_file, format):
        cdef TFullModel tmp_model
        cdef EModelType modelType = string_to_model_type(format)
        tmp_model = ReadModel(to_arcadia_string(model_file), modelType)
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
            to_arcadia_string(output_file),
            modelType,
            to_arcadia_string(export_parameters),
            False,
            &feature_id if pool else <TVector[TString]*>nullptr,
            &cat_features_hash_to_string if pool else <THashMap[ui32, TString]*>nullptr
        )

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
        try:
            params_json = to_native_str(self.__model.ModelInfo["params"])
            params_dict = loads(params_json)
            flat_params = params_dict["flat_params"]
            params = {str(key): value for key, value in iteritems(flat_params)}
            return params
        except Exception as e:
            return {}

    cpdef _get_plain_params(self):
        hasCatFeatures = len(self._get_cat_feature_indices()) != 0
        hasTextFeatures = len(self._get_text_feature_indices()) != 0
        cdef TJsonValue plainOptions = GetPlainJsonWithAllOptions(
            dereference(self.__model),
            hasCatFeatures,
            hasTextFeatures
        )
        return loads(to_native_str(WriteTJsonValue(plainOptions)))

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

    def _get_metadata_wrapper(self):
        return _MetadataHashProxy(self)

    def _get_feature_names(self):
        return [to_native_str(s) for s in GetModelUsedFeaturesNames(dereference(self.__model))]

    def _get_class_labels(self):
        return _get_model_class_labels(self.__model[0])

    cpdef _sum_models(self, models, weights, ctr_merge_policy):
        cdef TVector[TFullModel_const_ptr] models_vector
        cdef TVector[double] weights_vector
        cdef ECtrTableMergePolicy merge_policy
        if not TryFromString[ECtrTableMergePolicy](to_arcadia_string(ctr_merge_policy), merge_policy):
            raise CatBoostError("Unknown ctr table merge policy {}".format(ctr_merge_policy))
        assert(len(models) == len(weights))
        for model_id in range(len(models)):
            models_vector.push_back((<_CatBoost>models[model_id]).__model)
            weights_vector.push_back(weights[model_id])
        cdef TFullModel tmp_model = SumModels(models_vector, weights_vector, merge_policy)
        self.__model.Swap(tmp_model)

    cpdef _save_borders(self, output_file):
        SaveModelBorders( to_arcadia_string(output_file), dereference(self.__model))

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
        return _constarrayref_of_double_to_np_array(self.__model.ModelTrees.Get().GetLeafValues())

    cpdef _get_leaf_weights(self):
        result = np.empty(self.__model.ModelTrees.Get().GetLeafValues().size(), dtype=_npfloat64)
        cdef size_t curr_index = 0
        cdef TConstArrayRef[double] arrayView = self.__model.ModelTrees.Get().GetLeafWeights()
        for val in arrayView:
            result[curr_index] = val
            curr_index += 1
        assert curr_index == 0 or curr_index == self.__model.ModelTrees.Get().GetLeafValues().size(), (
            "wrong number of leaf weights")
        return result

    cpdef _get_tree_leaf_counts(self):
        return _vector_of_uints_to_np_array(self.__model.ModelTrees.Get().GetTreeLeafCounts())

    cpdef _set_leaf_values(self, new_leaf_values):
        assert isinstance(new_leaf_values, np.ndarray), "expected numpy.ndarray."
        assert new_leaf_values.dtype == np.float64, "leaf values should have type np.float64 (double)."
        assert len(new_leaf_values.shape) == 1, "leaf values should be a 1d-vector."
        assert new_leaf_values.shape[0] == self.__model.ModelTrees.Get().GetLeafValues().size(), (
            "count of leaf values should be equal to the leaf count.")
        cdef TVector[double] model_leafs = new_leaf_values
        self.__model.ModelTrees.GetMutable().SetLeafValues(model_leafs)

    cpdef _set_feature_names(self, feature_names):
            cdef TVector[TString] feature_names_vector
            for value in feature_names:
                feature_names_vector.push_back(to_arcadia_string(str(value)))
            SetModelExternalFeatureNames(feature_names_vector, self.__model)

    cpdef _convert_oblivious_to_asymmetric(self):
        self.__model.ModelTrees.GetMutable().ConvertObliviousToAsymmetric()



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


cdef object _get_hash_group_id(_PoolBase pool):
    cdef TMaybeData[TConstArrayRef[TGroupId]] arr_group_ids = pool.__pool.Get()[0].ObjectsData.Get()[0].GetGroupIds()
    if arr_group_ids.Defined():
        result_group_ids = []
        for group_id in arr_group_ids.GetRef():
            result_group_ids.append(group_id)

        return result_group_ids

    return None


cdef TCustomTrainTestSubsets _make_train_test_subsets(_PoolBase pool, folds) except *:
    num_data = pool.num_row()

    if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
        raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                             "or scikit-learn splitter object with split method")

    group_info = _get_hash_group_id(pool)

    if hasattr(folds, 'split'):
        if group_info is not None:
            flatted_group = group_info
        else:
            flatted_group = np.zeros(num_data, dtype=int)
        folds = folds.split(X=np.zeros(num_data), y=pool.get_label(), groups=flatted_group)

    cdef TVector[TVector[ui32]] custom_train_subsets
    cdef TVector[TVector[ui32]] custom_test_subsets

    if group_info is None:
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
        map_group_id_to_group_number = {}
        current_num = 0
        for idx in range(len(group_info)):
            if idx == 0 or group_info[idx] != group_info[idx - 1]:
                map_group_id_to_group_number[group_info[idx]] = current_num
                current_num = current_num + 1

        for train_test in folds:
            train = train_test[0]
            test = train_test[1]

            train_group = []

            custom_train_subsets.emplace_back()

            for idx in range(len(train)):
                current_group = group_info[train[idx]]
                if idx == 0 or current_group != group_info[train[idx - 1]]:
                    custom_train_subsets.back().push_back(map_group_id_to_group_number[current_group])
                    train_group.append(map_group_id_to_group_number[current_group])

            custom_test_subsets.emplace_back()

            for idx in range(len(test)):
                current_group = group_info[test[idx]]

                if map_group_id_to_group_number[current_group] in train_group:
                    raise CatBoostError('Objects with the same group id must be in the same fold.')

                if idx == 0 or current_group != group_info[test[idx - 1]]:
                    custom_test_subsets.back().push_back(map_group_id_to_group_number[current_group])

    cdef TCustomTrainTestSubsets result
    result.first = custom_train_subsets
    result.second = custom_test_subsets

    return result


cpdef _cv(dict params, _PoolBase pool, int fold_count, bool_t inverted, int partition_random_seed,
          bool_t shuffle, bool_t stratified, bool_t as_pandas, folds, type):
    prep_params = _PreprocessParams(params)
    cdef TCrossValidationParams cvParams
    cdef TVector[TCVResult] results

    cvParams.FoldCount = fold_count
    cvParams.PartitionRandSeed = partition_random_seed
    cvParams.Shuffle = shuffle
    cvParams.Stratified = stratified

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
        return pd.DataFrame.from_dict(cv_results)
    return cv_results


cdef _convert_to_visible_labels(EPredictionType predictionType, TVector[TVector[double]] raws, int thread_count, TFullModel* model):
    cdef size_t objectCount
    cdef size_t objectIdx
    cdef TConstArrayRef[double] raws1d

    if predictionType == string_to_prediction_type('Class'):
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


cdef class _StagedPredictIterator:
    cdef TVector[double] __flatApprox
    cdef TVector[TVector[double]] __approx
    cdef TVector[TVector[double]] __pred
    cdef TFullModel* __model
    cdef TLocalExecutor __executor
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
        self.__executor.RunAdditionalThreads(self.thread_count - 1)

    cdef _initialize_model_calcer(self, TFullModel* model, _PoolBase pool):
        self.__model = model
        self.__modelCalcerOnPool = new TModelCalcerOnPool(
            dereference(self.__model),
            pool.__pool.Get()[0].ObjectsData,
            &self.__executor
        )

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
        return self._metric_descriptions[metric_description]

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
                                                            to_arcadia_string(tmp_dir), delete_temp_dir_on_exit)

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


cpdef _eval_metric_util(label_param, approx_param, metric, weight_param, group_id_param, subgroup_id_param, pairs_param, thread_count):
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

    return EvalMetricsForUtils(<TConstArrayRef[TVector[float]]>(label), approx, to_arcadia_string(metric), weight, group_id, subgroup_id, pairs, thread_count)


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


log_cout = None
log_cerr = None


cdef void _CoutLogPrinter(const char* str, size_t len) except * with gil:
    cdef bytes bytes_str = str[:len]
    log_cout.write(to_native_str(bytes_str))


cdef void _CerrLogPrinter(const char* str, size_t len) except * with gil:
    cdef bytes bytes_str = str[:len]
    log_cerr.write(to_native_str(bytes_str))


cpdef _set_logger(cout, cerr):
    global log_cout
    global log_cerr
    log_cout = cout
    log_cerr = cerr
    SetCustomLoggingFunction(&_CoutLogPrinter, &_CerrLogPrinter)


cpdef _reset_logger():
    RestoreOriginalLogger()


cpdef _configure_malloc():
    ConfigureMalloc()


cpdef _library_init():
    LibraryInit()


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


cpdef is_groupwise_metric(metric_name):
    return IsGroupwiseMetric(to_arcadia_string(metric_name))


cpdef is_multiclass_metric(metric_name):
    return IsMultiClassCompatibleMetric(to_arcadia_string(metric_name))


cpdef is_pairwise_metric(metric_name):
    return IsPairwiseMetric(to_arcadia_string(metric_name))


cpdef is_minimizable_metric(metric_name):
    return IsMinOptimal(to_arcadia_string(metric_name))


cpdef is_maximizable_metric(metric_name):
    return IsMaxOptimal(to_arcadia_string(metric_name))


cpdef get_experiment_name(ui32 feature_set_idx, ui32 fold_idx):
    cdef TString experiment_name = GetExperimentName(feature_set_idx, fold_idx)
    cdef const char* c_experiment_name_string = experiment_name.c_str()
    cpdef bytes py_experiment_name_str = c_experiment_name_string[:experiment_name.size()]
    return py_experiment_name_str


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
    ResetTraceBackend(to_arcadia_string(file))


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

include "_monoforest.pxi"
include "_text_processing.pxi"
