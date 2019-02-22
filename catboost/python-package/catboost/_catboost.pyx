# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

import six
from six import iteritems, string_types, PY3
from six.moves import range
from json import dumps, loads, JSONEncoder
from copy import deepcopy
from collections import Sequence, defaultdict
import functools

import numpy as np
cimport numpy as np

import pandas as pd

np.import_array()

cimport cython
from cython.operator cimport dereference, preincrement

from libc.math cimport isnan
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool as bool_t
from libcpp cimport nullptr
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from util.generic.hash cimport THashMap
from util.generic.maybe cimport TMaybe
from util.generic.ptr cimport THolder, TIntrusivePtr
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui8, ui32, ui64, i64
from util.string.cast cimport StrToD, TryFromString, ToString


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


cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef void ProcessException()
    cdef void SetPythonInterruptHandler() nogil
    cdef void ResetPythonInterruptHandler() nogil


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
        )

    cdef cppclass TMaybeOwningConstArrayHolder[T]:
        @staticmethod
        TMaybeOwningConstArrayHolder[T] CreateNonOwning(TConstArrayRef[T] arrayRef)

        @staticmethod
        TMaybeOwningConstArrayHolder[T] CreateOwning(
            TConstArrayRef[T] arrayRef,
            TIntrusivePtr[IResourceHolder] resourceHolder
        )


cdef extern from "catboost/libs/options/binarization_options.h" namespace "NCatboostOptions" nogil:
    cdef cppclass TBinarizationOptions:
        TBinarizationOptions(...)


cdef extern from "catboost/libs/options/enums.h":
    cdef cppclass EFeatureType:
        bool_t operator==(EFeatureType)

    cdef EFeatureType EFeatureType_Float "EFeatureType::Float"
    cdef EFeatureType EFeatureType_Categorical "EFeatureType::Categorical"


    cdef cppclass EPredictionType:
        bool_t operator==(EPredictionType)

    cdef EPredictionType EPredictionType_Class "EPredictionType::Class"
    cdef EPredictionType EPredictionType_Probability "EPredictionType::Probability"
    cdef EPredictionType EPredictionType_RawFormulaVal "EPredictionType::RawFormulaVal"


cdef extern from "catboost/libs/quantization_schema/schema.h" namespace "NCB":
    cdef cppclass TPoolQuantizationSchema:
        pass


cdef extern from "catboost/libs/data_new/features_layout.h" namespace "NCB":
    cdef cppclass TFeatureMetaInfo:
        EFeatureType Type
        TString Name
        bool_t IsIgnored
        bool_t IsAvailable

    cdef cppclass TFeaturesLayout:
        TFeaturesLayout() except +ProcessException
        TFeaturesLayout(
            const ui32 featureCount,
            const TVector[ui32]& catFeatureIndices,
            const TVector[TString]& featureId,
            const TPoolQuantizationSchema* quantizationSchema
        )  except +ProcessException

        TConstArrayRef[TFeatureMetaInfo] GetExternalFeaturesMetaInfo() except +ProcessException
        TVector[TString] GetExternalFeatureIds() except +ProcessException
        void SetExternalFeatureIds(TConstArrayRef[TString] featureIds) except +ProcessException
        EFeatureType GetExternalFeatureType(ui32 externalFeatureIdx) except +ProcessException
        ui32 GetCatFeatureCount() except +ProcessException
        ui32 GetExternalFeatureCount() except +ProcessException
        TConstArrayRef[ui32] GetCatFeatureInternalIdxToExternalIdx() except +ProcessException


cdef extern from "catboost/libs/data_new/meta_info.h" namespace "NCB":
    cdef cppclass TDataMetaInfo:
        TIntrusivePtr[TFeaturesLayout] FeaturesLayout

        bool_t HasTarget
        ui32 BaselineCount
        bool_t HasGroupId
        bool_t HasGroupWeight
        bool_t HasSubgroupIds
        bool_t HasWeights
        bool_t HasTimestamp
        bool_t HasPairs

        # ColumnsInfo is not here because it is not used for now

        ui32 GetFeatureCount() except +ProcessException

cdef extern from "catboost/libs/data_new/order.h" namespace "NCB":
    cdef cppclass EObjectsOrder:
        pass

    cdef EObjectsOrder EObjectsOrder_Ordered "NCB::EObjectsOrder::Ordered"
    cdef EObjectsOrder EObjectsOrder_RandomShuffled "NCB::EObjectsOrder::RandomShuffled"
    cdef EObjectsOrder EObjectsOrder_Undefined "NCB::EObjectsOrder::Undefined"


cdef extern from "catboost/libs/data_types/pair.h":
    cdef cppclass TPair:
        ui32 WinnerId
        ui32 LoserId
        float Weight
        TPair(ui32 winnerId, ui32 loserId, float weight) nogil except +ProcessException

cdef extern from "catboost/libs/data_types/groupid.h":
    ctypedef ui64 TGroupId
    ctypedef ui32 TSubgroupId
    cdef TGroupId CalcGroupIdFor(const TStringBuf& token) except +ProcessException
    cdef TSubgroupId CalcSubgroupIdFor(const TStringBuf& token) except +ProcessException


cdef extern from "catboost/libs/data_new/util.h" namespace "NCB":
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


cdef extern from "catboost/libs/data_new/quantized_features_info.h" namespace "NCB":
    cdef cppclass TQuantizedFeaturesInfo:
        TQuantizedFeaturesInfo(...)

    ctypedef TIntrusivePtr[TQuantizedFeaturesInfo] TQuantizedFeaturesInfoPtr


cdef extern from "catboost/libs/data_new/objects_grouping.h" namespace "NCB":
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


cdef extern from "catboost/libs/data_new/objects.h" namespace "NCB":
    cdef cppclass TObjectsDataProvider:
        ui32 GetObjectCount()
        TMaybeData[TConstArrayRef[TGroupId]] GetGroupIds()
        TMaybeData[TConstArrayRef[TSubgroupId]] GetSubgroupIds()
        TMaybeData[TConstArrayRef[ui64]] GetTimestamp()
        const THashMap[ui32, TString]& GetCatFeaturesHashToString(ui32 catFeatureIdx) except +ProcessException

    cdef cppclass TRawObjectsDataProvider(TObjectsDataProvider):
        void SetGroupIds(TConstArrayRef[TStringBuf] groupStringIds) except +ProcessException
        void SetSubgroupIds(TConstArrayRef[TStringBuf] subgroupStringIds) except +ProcessException
        TVector[float] GetFeatureDataOldFormat(ui32 flatFeatureIdx) except +ProcessException

    cdef THashMap[ui32, TString] MergeCatFeaturesHashToString(const TObjectsDataProvider& objectsData) except +ProcessException

cdef extern from *:
    TRawObjectsDataProvider* dynamic_cast_to_TRawObjectsDataProvider "dynamic_cast<NCB::TRawObjectsDataProvider*>" (TObjectsDataProvider*)


cdef extern from "catboost/libs/data_new/weights.h" namespace "NCB":
    cdef cppclass TWeights[T]:
        T operator[](ui32 idx) except +ProcessException
        ui32 GetSize()
        bool_t IsTrivial()
        TConstArrayRef[T] GetNonTrivialData() except +ProcessException


ctypedef TConstArrayRef[TConstArrayRef[float]] TBaselineArrayRef


cdef extern from "catboost/libs/data_new/target.h" namespace "NCB":
    cdef cppclass TRawTargetDataProvider:
        TMaybeData[TConstArrayRef[TString]] GetTarget()
        TMaybeData[TBaselineArrayRef] GetBaseline()
        const TWeights[float]& GetWeights()
        const TWeights[float]& GetGroupWeights()
        TConstArrayRef[TPair] GetPairs()

    cdef cppclass ETargetType:
        pass

    cdef cppclass TTargetDataSpecification:
        ETargetType Type
        TString Description

    cdef cppclass TTargetDataProvider:
        pass

ctypedef TIntrusivePtr[TTargetDataProvider] TTargetDataProviderPtr
ctypedef THashMap[TTargetDataSpecification, TTargetDataProviderPtr] TTargetDataProviders


cdef extern from "catboost/libs/data_new/data_provider.h" namespace "NCB":
    cdef cppclass TDataProviderTemplate[TTObjectsDataProvider]:
        TDataMetaInfo MetaInfo
        TIntrusivePtr[TTObjectsDataProvider] ObjectsData
        TObjectsGroupingPtr ObjectsGrouping
        TRawTargetDataProvider RawTargetData

        bool_t operator==(const TDataProviderTemplate& rhs)
        TIntrusivePtr[TDataProviderTemplate[TTObjectsDataProvider]] GetSubset(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            int threadCount
        ) except +ProcessException
        ui32 GetObjectCount()

        void SetBaseline(TBaselineArrayRef baseline) except +ProcessException
        void SetGroupIds(TConstArrayRef[TGroupId] groupIds) except +ProcessException
        void SetGroupWeights(TConstArrayRef[float] groupWeights) except +ProcessException
        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException
        void SetSubgroupIds(TConstArrayRef[TSubgroupId] subgroupIds) except +ProcessException
        void SetWeights(TConstArrayRef[float] weights) except +ProcessException

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
        TTargetDataProviders TargetData


    ctypedef TProcessedDataProviderTemplate[TObjectsDataProvider] TProcessedDataProvider
    ctypedef TIntrusivePtr[TProcessedDataProvider] TProcessedDataProviderPtr


    cdef cppclass TTrainingDataProvidersTemplate[TTObjectsDataProvider]:
        TIntrusivePtr[TProcessedDataProviderTemplate[TObjectsDataProvider]] Learn
        TVector[TIntrusivePtr[TProcessedDataProviderTemplate[TObjectsDataProvider]]] Test

    ctypedef TTrainingDataProvidersTemplate[TObjectsDataProvider] TTrainingDataProviders


cdef extern from "catboost/libs/data_util/path_with_scheme.h" namespace "NCB":
    cdef cppclass TPathWithScheme:
        TString Scheme
        TString Path
        TPathWithScheme() except +ProcessException
        TPathWithScheme(const TStringBuf& pathWithScheme, const TStringBuf& defaultScheme) except +ProcessException
        bool_t Inited() except +ProcessException

cdef extern from "catboost/libs/data_util/line_data_reader.h" namespace "NCB":
    cdef cppclass TDsvFormatOptions:
        bool_t HasHeader
        char Delimiter

cdef extern from "catboost/libs/options/load_options.h" namespace "NCatboostOptions":
    cdef cppclass TDsvPoolFormatParams:
        TDsvFormatOptions Format
        TPathWithScheme CdFilePath


cdef extern from "catboost/libs/data_new/visitor.h" namespace "NCB":
    cdef cppclass IRawObjectsOrderDataVisitor:
        void Start(
            bool_t inBlock,
            const TDataMetaInfo& metaInfo,
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

        void AddTarget(ui32 localObjectIdx, const TString& value) except +ProcessException
        void AddTarget(ui32 localObjectIdx, float value) except +ProcessException
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

        void AddFloatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder[float] features) except +ProcessException
        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef[TString] feature) except +ProcessException
        void AddCatFeature(ui32 flatFeatureIdx, TConstArrayRef[TStringBuf] feature) except +ProcessException

        void AddCatFeature(ui32 flatFeatureIdx, TMaybeOwningConstArrayHolder[ui32] features) except +ProcessException

        void AddTarget(TConstArrayRef[TString] value) except +ProcessException
        void AddTarget(TConstArrayRef[float] value) except +ProcessException
        void AddBaseline(ui32 baselineIdx, TConstArrayRef[float] value) except +ProcessException
        void AddWeights(TConstArrayRef[float] value) except +ProcessException
        void AddGroupWeights(TConstArrayRef[float] value) except +ProcessException

        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException

        void Finish() except +ProcessException


cdef extern from "catboost/libs/data_new/data_provider_builders.h" namespace "NCB":
    cdef cppclass IDataProviderBuilder:
        TDataProviderPtr GetResult() except +ProcessException

    cdef cppclass TDataProviderBuilderOptions:
        pass

    cdef void CreateDataProviderBuilderAndVisitor[IVisitor](
        const TDataProviderBuilderOptions& options,
        THolder[IDataProviderBuilder]* dataProviderBuilder,
        IVisitor** loader
    ) except +ProcessException


cdef extern from "catboost/libs/data_new/load_data.h" namespace "NCB":
    cdef TDataProviderPtr ReadDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,
        const TPathWithScheme& groupWeightsFilePath,
        const TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector[ui32]& ignoredFeatures,
        EObjectsOrder objectsOrder,
        int threadCount,
        bool_t verbose
    ) nogil except +ProcessException

cdef extern from "catboost/libs/algo/hessian.h":
    cdef cppclass THessianInfo:
        TVector[double] Data


cdef extern from "catboost/libs/model/ctr_provider.h":
    cdef cppclass ECtrTableMergePolicy:
        pass


cdef extern from "catboost/libs/model/model.h":
    cdef cppclass TCatFeature:
        int FeatureIndex
        int FlatFeatureIndex
        TString FeatureId

    cdef cppclass TFloatFeature:
        bool_t HasNans
        int FeatureIndex
        int FlatFeatureIndex
        TVector[float] Borders
        TString FeatureId

    cdef cppclass TObliviousTrees:
        int ApproxDimension
        TVector[TVector[double]] LeafWeights
        TVector[TCatFeature] CatFeatures
        TVector[TFloatFeature] FloatFeatures
        void DropUnusedFeatures() except +ProcessException

    cdef cppclass TFullModel:
        TObliviousTrees ObliviousTrees

        bool_t operator==(const TFullModel& other) except +ProcessException
        bool_t operator!=(const TFullModel& other) except +ProcessException

        THashMap[TString, TString] ModelInfo
        void Swap(TFullModel& other) except +ProcessException
        size_t GetTreeCount() nogil except +ProcessException
        void Truncate(size_t begin, size_t end) except +ProcessException

    cdef cppclass EModelType:
        pass

    cdef void ExportModel(
        const TFullModel& model,
        const TString& modelFile,
        const EModelType format,
        const TString& userParametersJson,
        bool_t addFileFormatExtension,
        const TVector[TString]* featureId,
        const THashMap[ui32, TString]* catFeaturesHashToString
    ) except +ProcessException

    cdef void OutputModel(const TFullModel& model, const TString& modelFile) except +ProcessException
    cdef TFullModel ReadModel(const TString& modelFile, EModelType format) nogil except +ProcessException
    cdef TString SerializeModel(const TFullModel& model) except +ProcessException
    cdef TFullModel DeserializeModel(const TString& serializeModelString) nogil except +ProcessException
    cdef TVector[TString] GetModelUsedFeaturesNames(const TFullModel& model) except +ProcessException
    cdef TVector[TString] GetModelClassNames(const TFullModel& model) except +ProcessException

ctypedef const TFullModel* TFullModel_const_ptr

cdef extern from "catboost/libs/model/model.h":
    cdef TFullModel SumModels(TVector[TFullModel_const_ptr], TVector[double], ECtrTableMergePolicy) nogil except +ProcessException


cdef extern from "library/json/writer/json_value.h" namespace "NJson":
    cdef cppclass TJsonValue:
        pass

cdef extern from "library/containers/2d_array/2d_array.h":
    cdef cppclass TArray2D[T]:
        T* operator[] (size_t index) const

cdef extern from "library/threading/local_executor/local_executor.h" namespace "NPar":
    cdef cppclass TLocalExecutor:
        TLocalExecutor() nogil
        void RunAdditionalThreads(int threadCount) nogil except +ProcessException

cdef extern from "util/system/info.h" namespace "NSystemInfo":
    cdef size_t CachedNumberOfCpus() except +ProcessException

cdef extern from "catboost/libs/metrics/metric_holder.h":
    cdef cppclass TMetricHolder:
        TVector[double] Stats

        void Add(TMetricHolder& other) except +ProcessException

cdef extern from "catboost/libs/metrics/metric.h":
    cdef cppclass IMetric:
        TString GetDescription() const;
        bool_t IsAdditiveMetric() const;

cdef extern from "catboost/libs/metrics/metric.h":
    cdef bool_t IsMaxOptimal(const IMetric& metric);

cdef extern from "catboost/libs/algo/ders_holder.h":
    cdef cppclass TDers:
        double Der1
        double Der2


cdef extern from "catboost/libs/options/enum_helpers.h":
    cdef bool_t IsClassificationObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsRegressionObjective(const TString& lossFunction) nogil except +ProcessException

cdef extern from "catboost/libs/metrics/metric.h":
    cdef cppclass TCustomMetricDescriptor:
        void* CustomData

        TMetricHolder (*EvalFunc)(
            const TVector[TVector[double]]& approx,
            const TConstArrayRef[float] target,
            const TConstArrayRef[float] weight,
            int begin, int end, void* customData
        ) except * with gil

        TString (*GetDescriptionFunc)(void *customData) except * with gil
        bool_t (*IsMaxOptimalFunc)(void *customData) except * with gil
        double (*GetFinalErrorFunc)(const TMetricHolder& error, void *customData) except * with gil

cdef extern from "catboost/libs/algo/custom_objective_descriptor.h":
    cdef cppclass TCustomObjectiveDescriptor:
        void* CustomData

        void (*CalcDersRange)(
            int count,
            const double* approxes,
            const float* targets,
            const float* weights,
            TDers* ders,
            void* customData
        ) except * with gil

        void (*CalcDersMulti)(
            const TVector[double]& approx,
            float target,
            float weight,
            TVector[double]* ders,
            THessianInfo* der2,
            void* customData
        ) except * with gil

cdef extern from "catboost/libs/options/cross_validation_params.h":
    cdef cppclass TCrossValidationParams:
        ui32 FoldCount
        bool_t Inverted
        int PartitionRandSeed
        bool_t Shuffle
        bool_t Stratified
        double MaxTimeSpentOnFixedCostRatio
        ui32 DevMaxIterationsBatchSize

cdef extern from "catboost/libs/options/check_train_options.h":
    cdef void CheckFitParams(
        const TJsonValue& tree,
        const TCustomObjectiveDescriptor* objectiveDescriptor,
        const TCustomMetricDescriptor* evalMetricDescriptor
    ) nogil except +ProcessException

cdef extern from "catboost/libs/options/json_helper.h":
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
        const TJsonValue& params,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        TDataProviders pools,
        const TString& outputModelPath,
        TFullModel* model,
        const TVector[TEvalResult*]& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory
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
        const TJsonValue& jsonParams,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        TDataProviderPtr data,
        const TCrossValidationParams& cvParams,
        TVector[TCVResult]* results
    ) nogil except +ProcessException

cdef extern from "catboost/libs/algo/apply.h":
    cdef cppclass TModelCalcerOnPool:
        TModelCalcerOnPool(
            const TFullModel& model,
            TIntrusivePtr[TObjectsDataProvider] objectsData,
            TLocalExecutor* executor
        ) nogil
        void ApplyModelMulti(
            const EPredictionType predictionType,
            int begin,
            int end,
            TVector[double]* flatApprox,
            TVector[TVector[double]]* approx
        ) nogil except +ProcessException

    cdef TVector[double] ApplyModel(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        bool_t verbose,
        const EPredictionType predictionType,
        int begin,
        int end,
        int threadCount
    ) nogil except +ProcessException

    cdef TVector[TVector[double]] ApplyModelMulti(
        const TFullModel& calcer,
        const TObjectsDataProvider& objectsData,
        bool_t verbose,
        const EPredictionType predictionType,
        int begin,
        int end,
        int threadCount
    ) nogil except +ProcessException

cdef extern from "catboost/libs/algo/helpers.h":
    cdef void ConfigureMalloc() nogil except *

cdef extern from "catboost/libs/algo/roc_curve.h":
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
        ) nogil

        TRocCurve(const TVector[TRocPoint]& points) nogil

        double SelectDecisionBoundaryByFalsePositiveRate(
            double falsePositiveRate
        ) nogil except +ProcessException

        double SelectDecisionBoundaryByFalseNegativeRate(
            double falseNegativeRate
        ) nogil except +ProcessException

        double SelectDecisionBoundaryByIntersection() nogil except +ProcessException

        TVector[TRocPoint] GetCurvePoints() nogil except +ProcessException

        void Output(const TString& outputPath)

cdef extern from "catboost/libs/eval_result/eval_helpers.h":
    cdef TVector[TVector[double]] PrepareEval(
        const EPredictionType predictionType,
        const TVector[TVector[double]]& approx,
        int threadCount
    ) nogil except +ProcessException

    cdef TVector[TVector[double]] PrepareEvalForInternalApprox(
        const EPredictionType predictionType,
        const TFullModel& model,
        const TVector[TVector[double]]& approx,
        int threadCount
    ) nogil except +ProcessException

    cdef TVector[TString] ConvertTargetToExternalName(
        const TVector[float]& target,
        const TFullModel& model
    ) nogil except +ProcessException

cdef extern from "catboost/libs/eval_result/eval_result.h" namespace "NCB":
    cdef cppclass TEvalResult:
        TVector[TVector[TVector[double]]] GetRawValuesRef() except * with gil
        void ClearRawValues() except * with gil

cdef extern from "catboost/libs/init/init_reg.h" namespace "NCB":
    cdef void LibraryInit() nogil except *

cdef extern from "catboost/libs/fstr/calc_fstr.h":
    cdef TVector[TVector[double]] GetFeatureImportances(
        const TString& type,
        const TFullModel& model,
        const TDataProviderPtr dataset,
        int threadCount,
        int logPeriod
    ) nogil except +ProcessException

    cdef TVector[TVector[TVector[double]]] GetFeatureImportancesMulti(
        const TString& type,
        const TFullModel& model,
        const TDataProviderPtr dataset,
        int threadCount,
        int logPeriod
    ) nogil except +ProcessException

    TVector[TString] GetMaybeGeneratedModelFeatureIds(
        const TFullModel& model,
        const TDataProviderPtr dataset
    ) nogil except +ProcessException


cdef extern from "catboost/libs/documents_importance/docs_importance.h":
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

cdef extern from "catboost/libs/data_new/borders_io.h" namespace "NCB" nogil:
    void LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
        const TString& path,
        TQuantizedFeaturesInfo* quantizedFeaturesInfo) except *

cdef extern from "catboost/libs/data_new/loader.h" namespace "NCB" nogil:
    int IsMissingValue(const TStringBuf& s)

cdef inline float _FloatOrNanFromString(const TString& s) except *:
    cdef char* stop = NULL
    cdef double parsed = StrToD(s.data(), &stop)
    cdef float res
    if s.empty():
        res = _FLOAT_NAN
    elif stop == s.data() + s.size():
        res = parsed
    elif IsMissingValue(<TStringBuf>s):
        res = _FLOAT_NAN
    else:
        raise TypeError("Cannot convert '{}' to float".format(str(s)))
    return res

cdef extern from "catboost/libs/gpu_config/interface/get_gpu_device_count.h" namespace "NCB":
    cdef int GetGpuDeviceCount()

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
        const TVector[float]& label,
        const TVector[TVector[double]]& approx,
        const TString& metricName,
        const TVector[float]& weight,
        const TVector[TGroupId]& groupId,
        int threadCount
    ) nogil except +ProcessException

    cdef cppclass TMetricsPlotCalcerPythonWrapper:
        TMetricsPlotCalcerPythonWrapper(TVector[TString]& metrics, TFullModel& model, int ntree_start, int ntree_end,
                                        int eval_period, int thread_count, TString& tmpDir,
                                        bool_t flag) except +ProcessException
        TVector[const IMetric*] GetMetricRawPtrs() const
        TVector[TVector[double]] ComputeScores()
        void AddPool(const TDataProvider& srcData)


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


cdef _vector_of_double_to_np_array(TVector[double]& vec):
    result = np.empty(vec.size(), dtype=_npfloat64)
    for i in xrange(vec.size()):
        result[i] = vec[i]
    return result


cdef _2d_vector_of_double_to_np_array(TVector[TVector[double]]& vectors):
    cdef size_t subvec_size = vectors[0].size() if not vectors.empty() else 0
    result = np.empty([vectors.size(), subvec_size], dtype=_npfloat64)
    for i in xrange(vectors.size()):
        assert vectors[i].size() == subvec_size, "All subvectors should have the same length"
        for j in xrange(subvec_size):
            result[i][j] = vectors[i][j]
    return result


cdef _3d_vector_of_double_to_np_array(TVector[TVector[TVector[double]]]& vectors):
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
) except * with gil:
    cdef metricObject = <object>customData
    cdef TMetricHolder holder
    holder.Stats.resize(2)

    approxes = [_DoubleArrayWrapper.create(approx[i].data() + begin, end - begin) for i in xrange(approx.size())]
    targets = _FloatArrayWrapper.create(target.data() + begin, end - begin)

    if weight.size() == 0:
        weights = None
    else:
        weights = _FloatArrayWrapper.create(weight.data() + begin, end - begin)

    error, weight_ = metricObject.evaluate(approxes, targets, weights)

    holder.Stats[0] = error
    holder.Stats[1] = weight_
    return holder

cdef void _ObjectiveCalcDersRange(
    int count,
    const double* approxes,
    const float* targets,
    const float* weights,
    TDers* ders,
    void* customData
) except * with gil:
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

cdef void _ObjectiveCalcDersMulti(
    const TVector[double]& approx,
    float target,
    float weight,
    TVector[double]* ders,
    THessianInfo* der2,
    void* customData
) except * with gil:
    cdef objectiveObject = <object>(customData)

    approxes = _DoubleArrayWrapper.create(approx.data(), approx.size())

    ders_vector, second_ders_matrix = objectiveObject.calc_ders_multi(approxes, target, weight)
    for index, der in enumerate(ders_vector):
        dereference(ders)[index] = der

    index = 0
    for indY, line in enumerate(second_ders_matrix):
        for num in line[indY:]:
            dereference(der2).Data[index] = num
            index += 1

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

cdef EModelType string_to_model_type(model_type_str) except *:
    cdef EModelType model_type
    if not TryFromString[EModelType](to_arcadia_string(model_type_str), model_type):
        raise CatBoostError("Unknown model type {}.".format(model_type_str))
    return model_type


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

        params['verbose'] = int(params['verbose']) if 'verbose' in params else (
            params['metric_period'] if 'metric_period' in params else 1)

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

        dumps_params = dumps(params_to_json, cls=_NumpyAwareEncoder)

        if params_to_json.get("loss_function") == "Custom":
            self.customObjectiveDescriptor = _BuildCustomObjectiveDescriptor(params["loss_function"])
        if params_to_json.get("eval_metric") == "Custom":
            self.customMetricDescriptor = _BuildCustomMetricDescriptor(params["eval_metric"])

        self.tree = ReadTJsonValue(to_arcadia_string(dumps_params))


cdef TString to_arcadia_string(s):
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
cdef _npfloat32 = np.float32
cdef _npfloat64 = np.float64


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
    elif obj_type is float or obj_type is _npfloat32 or obj_type is _npfloat64:
        double_val = <double>id_object
        if isnan(double_val) or <i64>double_val != double_val:
            raise CatBoostError("bad object for id: {}".format(id_object))
        bytes_string_buf_representation[0] = ToString[i64](<i64>double_val)
    else:
        # this part is really heavy as it uses lot's of python internal magic, so put it down
        if isinstance(id_object, all_string_types_plus_bytes):
            # for some reason Cython does not allow assignment to dereferenced pointer, so use this trick instead
            bytes_string_buf_representation[0] = to_arcadia_string(id_object)
        else:
            if isnan(id_object) or int(id_object) != id_object:
                raise CatBoostError("bad object for id: {}".format(id_object))
            bytes_string_buf_representation[0] = ToString[int](int(id_object))


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


cdef TFeaturesLayout* _init_features_layout(data, cat_features, feature_names):
    cdef TVector[ui32] cat_features_vector
    cdef TVector[TString] feature_names_vector

    if isinstance(data, FeaturesData):
        feature_count = data.get_feature_count()
        cat_features = [i for i in range(data.get_num_feature_count(), feature_count)]
        feature_names = data.get_feature_names()
    else:
        feature_count = np.shape(data)[1]

    if cat_features is not None:
        for cat_feature in cat_features:
            cat_features_vector.push_back(cat_feature)

    if feature_names is not None:
        for feature_name in feature_names:
            feature_names_vector.push_back(to_arcadia_string(str(feature_name)))

    return new TFeaturesLayout(
        <ui32>feature_count,
        cat_features_vector,
        feature_names_vector,
        <TPoolQuantizationSchema*>nullptr)

cdef TVector[bool_t] _get_is_cat_feature_mask(const TFeaturesLayout* featuresLayout):
    cdef TVector[bool_t] mask
    mask.resize(featuresLayout.GetExternalFeatureCount(), False)

    cdef ui32 idx
    for idx in range(featuresLayout.GetExternalFeatureCount()):
        if featuresLayout[0].GetExternalFeatureType(idx) == EFeatureType_Categorical:
            mask[idx] = True

    return mask

cdef _get_object_count(data):
    if isinstance(data, FeaturesData):
        return data.get_object_count()
    else:
        return np.shape(data)[0]

cdef _set_features_order_data_np(
    const float [:,:] num_feature_values,
    object [:,:] cat_feature_values, # cannot be const due to https://github.com/cython/cython/issues/2485
    IRawFeaturesOrderDataVisitor* builder_visitor
):
    if (num_feature_values is None) and (cat_feature_values is None):
        raise CatBoostError('both num_feature_values and cat_feature_values are empty')

    cdef ui32 doc_count = <ui32>(
        num_feature_values.shape[0] if num_feature_values is not None else cat_feature_values.shape[0]
    )

    cdef ui32 num_feature_count = <ui32>(num_feature_values.shape[1] if num_feature_values is not None else 0)
    cdef ui32 cat_feature_count = <ui32>(cat_feature_values.shape[1] if cat_feature_values is not None else 0)

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
        builder_visitor[0].AddFloatFeature(
            dst_feature_idx,
            TMaybeOwningConstArrayHolder[float].CreateNonOwning(
                TConstArrayRef[float](&num_feature_values[0, num_feature_idx], doc_count)
                if doc_count > 0
                else TConstArrayRef[float]()
            )
        )
        dst_feature_idx += 1
    for cat_feature_idx in range(cat_feature_count):
        cat_factor_data.clear()
        for doc_idx in range(doc_count):
            factor_string = to_arcadia_string(cat_feature_values[doc_idx, cat_feature_idx])
            cat_factor_data.push_back(factor_string)
        builder_visitor[0].AddCatFeature(dst_feature_idx, <TConstArrayRef[TString]>cat_factor_data)
        dst_feature_idx += 1


cdef float get_float_feature(ui32 doc_idx, ui32 flat_feature_idx, src_value) except*:
    try:
        return _FloatOrNan(src_value)
    except TypeError as e:
        raise CatBoostError(
            'Bad value for num_feature[{},{}]="{}": {}'.format(
                doc_idx,
                flat_feature_idx,
                src_value,
                e
            )
        )


cdef TIntrusivePtr[TVectorHolder[float]] create_num_factor_data(
    ui32 flat_feature_idx,
    np.ndarray column_values
) except*:
    cdef TIntrusivePtr[TVectorHolder[float]] num_factor_data = new TVectorHolder[float]()
    cdef ui32 doc_idx

    num_factor_data.Get()[0].Data.resize(len(column_values))
    for doc_idx in range(len(column_values)):
        num_factor_data.Get()[0].Data[doc_idx] = get_float_feature(
            doc_idx,
            flat_feature_idx,
            column_values[doc_idx]
        )

    return num_factor_data


cdef get_cat_factor_bytes_representation(
    ui32 doc_idx,
    ui32 feature_idx,
    object factor,
    TString* factor_strbuf
):
    try:
        get_id_object_bytes_string_representation(factor, factor_strbuf)
    except CatBoostError:
        raise CatBoostError(
            'Invalid type for cat_feature[{},{}]={} :'
            ' cat_features must be integer or string, real number values and NaN values'
            ' should be converted to string.'.format(doc_idx, feature_idx, factor)
        )


# returns new data holders array
cdef object _set_features_order_data_pd_data_frame(
    data_frame,
    const TFeaturesLayout* features_layout,
    IRawFeaturesOrderDataVisitor* builder_visitor
):
    cdef TVector[bool_t] is_cat_feature_mask = _get_is_cat_feature_mask(features_layout)
    cdef ui32 doc_count = data_frame.shape[0]

    cdef TString factor_string

    # two pointers are needed as a workaround for Cython assignment of derived types restrictions
    cdef TIntrusivePtr[TVectorHolder[float]] num_factor_data
    cdef TIntrusivePtr[IResourceHolder] num_factor_data_holder

    cdef TVector[TString] cat_factor_data
    cdef ui32 doc_idx
    cdef ui32 flat_feature_idx
    cdef np.ndarray column_values # for columns that are not of type pandas.Categorical
    cdef bool_t column_type_is_pandas_Categorical

    cat_factor_data.reserve(doc_count)

    new_data_holders = []
    for flat_feature_idx, (column_name, column_data) in enumerate(data_frame.iteritems()):
        column_type_is_pandas_Categorical = column_data.dtype.name == 'category'
        if not column_type_is_pandas_Categorical:
            column_values = column_data.values
        if is_cat_feature_mask[flat_feature_idx]:
            cat_factor_data.clear()
            for doc_idx in range(doc_count):
                get_cat_factor_bytes_representation(
                    doc_idx,
                    flat_feature_idx,
                    column_data[doc_idx] if column_type_is_pandas_Categorical else column_values[doc_idx],
                    &factor_string
                )
                cat_factor_data.push_back(factor_string)
            builder_visitor[0].AddCatFeature(flat_feature_idx, <TConstArrayRef[TString]>cat_factor_data)
        elif ((not column_type_is_pandas_Categorical) and
              (column_values.dtype == np.float32) and
              column_values.flags.aligned and
              column_values.flags.c_contiguous
            ):

            new_data_holders.append(column_values)
            column_values.setflags(write=0)

            builder_visitor[0].AddFloatFeature(
                flat_feature_idx,
                TMaybeOwningConstArrayHolder[float].CreateNonOwning(
                    TConstArrayRef[float](<float*>column_values.data, doc_count)
                    if doc_count > 0
                    else TConstArrayRef[float]()
                )
            )
        else:
            num_factor_data = create_num_factor_data(
                flat_feature_idx,
                column_values if not column_type_is_pandas_Categorical else np.asarray(column_data)
            )
            num_factor_data_holder.Reset(num_factor_data.Get())
            builder_visitor[0].AddFloatFeature(
                flat_feature_idx,
                TMaybeOwningConstArrayHolder[float].CreateOwning(
                    <TConstArrayRef[float]>num_factor_data.Get()[0].Data,
                    num_factor_data_holder
                )
            )

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

    cdef TVector[bool_t] is_cat_feature_mask = _get_is_cat_feature_mask(features_layout)

    for doc_idx in range(doc_count):
        doc_data = data[doc_idx]
        for feature_idx in range(feature_count):
            factor = doc_data[feature_idx]
            if is_cat_feature_mask[feature_idx]:
                get_cat_factor_bytes_representation(
                    doc_idx,
                    feature_idx,
                    factor,
                    &factor_strbuf
                )
                builder_visitor[0].AddCatFeature(doc_idx, feature_idx, <TStringBuf>factor_strbuf)
            else:
                builder_visitor[0].AddFloatFeature(
                    doc_idx,
                    feature_idx,
                    get_float_feature(doc_idx, feature_idx, factor)
                )

cdef _set_data(data, const TFeaturesLayout* features_layout, IRawObjectsOrderDataVisitor* builder_visitor):
    if isinstance(data, FeaturesData):
        _set_data_np(data.num_feature_data, data.cat_feature_data, builder_visitor)
    else:
        if isinstance(data, np.ndarray) and data.dtype == np.float32:
            _set_data_np(data, None, builder_visitor)
        else:
            _set_data_from_generic_matrix(data, features_layout, builder_visitor)


cdef _set_label(label, IRawObjectsOrderDataVisitor* builder_visitor):
    for i in range(len(label)):
        if isinstance(label[i], all_string_types_plus_bytes):
            builder_visitor[0].AddTarget(
                <ui32>i,
                to_arcadia_string(label[i])
            )
        else:
            builder_visitor[0].AddTarget(
                <ui32>i,
                to_arcadia_string(str(label[i]))
            )


cdef _set_label_features_order(label, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef TVector[TString] labelVector
    cdef TString bytes_string_representation
    labelVector.reserve(len(label))
    for i in range(len(label)):
        if isinstance(label[i], all_string_types_plus_bytes):
            bytes_string_representation = to_arcadia_string(label[i])
        else:
            bytes_string_representation = to_arcadia_string(str(label[i]))
        labelVector.push_back(bytes_string_representation)
    builder_visitor[0].AddTarget(<TConstArrayRef[TString]>labelVector)


ctypedef fused IBuilderVisitor:
    IRawObjectsOrderDataVisitor
    IRawFeaturesOrderDataVisitor


cdef TVector[TPair] _make_pairs_vector(pairs, pairs_weight=None):
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
    for i in range(len(weight)):
        builder_visitor[0].AddWeight(i, float(weight[i]))

cdef _set_weight_features_order(weight, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef TVector[float] weightVector
    weightVector.reserve(len(weight))
    for i in range(len(weight)):
        weightVector.push_back(float(weight[i]))
    builder_visitor[0].AddWeights(<TConstArrayRef[float]>weightVector)

cdef TGroupId _calc_group_id_for(i, py_group_ids):
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
    for i in range(len(group_id)):
        builder_visitor[0].AddGroupId(i, _calc_group_id_for(i, group_id))

cdef _set_group_weight(group_weight, IRawObjectsOrderDataVisitor* builder_visitor):
    for i in range(len(group_weight)):
        builder_visitor[0].AddGroupWeight(i, float(group_weight[i]))

cdef _set_group_weight_features_order(group_weight, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef TVector[float] groupWeightVector
    groupWeightVector.reserve(len(group_weight))
    for i in range(len(group_weight)):
        groupWeightVector.push_back(float(group_weight[i]))
    builder_visitor[0].AddGroupWeights(<TConstArrayRef[float]>groupWeightVector)

cdef TSubgroupId _calc_subgroup_id_for(i, py_subgroup_ids):
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
    for i in range(len(subgroup_id)):
        builder_visitor[0].AddSubgroupId(i, _calc_subgroup_id_for(i, subgroup_id))

cdef _set_baseline(baseline, IRawObjectsOrderDataVisitor* builder_visitor):
    for i in range(len(baseline)):
        for j, value in enumerate(baseline[i]):
            builder_visitor[0].AddBaseline(i, j, float(value))

cdef _set_baseline_features_order(baseline, IRawFeaturesOrderDataVisitor* builder_visitor):
    cdef ui32 baseline_count = len(baseline[0])
    cdef TVector[float] one_dim_baseline

    for baseline_idx in range(baseline_count):
        one_dim_baseline.clear()
        one_dim_baseline.reserve(len(baseline))
        for i in range(len(baseline)):
            one_dim_baseline.push_back(float(baseline[i][baseline_idx]))
        builder_visitor[0].AddBaseline(baseline_idx, <TConstArrayRef[float]>one_dim_baseline)


cdef class _PoolBase:
    cdef TDataProviderPtr __pool
    cdef object target_type

    # possibly hold reference or list of references to data to allow using views to them in __pool
    cdef object __data_holders

    def __cinit__(self):
        self.__pool = TDataProviderPtr()
        self.target_type = None
        self.__data_holders = None

    def __dealloc__(self):
        self.__pool.Drop()

    def __deepcopy__(self, _):
        raise CatBoostError('Can\'t deepcopy _PoolBase object')

    def __eq__(self, _PoolBase other):
        return dereference(self.__pool.Get()) == dereference(other.__pool.Get())


    cpdef _read_pool(self, pool_file, cd_file, pairs_file, delimiter, bool_t has_header, int thread_count):
        cdef TPathWithScheme pool_file_path
        pool_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(pool_file), TStringBuf(<char*>'dsv'))

        cdef TPathWithScheme pairs_file_path
        if len(pairs_file):
            pairs_file_path = TPathWithScheme(<TStringBuf>to_arcadia_string(pairs_file), TStringBuf(<char*>'dsv'))

        cdef TDsvPoolFormatParams dsvPoolFormatParams
        dsvPoolFormatParams.Format.HasHeader = has_header
        dsvPoolFormatParams.Format.Delimiter = ord(delimiter)
        if len(cd_file):
            dsvPoolFormatParams.CdFilePath = TPathWithScheme(<TStringBuf>to_arcadia_string(cd_file), TStringBuf(<char*>'dsv'))

        thread_count = UpdateThreadCount(thread_count)

        cdef TVector[ui32] emptyIntVec

        self.__pool = ReadDataset(
            pool_file_path,
            pairs_file_path,
            TPathWithScheme(),
            dsvPoolFormatParams,
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
        baseline):

        cdef TDataProviderBuilderOptions options
        cdef THolder[IDataProviderBuilder] data_provider_builder
        cdef IRawFeaturesOrderDataVisitor* builder_visitor

        CreateDataProviderBuilderAndVisitor(options, &data_provider_builder, &builder_visitor)

        cdef TVector[TIntrusivePtr[IResourceHolder]] resource_holders
        builder_visitor[0].Start(
            data_meta_info,
            _get_object_count(data),
            EObjectsOrder_Undefined,
            resource_holders
        )

        if isinstance(data, FeaturesData):
            new_data_holders = data

            # needed because of https://github.com/cython/cython/issues/2485
            if data.cat_feature_data is not None:
                data.cat_feature_data.setflags(write=1)

            _set_features_order_data_np(
                data.num_feature_data,
                data.cat_feature_data,
                builder_visitor)

            # set after _set_features_order_data_np call because we can't pass const cat_feature_data to it
            # https://github.com/cython/cython/issues/2485
            if data.num_feature_data is not None:
                data.num_feature_data.setflags(write=0)
            if data.cat_feature_data is not None:
                data.cat_feature_data.setflags(write=0)
        elif isinstance(data, pd.DataFrame):
            new_data_holders = _set_features_order_data_pd_data_frame(
                data,
                data_meta_info.FeaturesLayout.Get(),
                builder_visitor
            )
        elif isinstance(data, np.ndarray) and data.dtype == np.float32:
            new_data_holders = data
            data.setflags(write=0)
            _set_features_order_data_np(data, None, builder_visitor)
        else:
            raise CatBoostError(
                '[Internal error] wrong data type for _init_features_order_layout_pool: ' + type(data)
            )

        num_class = 2
        if label is not None:
            _set_label_features_order(label, builder_visitor)
            num_class = len(set(list(label)))
            if len(label) > 0:
                self.target_type = type(label[0])
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

        self.__pool = data_provider_builder.Get()[0].GetResult()
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
        baseline):

        self.__data_holder = None # free previously used resources

        cdef TDataProviderBuilderOptions options
        cdef THolder[IDataProviderBuilder] data_provider_builder
        cdef IRawObjectsOrderDataVisitor* builder_visitor

        CreateDataProviderBuilderAndVisitor(options, &data_provider_builder, &builder_visitor)

        cdef TVector[TIntrusivePtr[IResourceHolder]] resource_holders
        builder_visitor[0].Start(
            False,
            data_meta_info,
            _get_object_count(data),
            EObjectsOrder_Undefined,
            resource_holders
        )
        builder_visitor[0].StartNextBlock(_get_object_count(data))

        _set_data(data, data_meta_info.FeaturesLayout.Get(), builder_visitor)

        num_class = 2
        if label is not None:
            _set_label(label, builder_visitor)
            num_class = len(set(list(label)))
            if len(label) > 0:
                self.target_type = type(label[0])
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

        self.__pool = data_provider_builder.Get()[0].GetResult()


    cpdef _init_pool(self, data, label, cat_features, pairs, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, feature_names):
        if group_weight is not None and weight is not None:
            raise CatBoostError('Pool must have either weight or group_weight.')

        cdef TDataMetaInfo data_meta_info
        data_meta_info.HasTarget = label is not None
        data_meta_info.BaselineCount = len(baseline[0]) if baseline is not None else 0
        data_meta_info.HasGroupId = group_id is not None
        data_meta_info.HasGroupWeight = group_weight is not None
        data_meta_info.HasSubgroupIds = subgroup_id is not None
        data_meta_info.HasWeights = weight is not None
        data_meta_info.HasTimestamp = False
        data_meta_info.HasPairs = pairs is not None

        data_meta_info.FeaturesLayout = _init_features_layout(data, cat_features, feature_names)

        do_use_raw_data_in_features_order = False
        if isinstance(data, FeaturesData):
            if ((data.num_feature_data is not None) and
                data.num_feature_data.flags.aligned and
                data.num_feature_data.flags.f_contiguous
               ):
                do_use_raw_data_in_features_order = True
        elif isinstance(data, pd.DataFrame):
            do_use_raw_data_in_features_order = True
        else:
            if isinstance(data, np.ndarray) and data.dtype == np.float32:
                if data.flags.aligned and data.flags.f_contiguous:
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
                baseline
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
                baseline
            )


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


    cdef _get_feature(self, TRawObjectsDataProvider* raw_objects_data_provider, factor_idx, dst_data):
        cdef TVector[float] factorData = raw_objects_data_provider[0].GetFeatureDataOldFormat(factor_idx)
        for doc in range(self.num_row()):
            dst_data[doc, factor_idx] = factorData[doc]


    cpdef get_features(self):
        """
        Get feature matrix from Pool.

        Returns
        -------
        feature matrix : np.array of shape (rows, cols)
        """
        cdef TRawObjectsDataProvider* raw_objects_data_provider = dynamic_cast_to_TRawObjectsDataProvider(
            self.__pool.Get()[0].ObjectsData.Get()
        )
        if not raw_objects_data_provider:
            raise CatBoostError('Pool does not have raw features data, only quantized')

        data = np.empty(self.shape, dtype=np.float32)

        for factor in range(self.num_col()):
            self._get_feature(raw_objects_data_provider, factor, data)

        return data

    cpdef get_label(self):
        """
        Get labels from Pool.

        Returns
        -------
        labels : list
        """
        cdef TMaybeData[TConstArrayRef[TString]] maybe_target = self.__pool.Get()[0].RawTargetData.GetTarget()
        if maybe_target.Defined():
            return [self.target_type(target_string.decode()) for target_string in maybe_target.GetRef()]
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

    cpdef get_cat_feature_hash_to_string(self):
        """
        Get mapping of float hash values to corresponding strings

        Returns
        -------
        hash_to_string : map
        """
        cdef const THashMap[ui32, TString]* cat_features_hash_to_string

        hash_to_string = {}

        cat_feature_count = self.__pool.Get()[0].MetaInfo.FeaturesLayout.Get()[0].GetCatFeatureCount()
        for cat_feature_idx in range(cat_feature_count):
            cat_features_hash_to_string = &(
                self.__pool.Get()[0].ObjectsData.Get()[0].GetCatFeaturesHashToString(cat_feature_idx)
            )

            # can't use canonical for loop here due to Cython's bugs:
            # https://github.com/cython/cython/issues/1451
            it = cat_features_hash_to_string[0].const_begin()
            while it != cat_features_hash_to_string[0].const_end():
                hash_to_string[ConvertCatFeatureHashToFloat(dereference(it).first)] = to_native_str(dereference(it).second)
                preincrement(it)

        return hash_to_string

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
        baseline : np.array
        """
        cdef TMaybeData[TBaselineArrayRef] maybe_baseline = self.__pool.Get()[0].RawTargetData.GetBaseline()
        cdef TBaselineArrayRef baseline
        if maybe_baseline.Defined():
            baseline = maybe_baseline.GetRef()
            result = np.array((self.num_row(), baseline.size()), dtype=np.float32)
            for baseline_idx in range(baseline.size()):
                for object_idx in range(self.num_row()):
                    result[object_idx, baseline_idx] = baseline[baseline_idx][object_idx]
            return result
        else:
            return np.array((self.num_row(), 0), dtype=np.float32)

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
            thread_count
        )
        self.target_type = pool.target_type

    @property
    def is_empty_(self):
        """
        Check if Pool is empty (contains no objects).

        Returns
        -------
        is_empty_ : bool
        """
        return self.num_row() == 0


cdef class _CatBoost:
    cdef TFullModel* __model
    cdef TVector[TEvalResult*] __test_evals
    cdef TMetricsAndTimeLeftHistory __metrics_history

    def __cinit__(self):
        self.__model = new TFullModel()

    def __dealloc__(self):
        del self.__model
        for i in range(self.__test_evals.size()):
            del self.__test_evals[i]

    def __eq__(self, _CatBoost other):
        return self.__model == other.__model

    def __neq__(self, _CatBoost other):
        return self.__model != other.__model

    cpdef _reserve_test_evals(self, num_tests):
        if self.__test_evals.size() < num_tests:
            self.__test_evals.resize(num_tests)
        for i in range(num_tests):
            if self.__test_evals[i] == NULL:
                self.__test_evals[i] = new TEvalResult()

    cpdef _clear_test_evals(self):
        for i in range(self.__test_evals.size()):
            dereference(self.__test_evals[i]).ClearRawValues()

    cpdef _train(self, _PoolBase train_pool, test_pools, dict params, allow_clear_pool):
        _input_borders = params.pop("input_borders", None)
        prep_params = _PreprocessParams(params)
        cdef int thread_count = params.get("thread_count", 1)
        cdef TDataProviders dataProviders
        dataProviders.Learn = train_pool.__pool
        cdef _PoolBase test_pool
        cdef TVector[ui32] ignored_features
        cdef TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
        cdef TString input_borders_str
        if isinstance(test_pools, list):
            if params.get('task_type', 'CPU') == 'GPU' and len(test_pools) > 1:
                raise CatBoostError('Multiple eval sets are not supported on GPU')
            for test_pool in test_pools:
                dataProviders.Test.push_back(test_pool.__pool)
        else:
            test_pool = test_pools
            dataProviders.Test.push_back(test_pool.__pool)
        self._reserve_test_evals(dataProviders.Test.size())
        self._clear_test_evals()

        if (_input_borders):
            quantizedFeaturesInfo = new TQuantizedFeaturesInfo(
                dereference(dereference(dataProviders.Learn.Get()).MetaInfo.FeaturesLayout.Get()),
                TConstArrayRef[ui32](),
                TBinarizationOptions()
            )
            input_borders_str = to_arcadia_string(_input_borders)
            with nogil:
                LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
                    input_borders_str,
                    quantizedFeaturesInfo.Get())

        with nogil:
            SetPythonInterruptHandler()
            try:
                TrainModel(
                    prep_params.tree,
                    quantizedFeaturesInfo,
                    prep_params.customObjectiveDescriptor,
                    prep_params.customMetricDescriptor,
                    dataProviders,
                    TString(<const char*>""),
                    self.__model,
                    self.__test_evals,
                    &self.__metrics_history
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
        num_iterations = self.__metrics_history.LearnMetricsHistory.size()
        for iter in range(num_iterations):
            for metric, value in self.__metrics_history.LearnMetricsHistory[iter]:
                metrics_evals["learn"][to_native_str(metric)].append(value)
            if not self.__metrics_history.TestMetricsHistory.empty():
                num_tests = self.__metrics_history.TestMetricsHistory[iter].size()
                for test in range(num_tests):
                    for metric, value in self.__metrics_history.TestMetricsHistory[iter][test]:
                        metrics_evals["validation_" + str(test)][to_native_str(metric)].append(value)
        return {k: dict(v) for k, v in iteritems(metrics_evals)}

    cpdef _get_best_score(self):
        if self.__metrics_history.LearnBestError.empty():
            return {}
        best_scores = {}
        best_scores["learn"] = {}
        for metric, best_error in self.__metrics_history.LearnBestError:
            best_scores["learn"][to_native_str(metric)] = best_error
        for testIdx in range(self.__metrics_history.TestBestError.size()):
            best_scores["validation_" + str(testIdx)] = {}
            for metric, best_error in self.__metrics_history.TestBestError[testIdx]:
                best_scores["validation_" + str(testIdx)][to_native_str(metric)] = best_error
        return best_scores

    cpdef _get_best_iteration(self):
        if self.__metrics_history.BestIteration.Defined():
            return self.__metrics_history.BestIteration.GetRef()
        return None

    cpdef _has_leaf_weights_in_model(self):
        return not self.__model.ObliviousTrees.LeafWeights.empty()

    cpdef _get_cat_feature_indices(self):
        return [feature.FlatFeatureIndex for feature in self.__model.ObliviousTrees.CatFeatures]

    cpdef _get_float_feature_indices(self):
        return [feature.FlatFeatureIndex for feature in self.__model.ObliviousTrees.FloatFeatures]

    cpdef _base_predict(self, _PoolBase pool, str prediction_type, int ntree_start, int ntree_end, int thread_count, bool_t verbose):
        cdef TVector[double] pred
        cdef EPredictionType predictionType = PyPredictionType(prediction_type).predictionType
        thread_count = UpdateThreadCount(thread_count);
        cdef const TObjectsDataProvider* objectsData = &pool.__pool.Get()[0].ObjectsData.Get()[0]
        with nogil:
            pred = ApplyModel(
                dereference(self.__model),
                dereference(objectsData),
                verbose,
                predictionType,
                ntree_start,
                ntree_end,
                thread_count
            )
        return _vector_of_double_to_np_array(pred)


    cpdef _base_predict_multi(self, _PoolBase pool, str prediction_type, int ntree_start, int ntree_end,
                              int thread_count, bool_t verbose):
        cdef TVector[TVector[double]] pred
        cdef EPredictionType predictionType = PyPredictionType(prediction_type).predictionType
        thread_count = UpdateThreadCount(thread_count);
        cdef const TObjectsDataProvider* objectsData = &pool.__pool.Get()[0].ObjectsData.Get()[0]
        with nogil:
            pred = ApplyModelMulti(
                dereference(self.__model),
                dereference(objectsData),
                verbose,
                predictionType,
                ntree_start,
                ntree_end,
                thread_count
            )
        return _convert_to_visible_labels(predictionType, pred, thread_count, self.__model)

    cpdef _staged_predict_iterator(self, _PoolBase pool, str prediction_type, int ntree_start, int ntree_end, int eval_period, int thread_count, verbose):
        thread_count = UpdateThreadCount(thread_count);
        stagedPredictIterator = _StagedPredictIterator(prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose)
        stagedPredictIterator._initialize_model_calcer(self.__model, pool)
        return stagedPredictIterator

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

    cpdef _calc_fstr(self, type_name, _PoolBase pool, int thread_count, int verbose):
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
        cdef TString type_name_str = to_arcadia_string(type_name)
        if type_name == 'ShapValues' and dereference(self.__model).ObliviousTrees.ApproxDimension > 1:
            with nogil:
                fstr_multi = GetFeatureImportancesMulti(
                    type_name_str,
                    dereference(self.__model),
                    dataProviderPtr,
                    thread_count,
                    verbose
                )
            return _3d_vector_of_double_to_np_array(fstr_multi), native_feature_ids
        else:
            with nogil:
                fstr = GetFeatureImportances(
                    type_name_str,
                    dereference(self.__model),
                    dataProviderPtr,
                    thread_count,
                    verbose
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

    cpdef _base_drop_unused_features(self):
        self.__model.ObliviousTrees.DropUnusedFeatures()

    cpdef _load_model(self, model_file, format):
        cdef TFullModel tmp_model
        cdef EModelType modelType = string_to_model_type(format)
        tmp_model = ReadModel(to_arcadia_string(model_file), modelType)
        self.__model.Swap(tmp_model)

    cpdef _save_model(self, output_file, format, export_parameters, _PoolBase pool):
        cdef EModelType modelType = string_to_model_type(format)

        cdef TVector[TString] feature_id
        if pool:
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

    def _get_class_names(self):
        return [to_native_str(s) for s in GetModelClassNames(dereference(self.__model))]

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


cpdef _cv(dict params, _PoolBase pool, int fold_count, bool_t inverted, int partition_random_seed,
          bool_t shuffle, bool_t stratified, bool_t as_pandas, double max_time_spent_on_fixed_cost_ratio,
          int dev_max_iterations_batch_size):
    prep_params = _PreprocessParams(params)
    cdef TCrossValidationParams cvParams
    cdef TVector[TCVResult] results

    cvParams.FoldCount = fold_count
    cvParams.PartitionRandSeed = partition_random_seed
    cvParams.Shuffle = shuffle
    cvParams.Stratified = stratified
    cvParams.Inverted = inverted
    cvParams.MaxTimeSpentOnFixedCostRatio = max_time_spent_on_fixed_cost_ratio
    cvParams.DevMaxIterationsBatchSize = <ui32>dev_max_iterations_batch_size

    with nogil:
        SetPythonInterruptHandler()
        try:
            CrossValidate(
                prep_params.tree,
                prep_params.customObjectiveDescriptor,
                prep_params.customMetricDescriptor,
                pool.__pool,
                cvParams,
                &results)
        finally:
            ResetPythonInterruptHandler()

    result = defaultdict(list)
    metric_count = results.size()
    used_metric_names = set()
    for metric_idx in xrange(metric_count):
        metric_name = to_native_str(results[metric_idx].Metric.c_str())
        if metric_name in used_metric_names:
            continue
        used_metric_names.add(metric_name)
        fill_iterations_column = 'iterations' not in result
        if fill_iterations_column:
            result['iterations'] = list()
        for it in xrange(results[metric_idx].AverageTrain.size()):
            iteration = results[metric_idx].Iterations[it]
            if fill_iterations_column:
                result['iterations'].append(iteration)
            else:
                # ensure that all metrics have the same iterations specified
                assert(result['iterations'][it] == iteration)
            result["test-" + metric_name + "-mean"].append(results[metric_idx].AverageTest[it])
            result["test-" + metric_name + "-std"].append(results[metric_idx].StdDevTest[it])
            result["train-" + metric_name + "-mean"].append(results[metric_idx].AverageTrain[it])
            result["train-" + metric_name + "-std"].append(results[metric_idx].StdDevTrain[it])

    if as_pandas:
        return pd.DataFrame.from_dict(result)
    return result


cdef _FloatOrStringFromString(char* s):
    cdef char* stop = NULL
    cdef double parsed = StrToD(s, &stop)
    cdef float res
    if len(s) == 0:
        return str()
    elif stop == s + len(s):
        return float(parsed)
    return to_native_str(bytes(s))


cdef _convert_to_visible_labels(EPredictionType predictionType, TVector[TVector[double]] raws, int thread_count, TFullModel* model):
    if predictionType == PyPredictionType('Class').predictionType:
        return [[_FloatOrStringFromString(value) for value \
            in ConvertTargetToExternalName([value for value in raws[0]], dereference(model))]]
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
        self.predictionType = PyPredictionType(prediction_type).predictionType
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

    def next(self):
        if self.ntree_start >= self.ntree_end:
            raise StopIteration

        dereference(self.__modelCalcerOnPool).ApplyModelMulti(
            PyPredictionType('InternalRawFormulaVal').predictionType,
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

        return _convert_to_visible_labels(self.predictionType, self.__pred, self.thread_count, self.__model)


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


cpdef _eval_metric_util(label_param, approx_param, metric, weight_param, group_id_param, thread_count):
    if (len(label_param) != len(approx_param[0])):
        raise CatBoostError('Label and approx should have same sizes.')
    doc_count = len(label_param);

    cdef TVector[float] label
    label.resize(doc_count)
    for i in range(doc_count):
        label[i] = float(label_param[i])

    approx_dimention = len(approx_param)
    cdef TVector[TVector[double]] approx
    approx.resize(approx_dimention)
    for i in range(approx_dimention):
        approx[i].resize(doc_count)
        for j in range(doc_count):
            approx[i][j] = float(approx_param[i][j])

    cdef TVector[float] weight
    if weight_param is not None:
        if (len(weight_param) != doc_count):
            raise CatBoostError('Label and weight should have same sizes.')
        weight.resize(doc_count)
        for i in range(doc_count):
            weight[i] = float(weight_param[i])

    cdef TString group_id_strbuf

    cdef TVector[TGroupId] group_id;
    if group_id_param is not None:
        if (len(group_id_param) != doc_count):
            raise CatBoostError('Label and group_id should have same sizes.')
        group_id.resize(doc_count)
        for i in range(doc_count):
            get_id_object_bytes_string_representation(group_id_param[i], &group_id_strbuf)
            group_id[i] = CalcGroupIdFor(<TStringBuf>group_id_strbuf)

    thread_count = UpdateThreadCount(thread_count);

    return EvalMetricsForUtils(label, approx, to_arcadia_string(metric), weight, group_id, thread_count)


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


cpdef is_regression_objective(loss_name):
    return IsRegressionObjective(to_arcadia_string(loss_name))


cpdef _check_train_params(dict params):
    params_to_check = params.copy()
    if 'cat_features' in params_to_check:
        del params_to_check['cat_features']
    if 'input_borders' in params_to_check:
        del params_to_check['input_borders']

    prep_params = _PreprocessParams(params_to_check)
    CheckFitParams(
        prep_params.tree,
        prep_params.customObjectiveDescriptor.Get(),
        prep_params.customMetricDescriptor.Get())


cpdef _get_gpu_device_count():
    return GetGpuDeviceCount()


cpdef _reset_trace_backend(file):
    ResetTraceBackend(to_arcadia_string(file))
