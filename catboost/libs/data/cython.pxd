# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport *
from catboost.libs.column_description.cython cimport TTagDescription
from catboost.libs.helpers.cython cimport *
from catboost.libs.model.cython cimport TFullModel
from catboost.private.libs.cython cimport ERawTargetType, TGroupId, TSubgroupId

from libcpp cimport bool as bool_t

from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from util.generic.hash cimport THashMap
from util.generic.maybe cimport TMaybe
from util.generic.ptr cimport THolder, TIntrusivePtr
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui32, ui64

from catboost.private.libs.data_util.cython cimport *


cdef extern from "catboost/libs/data/features_layout.h" namespace "NCB":
    cdef cppclass TFeatureMetaInfo:
        EFeatureType Type
        TString Name
        bool_t IsSparse
        bool_t IsIgnored
        bool_t IsAvailable

    cdef cppclass TFeaturesLayout:
        TFeaturesLayout() noexcept
        TFeaturesLayout(const ui32 featureCount) except +ProcessException
        TFeaturesLayout(
            const ui32 featureCount,
            const TVector[ui32]& catFeatureIndices,
            const TVector[ui32]& textFeatureIndices,
            const TVector[ui32]& embeddingFeatureIndices,
            const TVector[TString]& featureId,
            bool_t hasGraph,
            const THashMap[TString, TTagDescription]& featureTags,
            bool_t allFeaturesAreSparse
        ) except +ProcessException

        TConstArrayRef[TFeatureMetaInfo] GetExternalFeaturesMetaInfo() noexcept
        TVector[TString] GetExternalFeatureIds() except +ProcessException
        void SetExternalFeatureIds(TConstArrayRef[TString] featureIds) except +ProcessException
        EFeatureType GetExternalFeatureType(ui32 externalFeatureIdx) except +ProcessException
        ui32 GetFloatFeatureCount() noexcept
        ui32 GetCatFeatureCount() noexcept
        ui32 GetTextFeatureCount() noexcept
        ui32 GetEmbeddingFeatureCount() noexcept
        ui32 GetExternalFeatureCount() noexcept
        TConstArrayRef[ui32] GetCatFeatureInternalIdxToExternalIdx() noexcept
        TConstArrayRef[ui32] GetTextFeatureInternalIdxToExternalIdx() noexcept
        TConstArrayRef[ui32] GetEmbeddingFeatureInternalIdxToExternalIdx() noexcept

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
        bool_t HasGraph

        # ColumnsInfo is not here because it is not used for now

        ui32 GetFeatureCount() noexcept

cdef extern from "catboost/libs/data/order.h" namespace "NCB":
    cdef cppclass EObjectsOrder:
        pass

    cdef EObjectsOrder EObjectsOrder_Ordered "NCB::EObjectsOrder::Ordered"
    cdef EObjectsOrder EObjectsOrder_RandomShuffled "NCB::EObjectsOrder::RandomShuffled"
    cdef EObjectsOrder EObjectsOrder_Undefined "NCB::EObjectsOrder::Undefined"


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

        void AddEmbeddingFeature(
            ui32 localObjectIdx,
            ui32 flatFeatureIdx,
            TMaybeOwningConstArrayHolder[float] feature
        ) except +ProcessException

        void AddTarget(ui32 localObjectIdx, const TString& value) except +ProcessException
        void AddTarget(ui32 localObjectIdx, float value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) except +ProcessException
        void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) except +ProcessException
        void AddWeight(ui32 localObjectIdx, float value) except +ProcessException
        void AddGroupWeight(ui32 localObjectIdx, float value) except +ProcessException

        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException
        void SetGraph(TConstArrayRef[TPair] pairs) except +ProcessException

        void Finish() except +ProcessException

    cdef cppclass IRawFeaturesOrderDataVisitor:
        void Start(
            const TDataMetaInfo& metaInfo,
            ui32 objectCount,
            EObjectsOrder objectsOrder,
            TVector[TIntrusivePtr[IResourceHolder]] resourceHolders
        ) except +ProcessException

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

        void AddEmbeddingFeature(
            ui32 flatFeatureIdx,
            ITypedSequencePtr[TMaybeOwningConstArrayHolder[float]] features
        ) except +ProcessException

        void AddTarget(TConstArrayRef[TString] value) except +ProcessException
        void AddTarget(ITypedSequencePtr[float] value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, TConstArrayRef[TString] value) except +ProcessException
        void AddTarget(ui32 flatTargetIdx, ITypedSequencePtr[float] value) except +ProcessException
        void AddBaseline(ui32 baselineIdx, TConstArrayRef[float] value) except +ProcessException
        void AddWeights(TConstArrayRef[float] value) except +ProcessException
        void AddGroupWeights(TConstArrayRef[float] value) except +ProcessException

        void SetPairs(TConstArrayRef[TPair] pairs) except +ProcessException
        void SetGraph(TConstArrayRef[TPair] pairs) except +ProcessException

        void Finish() except +ProcessException


ctypedef TIntrusivePtr[TTargetDataProvider] TTargetDataProviderPtr
ctypedef TIntrusivePtr[TQuantizedObjectsDataProvider] TQuantizedObjectsDataProviderPtr


cdef extern from *:
    TRawObjectsDataProvider* dynamic_cast_to_TRawObjectsDataProvider "dynamic_cast<NCB::TRawObjectsDataProvider*>" (TObjectsDataProvider*)
    TQuantizedObjectsDataProvider* dynamic_cast_to_TQuantizedObjectsDataProvider "dynamic_cast<NCB::TQuantizedObjectsDataProvider*>" (TObjectsDataProvider*)


cdef extern from "catboost/libs/data/weights.h" namespace "NCB":
    cdef cppclass TWeights[T]:
        T operator[](ui32 idx) except +ProcessException
        ui32 GetSize() noexcept
        bool_t IsTrivial() noexcept
        TConstArrayRef[T] GetNonTrivialData() except +ProcessException


ctypedef TConstArrayRef[TConstArrayRef[float]] TBaselineArrayRef


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
        TMaybeOwningArrayHolder[float] ExtractValues(ILocalExecutor* localExecutor) except +ProcessException

cdef extern from "catboost/libs/data/objects.h":
    cdef void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData) except +ProcessException

cdef extern from "catboost/libs/data/objects.h" namespace "NCB":
    cdef cppclass TObjectsDataProvider:
        ui32 GetObjectCount() noexcept
        bool_t EqualTo(const TObjectsDataProvider& rhs, bool_t ignoreSparsity) except +ProcessException
        TMaybeData[TConstArrayRef[TGroupId]] GetGroupIds() except +ProcessException
        TMaybeData[TConstArrayRef[TSubgroupId]] GetSubgroupIds() except +ProcessException
        TMaybeData[TConstArrayRef[ui64]] GetTimestamp() noexcept
        const THashMap[ui32, TString]& GetCatFeaturesHashToString(ui32 catFeatureIdx) except +ProcessException
        TFeaturesLayoutPtr GetFeaturesLayout() noexcept

    cdef cppclass TRawObjectsDataProvider(TObjectsDataProvider):
        void SetGroupIds(TConstArrayRef[TStringBuf] groupStringIds) except +ProcessException
        void SetSubgroupIds(TConstArrayRef[TStringBuf] subgroupStringIds) except +ProcessException
        TMaybeData[const TFloatValuesHolder*] GetFloatFeature(ui32 floatFeatureIdx) except +ProcessException

    cdef cppclass TQuantizedObjectsDataProvider(TObjectsDataProvider):
        TQuantizedFeaturesInfoPtr GetQuantizedFeaturesInfo() except +ProcessException

    cdef THashMap[ui32, TString] MergeCatFeaturesHashToString(const TObjectsDataProvider& objectsData) except +ProcessException


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
        void SetGraph(TConstArrayRef[TPair] pairs) except +ProcessException
        void SetSubgroupIds(TConstArrayRef[TSubgroupId] subgroupIds) except +ProcessException
        void SetWeights(TConstArrayRef[float] weights) except +ProcessException
        void SetTimestamps(TConstArrayRef[ui64] timestamps) except +ProcessException

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


    cdef cppclass TTrainingDataProviders:
        TIntrusivePtr[TProcessedDataProviderTemplate[TObjectsDataProvider]] Learn
        TVector[TIntrusivePtr[TProcessedDataProviderTemplate[TObjectsDataProvider]]] Test


cdef extern from "catboost/libs/data/data_provider_builders.h" namespace "NCB":
    cdef cppclass IDataProviderBuilder:
        TDataProviderPtr GetResult() except +ProcessException

    cdef cppclass TDataProviderBuilderOptions:
        pass

    cdef void CreateDataProviderBuilderAndVisitor[IVisitor](
        const TDataProviderBuilderOptions& options,
        ILocalExecutor* localExecutor,
        THolder[IDataProviderBuilder]* dataProviderBuilder,
        IVisitor** loader
    ) except +ProcessException


cdef extern from "catboost/libs/data/target.h" namespace "NCB":
    cdef cppclass TRawTargetDataProvider:
        ERawTargetType GetTargetType() noexcept
        void GetNumericTarget(TArrayRef[TArrayRef[float]] dst) except +ProcessException
        void GetStringTargetRef(TVector[TConstArrayRef[TString]]* dst) except +ProcessException
        TMaybeData[TBaselineArrayRef] GetBaseline() noexcept
        const TWeights[float]& GetWeights() noexcept
        const TWeights[float]& GetGroupWeights() noexcept
        TConstArrayRef[TPair] GetPairs() noexcept

    cdef cppclass ETargetType:
        pass

    cdef cppclass TTargetDataSpecification:
        ETargetType Type
        TString Description

    cdef cppclass TTargetDataProvider:
        pass


cdef extern from "catboost/libs/data/feature_names_converter.h":
    cdef void ConvertFeaturesFromStringToIndices(
        const TPathWithScheme& cdFilePath,
        const TPathWithScheme& featureNamesPath,
        const TPathWithScheme& poolMetaInfoPath,
        TJsonValue* featuresArrayJson
    ) except +ProcessException
