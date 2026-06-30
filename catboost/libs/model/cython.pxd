# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport *

from libcpp cimport bool as bool_t

from util.generic.array_ref cimport TConstArrayRef
from util.generic.hash cimport THashMap
from util.generic.ptr cimport THolder
from util.generic.string cimport TString
from util.generic.vector cimport TVector
from util.system.types cimport ui16


cdef extern from "catboost/libs/model/features.h":
    cdef cppclass ENanValueTreatment "TFloatFeature::ENanValueTreatment":
        bool_t operator==(ENanValueTreatment)

    cdef ENanValueTreatment ENanValueTreatment_AsIs "TFloatFeature::ENanValueTreatment::AsIs"
    cdef ENanValueTreatment ENanValueTreatment_AsFalse "TFloatFeature::ENanValueTreatment::AsFalse"
    cdef ENanValueTreatment ENanValueTreatment_AsTrue "TFloatFeature::ENanValueTreatment::AsTrue"


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

    cdef cppclass TEmbeddingFeature:
        TFeaturePosition Position
        TString FeatureId
        int Dimension

    cdef cppclass TNonSymmetricTreeStepNode:
        ui16 LeftSubtreeDiff
        ui16 RightSubtreeDiff

    cdef cppclass IModelTreeData:
        TConstArrayRef[int] GetTreeSplits() except +ProcessException
        TConstArrayRef[int] GetTreeSizes() except +ProcessException
        TConstArrayRef[TNonSymmetricTreeStepNode] GetNonSymmetricStepNodes() except +ProcessException
        TConstArrayRef[ui32] GetNonSymmetricNodeIdToLeafId() except +ProcessException
        TConstArrayRef[double] GetLeafValues() except +ProcessException
        TConstArrayRef[double] GetLeafWeights() except +ProcessException

        void SetTreeSplits(const TVector[int]&) except +ProcessException
        void SetTreeSizes(const TVector[int]&) except +ProcessException
        void SetNonSymmetricStepNodes(const TVector[TNonSymmetricTreeStepNode]&) except +ProcessException
        void SetNonSymmetricNodeIdToLeafId(const TVector[ui32]&) except +ProcessException
        void SetLeafValues(const TVector[double]&) except +ProcessException
        void SetLeafWeights(const TVector[double]&) except +ProcessException
        THolder[IModelTreeData] Clone(ECloningPolicy policy) except +ProcessException

    cdef cppclass TModelTrees:
        int GetDimensionCount() noexcept
        TConstArrayRef[TCatFeature] GetCatFeatures() noexcept
        TConstArrayRef[TTextFeature] GetTextFeatures() noexcept
        TConstArrayRef[TEmbeddingFeature] GetEmbeddingFeatures() noexcept
        TConstArrayRef[TFloatFeature] GetFloatFeatures() noexcept
        void DropUnusedFeatures() except +ProcessException
        TVector[ui32] GetTreeLeafCounts() except +ProcessException
        const THolder[IModelTreeData]& GetModelTreeData() noexcept

        void ConvertObliviousToAsymmetric() except +ProcessException

    cdef cppclass TCOWTreeWrapper:
        const TModelTrees& operator*() noexcept
        const TModelTrees* Get() noexcept
        TModelTrees* GetMutable() except +ProcessException

    cdef cppclass TFullModel:
        TCOWTreeWrapper ModelTrees
        THashMap[TString, TString] ModelInfo

        bool_t operator==(const TFullModel& other) except +ProcessException
        bool_t operator!=(const TFullModel& other) except +ProcessException

        void Load(IInputStream* stream) except +ProcessException
        void Swap(TFullModel& other) except +ProcessException
        size_t GetTreeCount() except +ProcessException nogil
        size_t GetDimensionsCount() noexcept nogil
        void Truncate(size_t begin, size_t end) except +ProcessException
        bool_t IsOblivious() except +ProcessException
        TString GetLossFunctionName() except +ProcessException
        double GetBinClassProbabilityThreshold() except +ProcessException
        TVector[TJsonValue] GetModelClassLabels() except +ProcessException
        const TScaleAndBias& GetScaleAndBias() except +ProcessException
        void SetScaleAndBias(const TScaleAndBias&) except +ProcessException
        void InitNonOwning(const void* binaryBuffer, size_t binarySize) except +ProcessException
        void SetEvaluatorType(EFormulaEvaluatorType evaluatorType) except +ProcessException

    cdef cppclass EModelType:
        pass

    cdef TFullModel ReadModel(const TString& modelFile, EModelType format) except +ProcessException nogil
    cdef TFullModel ReadZeroCopyModel(const void* binaryBuffer, size_t binaryBufferSize) except +ProcessException nogil
    cdef TString SerializeModel(const TFullModel& model) except +ProcessException
    cdef TFullModel DeserializeModel(const TString& serializeModelString) except +ProcessException nogil
    cdef TVector[TString] GetModelUsedFeaturesNames(const TFullModel& model) except +ProcessException
    void SetModelExternalFeatureNames(const TVector[TString]& featureNames, TFullModel* model) except +ProcessException nogil
    cdef void SaveModelBorders(const TString& file, const TFullModel& model) except +ProcessException nogil
    cdef THashMap[int, ENanValueTreatment] GetNanTreatments(const TFullModel& model) except +ProcessException nogil
