from libcpp cimport bool as bool_t

from util.generic.vector cimport TVector


cdef extern from "catboost/libs/monoforest/enums.h" namespace "NMonoForest":
    cdef cppclass EBinSplitType:
        bool_t operator==(EBinSplitType) noexcept

    cdef EBinSplitType EBinSplitType_TakeGreater "NMonoForest::EBinSplitType::TakeGreater"
    cdef EBinSplitType EBinSplitType_TakeEqual "NMonoForest::EBinSplitType::TakeBin"


cdef extern from "catboost/libs/monoforest/enums.h" namespace "NMonoForest":
    cdef cppclass EFeatureType "NMonoForest::EFeatureType":
        bool_t operator==(EFeatureType) noexcept
    cdef EFeatureType EMonoForestFeatureType_Float "NMonoForest::EFeatureType::Float"
    cdef EFeatureType EMonoForestFeatureType_OneHot "NMonoForest::EFeatureType::OneHot"


cdef extern from "catboost/libs/monoforest/interpretation.h" namespace "NMonoForest":
    cdef cppclass TBorderExplanation:
        float Border
        double ProbabilityToSatisfy
        TVector[double] ExpectedValueChange

    cdef cppclass TFeatureExplanation:
        int FeatureIdx
        EFeatureType FeatureType
        TVector[double] ExpectedBias
        TVector[TBorderExplanation] BordersExplanations