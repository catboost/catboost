# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport ProcessException

from libcpp cimport bool as bool_t

from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui32, ui64


cdef extern from "catboost/private/libs/algo_helpers/hessian.h":
    cdef cppclass THessianInfo:
        TVector[double] Data

cdef extern from "catboost/private/libs/algo/learn_context.h":
    cdef cppclass TLearnProgress:
        pass


cdef extern from "catboost/private/libs/algo_helpers/ders_holder.h":
    cdef cppclass TDers:
        double Der1
        double Der2


cdef extern from "catboost/private/libs/options/enum_helpers.h":
    cdef bool_t IsClassificationObjective(const TString& lossFunction) except +ProcessException nogil
    cdef bool_t IsCvStratifiedObjective(const TString& lossFunction) except +ProcessException nogil
    cdef bool_t IsRegressionObjective(const TString& lossFunction) except +ProcessException nogil
    cdef bool_t IsMultiRegressionObjective(const TString& lossFunction) except +ProcessException nogil
    cdef bool_t IsMultiTargetObjective(const TString& lossFunction) except +ProcessException nogil
    cdef bool_t IsSurvivalRegressionObjective(const TString& lossFunction) except +ProcessException nogil
    cdef bool_t IsGroupwiseMetric(const TString& metricName) except +ProcessException nogil
    cdef bool_t IsMultiClassCompatibleMetric(const TString& metricName) except +ProcessException nogil
    cdef bool_t IsPairwiseMetric(const TString& metricName) except +ProcessException nogil
    cdef bool_t IsRankingMetric(const TString& metricName) except +ProcessException nogil
    cdef bool_t IsUserDefined(const TString& metricName) except +ProcessException nogil
    cdef bool_t HasGpuImplementation(const TString& metricName) except +ProcessException nogil


cdef extern from "catboost/private/libs/options/binarization_options.h" namespace "NCatboostOptions" nogil:
    cdef cppclass TBinarizationOptions:
        TBinarizationOptions(...)


cdef extern from "catboost/private/libs/options/enums.h" namespace "NCB":
    cdef cppclass ERawTargetType:
        bool_t operator==(ERawTargetType) noexcept

    cdef ERawTargetType ERawTargetType_Boolean "NCB::ERawTargetType::Boolean"
    cdef ERawTargetType ERawTargetType_Integer "NCB::ERawTargetType::Integer"
    cdef ERawTargetType ERawTargetType_Float "NCB::ERawTargetType::Float"
    cdef ERawTargetType ERawTargetType_String "NCB::ERawTargetType::String"
    cdef ERawTargetType ERawTargetType_None "NCB::ERawTargetType::None"

cdef extern from "catboost/private/libs/options/model_based_eval_options.h" namespace "NCatboostOptions" nogil:
    cdef TString GetExperimentName(ui32 featureSetIdx, ui32 foldIdx) except +ProcessException


cdef extern from "catboost/private/libs/quantization_schema/schema.h" namespace "NCB":
    cdef cppclass TPoolQuantizationSchema:
        pass


cdef extern from "catboost/private/libs/data_types/groupid.h":
    ctypedef ui64 TGroupId
    ctypedef ui32 TSubgroupId
    cdef TGroupId CalcGroupIdFor(const TStringBuf& token) noexcept
    cdef TSubgroupId CalcSubgroupIdFor(const TStringBuf& token) noexcept
