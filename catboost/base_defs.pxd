from libcpp cimport bool as bool_t

from util.generic.string cimport TString
from util.generic.vector cimport TVector
from util.system.types cimport ui32, i64


ctypedef const TString const_TString

ctypedef enum ECloningPolicy: Default, CloneAsSolid

cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef void ProcessException()


cdef extern from "library/cpp/threading/local_executor/local_executor.h" namespace "NPar":
    cdef cppclass ILocalExecutor:
        pass


cdef extern from "library/cpp/threading/local_executor/tbb_local_executor.h" namespace "NPar":
    cdef cppclass TTbbLocalExecutor[false]:
        TTbbLocalExecutor(int nThreads) except +ProcessException nogil


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
        EJsonValueType GetType() noexcept
        i64 GetInteger() except +ProcessException
        double GetDouble() except +ProcessException
        const TString& GetString() except +ProcessException


cdef extern from "util/stream/input.h":
    cdef cppclass IInputStream:
        size_t Read(void* buf, size_t len) except +ProcessException


cdef extern from "catboost/libs/model/enums.h":
    cdef cppclass EFormulaEvaluatorType:
        bool_t operator==(EFormulaEvaluatorType) noexcept

    cdef EFormulaEvaluatorType EFormulaEvaluatorType_CPU "EFormulaEvaluatorType::CPU"
    cdef EFormulaEvaluatorType EFormulaEvaluatorType_GPU "EFormulaEvaluatorType::GPU"


cdef extern from "catboost/libs/model/scale_and_bias.h":
    cdef cppclass TScaleAndBias:
        TScaleAndBias() noexcept
        TScaleAndBias(double scale, TVector[double]& bias) except +ProcessException

        double Scale
        TVector[double] Bias

        TVector[double]& GetBiasRef() noexcept


cdef extern from "catboost/private/libs/options/enums.h":
    cdef cppclass EFeatureType:
        bool_t operator==(EFeatureType) noexcept

    cdef EFeatureType EFeatureType_Float "EFeatureType::Float"
    cdef EFeatureType EFeatureType_Categorical "EFeatureType::Categorical"
    cdef EFeatureType EFeatureType_Text "EFeatureType::Text"
    cdef EFeatureType EFeatureType_Embedding "EFeatureType::Embedding"


    cdef cppclass EPredictionType:
        bool_t operator==(EPredictionType) noexcept

    cdef EPredictionType EPredictionType_Class "EPredictionType::Class"
    cdef EPredictionType EPredictionType_Probability "EPredictionType::Probability"
    cdef EPredictionType EPredictionType_LogProbability "EPredictionType::LogProbability"
    cdef EPredictionType EPredictionType_RawFormulaVal "EPredictionType::RawFormulaVal"
    cdef EPredictionType EPredictionType_Exponent "EPredictionType::Exponent"
    cdef EPredictionType EPredictionType_RMSEWithUncertainty "EPredictionType::RMSEWithUncertainty"

    cdef cppclass EFstrType:
        pass

    cdef cppclass EExplainableModelOutput:
        pass

    cdef cppclass ECalcTypeShapValues:
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


cdef extern from "catboost/private/libs/data_types/pair.h":
    cdef cppclass TPair:
        ui32 WinnerId
        ui32 LoserId
        float Weight
        TPair(ui32 winnerId, ui32 loserId, float weight) noexcept
