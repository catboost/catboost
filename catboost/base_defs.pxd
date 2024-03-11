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

cimport cython
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
from util.generic.maybe cimport TMaybe
from util.generic.ptr cimport THolder, TIntrusivePtr, MakeHolder
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui8, ui16, ui32, ui64, i32, i64
from util.string.cast cimport StrToD, TryFromString, ToString

ctypedef const TString const_TString

ctypedef enum ECloningPolicy: Default, CloneAsSolid

cdef extern from "catboost/python-package/catboost/helpers.h":
    cdef void ProcessException()


cdef extern from "library/cpp/threading/local_executor/local_executor.h" namespace "NPar":
    cdef cppclass ILocalExecutor:
        pass


cdef extern from "library/cpp/threading/local_executor/tbb_local_executor.h" namespace "NPar":
    cdef cppclass TTbbLocalExecutor[false]:
        TTbbLocalExecutor(int nThreads) nogil


cdef extern from "catboost/private/libs/options/json_helper.h":
    cdef TString WriteTJsonValue(const TJsonValue& jsonValue) except +ProcessException


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


cdef extern from "util/stream/input.h":
    cdef cppclass IInputStream:
        size_t Read(void* buf, size_t len) except +ProcessException


cdef extern from "catboost/libs/model/enums.h":
    cdef cppclass EFormulaEvaluatorType:
        bool_t operator==(EFormulaEvaluatorType)

    cdef EFormulaEvaluatorType EFormulaEvaluatorType_CPU "EFormulaEvaluatorType::CPU"
    cdef EFormulaEvaluatorType EFormulaEvaluatorType_GPU "EFormulaEvaluatorType::GPU"


cdef extern from "catboost/libs/model/scale_and_bias.h":
    cdef cppclass TScaleAndBias:
        TScaleAndBias()
        TScaleAndBias(double scale, TVector[double]& bias)

        double Scale
        TVector[double] Bias

        TVector[double]& GetBiasRef()


cdef extern from "catboost/private/libs/options/enums.h":
    cdef cppclass EFeatureType:
        bool_t operator==(EFeatureType)

    cdef EFeatureType EFeatureType_Float "EFeatureType::Float"
    cdef EFeatureType EFeatureType_Categorical "EFeatureType::Categorical"
    cdef EFeatureType EFeatureType_Text "EFeatureType::Text"
    cdef EFeatureType EFeatureType_Embedding "EFeatureType::Embedding"


    cdef cppclass EPredictionType:
        bool_t operator==(EPredictionType)

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
        TPair(ui32 winnerId, ui32 loserId, float weight)
