# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport *

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
    cdef bool_t IsClassificationObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsCvStratifiedObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsRegressionObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsMultiRegressionObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsMultiTargetObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsSurvivalRegressionObjective(const TString& lossFunction) nogil except +ProcessException
    cdef bool_t IsGroupwiseMetric(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsMultiClassCompatibleMetric(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsPairwiseMetric(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsRankingMetric(const TString& metricName) nogil except +ProcessException
    cdef bool_t IsUserDefined(const TString& metricName) nogil except +ProcessException
    cdef bool_t HasGpuImplementation(const TString& metricName) nogil except +ProcessException


cdef extern from "catboost/private/libs/options/binarization_options.h" namespace "NCatboostOptions" nogil:
    cdef cppclass TBinarizationOptions:
        TBinarizationOptions(...)


cdef extern from "catboost/private/libs/options/enums.h" namespace "NCB":
    cdef cppclass ERawTargetType:
        bool_t operator==(ERawTargetType)

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
    cdef TGroupId CalcGroupIdFor(const TStringBuf& token) except +ProcessException
    cdef TSubgroupId CalcSubgroupIdFor(const TStringBuf& token) except +ProcessException

