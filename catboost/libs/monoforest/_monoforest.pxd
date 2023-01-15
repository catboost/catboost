from catboost.base_defs cimport *
from catboost.libs.model.cython cimport TFullModel

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


cdef extern from "catboost/libs/monoforest/enums.h" namespace "NMonoForest":
    cdef cppclass EBinSplitType:
        bool_t operator==(EBinSplitType)

    cdef EBinSplitType EBinSplitType_TakeGreater "NMonoForest::EBinSplitType::TakeGreater"
    cdef EBinSplitType EBinSplitType_TakeEqual "NMonoForest::EBinSplitType::TakeBin"


cdef extern from "catboost/libs/monoforest/enums.h" namespace "NMonoForest":
    cdef cppclass EFeatureType:
        bool_t operator==(EFeatureType)


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