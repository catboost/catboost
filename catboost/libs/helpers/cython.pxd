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


cdef extern from "catboost/libs/helpers/wx_test.h" nogil:
    cdef cppclass TWxTestResult:
        double WPlus
        double WMinus
        double PValue
    cdef TWxTestResult WxTest(const TVector[double]& baseline, const TVector[double]& test) nogil except +ProcessException


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

    cdef ITypedSequencePtr[TMaybeOwningConstArrayHolder[TInterfaceValue]] MakeTypeCastArraysHolderFromVector[
        TInterfaceValue,
        TStoredValue
    ](
        TVector[TMaybeOwningConstArrayHolder[TStoredValue]]& values
    ) except +ProcessException


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
