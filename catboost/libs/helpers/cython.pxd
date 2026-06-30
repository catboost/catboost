# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport TJsonValue, ProcessException

from libcpp cimport bool as bool_t

from util.generic.array_ref cimport TArrayRef, TConstArrayRef
from util.generic.ptr cimport TIntrusivePtr
from util.generic.string cimport TString
from util.generic.vector cimport TVector


cdef extern from "catboost/libs/helpers/wx_test.h" nogil:
    cdef cppclass TWxTestResult:
        double WPlus
        double WMinus
        double PValue
    cdef TWxTestResult WxTest(const TVector[double]& baseline, const TVector[double]& test) except +ProcessException nogil


cdef extern from "catboost/libs/helpers/resource_holder.h" namespace "NCB":
    cdef cppclass IResourceHolder:
        pass

    cdef cppclass TVectorHolder[T](IResourceHolder):
        TVector[T] Data


cdef extern from "catboost/libs/helpers/maybe_owning_array_holder.h" namespace "NCB":
    cdef cppclass TMaybeOwningArrayHolder[T]:
        @staticmethod
        TMaybeOwningArrayHolder[T] CreateNonOwning(TArrayRef[T] arrayRef) noexcept

        @staticmethod
        TMaybeOwningArrayHolder[T] CreateOwning(
            TArrayRef[T] arrayRef,
            TIntrusivePtr[IResourceHolder] resourceHolder
        ) except +ProcessException

        T operator[](size_t idx) except +ProcessException

    cdef cppclass TMaybeOwningConstArrayHolder[T]:
        @staticmethod
        TMaybeOwningConstArrayHolder[T] CreateNonOwning(TConstArrayRef[T] arrayRef) noexcept

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

cdef extern from "catboost/libs/helpers/json_helpers.h":
    cdef TString WriteTJsonValue(const TJsonValue& jsonValue) except +ProcessException nogil
