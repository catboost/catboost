# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport ProcessException
from catboost.libs.model.cython cimport TFullModel, TFeaturePosition

from util.generic.string cimport TString
from util.generic.vector cimport TVector

cdef extern from "catboost/libs/carry_model/carry.h":
    cdef TFullModel CarryModelByFeatureIndex(const TFullModel& model, const TVector[int]& factorFeatureIndexes, const TVector[TVector[double]]& factorsValues) except +ProcessException nogil
    cdef TFullModel CarryModelByFlatIndex(const TFullModel& model, const TVector[int]& factorFlatIndexes, const TVector[TVector[double]]& factorsValues) except +ProcessException nogil
    cdef TFullModel CarryModelByName(const TFullModel& model, const TVector[TString]& factorNames, const TVector[TVector[double]]& factorsValues) except +ProcessException nogil
    cdef TFullModel CarryModel(const TFullModel& model, const TVector[TFeaturePosition]& factors, const TVector[TVector[double]]& factorValues) except +ProcessException nogil

    cdef TFullModel UpliftModelByFeatureIndex(const TFullModel& model, const TVector[int]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) except +ProcessException nogil
    cdef TFullModel UpliftModelByFlatIndex(const TFullModel& model, const TVector[int]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) except +ProcessException nogil
    cdef TFullModel UpliftModelByName(const TFullModel& model, const TVector[TString]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) except +ProcessException nogil
    cdef TFullModel UpliftModel(const TFullModel& model, const TVector[TFeaturePosition]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) except +ProcessException nogil
