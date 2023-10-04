# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from catboost.base_defs cimport *
from catboost.libs.model.cython cimport TFullModel, TFeaturePosition

from util.generic.string cimport TString
from util.generic.vector cimport TVector

cdef extern from "catboost/libs/carry_model/carry.h":
    cdef TFullModel CarryModelByFeatureIndex(const TFullModel& model, const TVector[int]& factorFeatureIndexes, const TVector[TVector[double]]& factorsValues) nogil except +ProcessException
    cdef TFullModel CarryModelByFlatIndex(const TFullModel& model, const TVector[int]& factorFlatIndexes, const TVector[TVector[double]]& factorsValues) nogil except +ProcessException
    cdef TFullModel CarryModelByName(const TFullModel& model, const TVector[TString]& factorNames, const TVector[TVector[double]]& factorsValues) nogil except +ProcessException
    cdef TFullModel CarryModel(const TFullModel& model, const TVector[TFeaturePosition]& factors, const TVector[TVector[double]]& factorValues) nogil except +ProcessException

    cdef TFullModel UpliftModelByFeatureIndex(const TFullModel& model, const TVector[int]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) nogil except +ProcessException
    cdef TFullModel UpliftModelByFlatIndex(const TFullModel& model, const TVector[int]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) nogil except +ProcessException
    cdef TFullModel UpliftModelByName(const TFullModel& model, const TVector[TString]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) nogil except +ProcessException
    cdef TFullModel UpliftModel(const TFullModel& model, const TVector[TFeaturePosition]& factors, const TVector[double]& baseValues, const TVector[double]& nextValues) nogil except +ProcessException
