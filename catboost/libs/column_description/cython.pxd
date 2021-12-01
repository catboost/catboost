# distutils: language = c++
# coding: utf-8
# cython: wraparound=False

from util.generic.vector cimport TVector
from util.system.types cimport ui32


cdef extern from "catboost/libs/column_description/feature_tag.h" namespace "NCB":
    cdef cppclass TTagDescription:
        TVector[ui32] Features
        float Cost
