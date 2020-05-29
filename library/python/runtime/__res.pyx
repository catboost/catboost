from libcpp cimport bool

from util.generic.string cimport TString, TStringBuf


cdef extern from "library/cpp/resource/resource.h" namespace "NResource":
    cdef size_t Count() except +
    cdef TStringBuf KeyByIndex(size_t idx) except +
    cdef bool FindExact(const TStringBuf key, TString* result) nogil except +


def count():
    return Count()


def key_by_index(idx):
    cdef TStringBuf ret = KeyByIndex(idx)

    return ret.Data()[:ret.Size()]


def find(s):
    cdef TString res

    if FindExact(TStringBuf(s, len(s)), &res):
        return res.c_str()[:res.length()]

    return None


include "importer.pxi"
