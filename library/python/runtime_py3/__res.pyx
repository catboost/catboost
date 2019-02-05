# cython: language_level=3

from _codecs import utf_8_decode, utf_8_encode

from libcpp cimport bool


cdef extern from "util/generic/string.h":
    cdef cppclass TString:
        const char* c_str()
        size_t length()


cdef extern from "util/generic/strbuf.h":
    cdef cppclass TStringBuf:
        TStringBuf()
        TStringBuf(const char* buf, size_t len)
        const char* Data()
        size_t Size()


cdef extern from "library/resource/resource.h" namespace "NResource":
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

    if isinstance(s, str):
        s = utf_8_encode(s)[0]

    if FindExact(TStringBuf(s, len(s)), &res):
        return res.c_str()[:res.length()]

    return None


include "importer.pxi"
