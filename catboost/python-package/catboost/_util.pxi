# distutils: language = c++
# coding: utf-8
# cython: wraparound=False, boundscheck=False, initializedcheck=False
# cython: language_level=2

from cpython.version cimport PY_MAJOR_VERSION

from util.generic.array_ref cimport TConstArrayRef
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector
from util.system.types cimport ui32

import numpy as np
cimport numpy as np  # noqa


cdef extern from "Python.h":
    char* PyUnicode_AsUTF8AndSize(object s, Py_ssize_t* l)


np.import_array()

cdef _util_npbytes_ = np.bytes_
cdef _util_npunicode_ = np.str_ if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else np.unicode_


cdef inline TString to_arcadia_string(s) except *:
    cdef const unsigned char[:] bytes_s
    cdef const char* utf8_str_pointer
    cdef Py_ssize_t utf8_str_size
    cdef type s_type = type(s)
    if len(s) == 0:
        return TString()
    if s_type is unicode or s_type is _util_npunicode_:
        # Fast path for most common case(s).
        if PY_MAJOR_VERSION >= 3:
            # we fallback to calling .encode method to properly report error
            utf8_str_pointer = PyUnicode_AsUTF8AndSize(s, &utf8_str_size)
            if utf8_str_pointer != nullptr:
                return TString(utf8_str_pointer, utf8_str_size)
        else:
            tmp = (<unicode>s).encode('utf8')
            return TString(<const char*>tmp, len(tmp))
    elif s_type is bytes or s_type is _util_npbytes_:
        return TString(<const char*>s, len(s))

    if PY_MAJOR_VERSION >= 3 and hasattr(s, 'encode'):
        # encode to the specific encoding used inside of the module
        bytes_s = s.encode('utf8')
    else:
        bytes_s = s
    return TString(<const char*>&bytes_s[0], len(bytes_s))


# versions for both TStringBuf and TString are needed because of Cython's bugs that prevent conversion
# of 'const TString&' to 'TStringBuf' in generated code

cdef inline bytes to_bytes(const TString& s):
    return bytes(s.data()[:s.size()])

cdef inline to_str(const TString& s):
    cdef bytes bstr = to_bytes(s)
    if PY_MAJOR_VERSION >= 3:
        return bstr.decode()
    else:
        return bstr


# TODO: use std::vector, std::string instead of TVector, TString because STL's type conversion is supported
# natively by Cython

ctypedef fused common_tvector_type:
    int
    ui32
    double
    TString


cdef tvector_to_py(TConstArrayRef[common_tvector_type] src):
    cdef size_t i = 0
    cdef size_t src_size = src.size()
    res = []

    for i in xrange(src_size):
        if common_tvector_type is TString:
            res.append(to_str(src[i]))
        else:
            res.append(src[i])

    return res


cdef tvector_tvector_to_py(TConstArrayRef[TVector[common_tvector_type]] src):
    cdef size_t i = 0
    cdef size_t src_size = src.size()
    res = []

    for i in xrange(src_size):
        res.append(tvector_to_py(<TConstArrayRef[common_tvector_type]>src[i]))

    return res

# dummy parameter to allow to select impl by explicit 'indexing'
cdef TVector[common_tvector_type] py_to_tvector(src, common_tvector_type* dummy=NULL) except*:
    cdef TVector[common_tvector_type] res
    res.reserve(len(src))

    for e in src:
        if common_tvector_type is TString:
            res.push_back(to_arcadia_string(e))
        else:
            res.push_back(e)

    return res


# dummy parameter to allow to select impl by explicit 'indexing'
cdef TVector[TVector[common_tvector_type]] py_to_tvector_tvector(src, common_tvector_type* dummy=NULL) except*:
    cdef TVector[TVector[common_tvector_type]] res
    res.reserve(len(src))

    for e in src:
        res.push_back(py_to_tvector[common_tvector_type](e))

    return res
