from libcpp cimport bool as bool_t

from catboost.base_defs cimport ProcessException

from util.generic.string cimport TString, TStringBuf


cdef extern from "catboost/private/libs/data_util/path_with_scheme.h" namespace "NCB":
    cdef cppclass TPathWithScheme:
        TString Scheme
        TString Path
        TPathWithScheme() noexcept
        TPathWithScheme(const TStringBuf& pathWithScheme, const TStringBuf& defaultScheme) except +ProcessException
        bool_t Inited() noexcept