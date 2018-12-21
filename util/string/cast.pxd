from util.generic.string cimport TString

from libcpp cimport bool as bool_t

cdef extern from "<util/string/cast.h>" nogil:
    T FromString[T](const TString&) except +
    bool_t TryFromString[T](const TString&, T&) except +
    TString ToString[T](const T&) except +

    cdef double StrToD(const char* b, char** se) except +
