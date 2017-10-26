from util.generic.string cimport TString

cdef extern from "<util/string/cast.h>" nogil:
    TString ToString[T](const T&) except +
