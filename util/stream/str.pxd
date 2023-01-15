from util.generic.ptr cimport THolder
from util.generic.string cimport TString, TStringBuf
from util.stream.output cimport IOutputStream


cdef extern from "<util/stream/str.h>" nogil:
    cdef cppclass TStringOutput(IOutputStream):
        TStringOutput() except+
        TStringOutput(TString&) except+
        void Reserve(size_t) except+

ctypedef THolder[TStringOutput] TStringOutputPtr
