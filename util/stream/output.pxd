from util.generic.string cimport TStringBuf


cdef extern from "<util/stream/output.h>" nogil:
    cdef cppclass IOutputStream:
        IOutputStream()
        void Flush() except+
        void Finish() except+

        void WriteChar "Write"(char) except+
        void WriteBuf "Write"(const TStringBuf) except+
        void Write(const void*, size_t) except+
