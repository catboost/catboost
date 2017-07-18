cdef extern from "util/digest/fnv.h":
    T FnvHash[T](const char* buf, size_t len) nogil
