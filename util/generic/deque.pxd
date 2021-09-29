from libcpp.deque cimport deque


cdef extern from "<util/generic/deque.h>" nogil:
    cdef cppclass TDeque[T](deque):
        TDeque() except +
        TDeque(size_t) except +
        TDeque(size_t, const T&) except +
        TDeque(const TDeque&) except +
