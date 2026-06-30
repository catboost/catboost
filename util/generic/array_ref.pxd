from libcpp cimport bool as bool_t


cdef extern from "util/generic/array_ref.h" nogil:
    cdef cppclass TArrayRef[T]:
        TArrayRef(...)

        T& operator[](size_t)

        bool_t empty()
        T* data()
        size_t size()
        T* begin()
        T* end()

    cdef cppclass TConstArrayRef[T]:
        TConstArrayRef(...)

        const T& operator[](size_t)

        bool_t empty()
        const T* data()
        size_t size()
        const T* begin()
        const T* end()
