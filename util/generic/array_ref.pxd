from libcpp cimport bool as bool_t


cdef extern from "util/generic/array_ref.h" nogil:
    cdef cppclass TArrayRef[T]:
        TArrayRef(...) except +

        T& operator[](size_t)

        bool_t empty()
        T* data() except +
        size_t size() except +
        T* begin() except +
        T* end() except +

    cdef cppclass TConstArrayRef[T]:
        TConstArrayRef(...) except +

        const T& operator[](size_t)

        bool_t empty()
        const T* data() except +
        size_t size() except +
        const T* begin() except +
        const T* end() except +
