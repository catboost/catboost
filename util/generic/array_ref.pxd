
cdef extern from "util/generic/array_ref.h" nogil:
    cdef cppclass TArrayRef[T]:
        TArrayRef(...) except +

        T& operator[](size_t)

        T* data() except +
        size_t size() except +
        T* begin() except +
        T* end() except +

    cdef cppclass TConstArrayRef[T]:
        TConstArrayRef(...) except +

        const T& operator[](size_t)

        const T* data() except +
        size_t size() except +
        const T* begin() except +
        const T* end() except +
