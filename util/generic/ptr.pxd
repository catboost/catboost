cdef extern from "<util/generic/ptr.h>" nogil:
    cdef cppclass THolder[T]:
        THolder()
        T* Get()
        void Destroy()
        T* Release()
        void Reset()
        void Reset(T*)
        void Swap(THolder[T])

    cdef THolder[T] MakeHolder[T](...)
