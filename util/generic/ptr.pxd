cdef extern from "<util/generic/ptr.h>" nogil:
    cdef cppclass THolder[T]:
        THolder(...)
        T* Get()
        void Destroy()
        T* Release()
        void Reset()
        void Reset(T*)
        void Swap(THolder[T])


    cdef THolder[T] MakeHolder[T](...)


    cdef cppclass TIntrusivePtr[T]:
        TIntrusivePtr()
        TIntrusivePtr(T*)
        TIntrusivePtr& operator=(...)
        void Reset(T*)
        T* Get()
        T* Release()
        void Drop()


    cdef cppclass TIntrusiveConstPtr[T]:
        TIntrusiveConstPtr()
        TIntrusiveConstPtr(T*)
        TIntrusiveConstPtr& operator=(...)
        void Reset(T*)
        const T* Get()
        void Drop()


    cdef cppclass TAtomicSharedPtr[T]:
        TAtomicSharedPtr()
        TAtomicSharedPtr(T*)
        T& operator*()
        T* Get()
        void Reset(T*)


    cdef TAtomicSharedPtr[T] MakeAtomicShared[T](...)
