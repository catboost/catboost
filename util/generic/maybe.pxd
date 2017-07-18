cdef extern from "<util/generic/maybe.h>" nogil:
    cdef cppclass TNothing:
        pass

    cdef TNothing Nothing()

    cdef cppclass TMaybe[T]:
        TMaybe(...) except +

        TMaybe& operator=(...) except +

        void ConstructInPlace(...) except +
        void Clear() except +

        bint Defined()
        bint Empty()

        void CheckDefined() except +

        T* Get() except +
        T& GetRef() except +

        T GetOrElse(T&) except +
        TMaybe OrElse(TMaybe&) except +

        TMaybe[U] Cast[U]() except +

        void Swap(TMaybe& other) except +

        bint operator ==[U](U&) except +
        bint operator !=[U](U&) except +
        bint operator <[U](U&) except +
        bint operator >[U](U&) except +
        bint operator <=[U](U&) except +
        bint operator >=[U](U&) except +
