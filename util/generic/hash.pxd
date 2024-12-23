from libcpp.pair cimport pair

cdef extern from "util/generic/hash.h" nogil:
    cdef cppclass THashMap[T, U]:
        cppclass iterator:
            pair[T, U]& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)

        cppclass const_iterator(iterator):
            pass

        cppclass reverse_iterator:
            pair[T, U]& operator*()
            iterator operator++()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)

        cppclass const_reverse_iterator(reverse_iterator):
            pass

        THashMap() except +
        THashMap(THashMap&) except +
        U& operator[](T&)
        THashMap& operator=(THashMap&)

        bint operator==(THashMap&)
        bint operator!=(THashMap&)

        U& at(T&) except +
        iterator begin()
        const_iterator const_begin "begin"()
        void clear()
        size_t count(T&)
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        pair[iterator, iterator] equal_range(T&)
        void erase(iterator) except +
        void erase(iterator, iterator) except +
        size_t erase(T&)
        iterator find(T&)
        bint contains(T&)
        const_iterator const_find "find"(T&)
        pair[iterator, bint] insert(pair[T, U]) # XXX pair[T,U]&
        iterator insert(iterator, pair[T, U]) # XXX pair[T,U]&
        size_t max_size()
        size_t size()
        void swap(THashMap&)
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
