from libcpp.pair cimport pair

cdef extern from "util/generic/hash_set.h" nogil:
    cdef cppclass THashSet[T]:
        cppclass iterator:
            T& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)

        cppclass const_iterator(iterator):
            pass

        THashSet() except +
        THashSet(THashSet&) except +
        THashSet& operator=(THashSet&)

        bint operator==(THashSet&)
        bint operator!=(THashSet&)

        iterator begin()
        const_iterator const_begin "begin"()
        void clear()
        size_t count(T&)
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        void erase(iterator) except +
        void erase(iterator, iterator) except +
        size_t erase(T&)
        iterator find(T&)
        bint contains(T&)
        const_iterator const_find "find"(T&)
        pair[iterator, bint] insert(T)
        iterator insert(iterator, T)
        size_t size()
        void swap(THashSet&)
