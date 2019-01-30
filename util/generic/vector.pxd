cdef extern from "<util/generic/vector.h>" nogil:
    cdef cppclass TVector[T]:
        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(iterator)
            bint operator!=(iterator)
            bint operator<(iterator)
            bint operator>(iterator)
            bint operator<=(iterator)
            bint operator>=(iterator)

        cppclass reverse_iterator:
            T& operator*()
            reverse_iterator operator++()
            reverse_iterator operator--()
            reverse_iterator operator+(size_t)
            reverse_iterator operator-(size_t)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)

        cppclass const_iterator(iterator):
            pass

        cppclass const_reverse_iterator(reverse_iterator):
            pass

        TVector() except +
        TVector(TVector&) except +
        TVector(size_t) except +
        TVector(size_t, T&) except +

        bint operator==(TVector&)
        bint operator!=(TVector&)
        bint operator<(TVector&)
        bint operator>(TVector&)
        bint operator<=(TVector&)
        bint operator>=(TVector&)

        void assign(size_t, const T&) except +
        void assign[input_iterator](input_iterator, input_iterator) except +

        T& at(size_t) except +
        T& operator[](size_t)

        T& back()
        iterator begin()
        const_iterator const_begin "begin"()
        size_t capacity()
        void clear() except +
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        iterator erase(iterator) except +
        iterator erase(iterator, iterator) except +
        T& front()
        iterator insert(iterator, const T&) except +
        void insert(iterator, size_t, const T&) except +
        void insert[Iter](iterator, Iter, Iter) except +
        size_t max_size()
        void pop_back() except +
        void push_back(T&) except +
        void emplace_back(...) except +
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        void reserve(size_t) except +
        void resize(size_t) except +
        void resize(size_t, T&) except +
        size_t size()
        void swap(TVector&) except +

        # C++11 methods
        T* data()
        void shrink_to_fit() except +
