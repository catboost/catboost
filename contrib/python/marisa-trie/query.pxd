cdef extern from "<marisa/query.h>" namespace "marisa" nogil:

    cdef cppclass Query:
        Query()
        Query(Query &query)

        #Query &operator=(Query &query)

        char operator[](int i)

        void set_str(char *str)
        void set_str(char *ptr, int length)
        void set_id(int id)

        char *ptr()
        int length()
        int id()

        void clear()
        void swap(Query &rhs)
