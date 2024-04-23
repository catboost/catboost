cdef extern from "<marisa/key.h>" namespace "marisa" nogil:

    cdef cppclass Key:
        Key()
        Key(Key &query)

        #Key &operator=(Key &query)

        char operator[](int i)

        void set_str(char *str)
        void set_str(char *ptr, int length)
        void set_id(int id)
        void set_weight(float weight)

        char *ptr()
        int length()
        int id()
        float weight()

        void clear()
        void swap(Key &rhs)
