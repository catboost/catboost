cimport key

cdef extern from "<marisa/keyset.h>" namespace "marisa" nogil:
    cdef cppclass Keyset:

#        cdef enum constants:
#            BASE_BLOCK_SIZE  = 4096
#            EXTRA_BLOCK_SIZE = 1024
#            KEY_BLOCK_SIZE   = 256

        Keyset()

        void push_back(key.Key &key)
        void push_back(key.Key &key, char end_marker)

        void push_back(char *str)
        void push_back(char *ptr, int length)
        void push_back(char *ptr, int length, float weight)

        key.Key &operator[](int i)

        int num_keys()
        bint empty()

        int size()
        int total_length()

        void reset()
        void clear()
        void swap(Keyset &rhs)
