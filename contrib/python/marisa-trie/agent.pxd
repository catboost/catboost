cimport query, key

cdef extern from "<marisa/agent.h>" namespace "marisa" nogil:
    cdef cppclass Agent:
        Agent() except +

        query.Query &query()
        key.Key &key()

        void set_query(char *str)
        void set_query(char *ptr, int length)
        void set_query(int key_id)

        void set_key(char *str)
        void set_key(char *ptr, int length)
        void set_key(int id)

        void clear()

        void init_state()

        void swap(Agent &rhs)
