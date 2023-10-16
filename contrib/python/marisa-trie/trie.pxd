cimport agent
cimport base
cimport keyset


cdef extern from "<marisa/trie.h>" namespace "marisa" nogil:

    cdef cppclass Trie:
        Trie()

        void build(keyset.Keyset &keyset, int config_flags) except +
        void build(keyset.Keyset &keyset) except +

        void mmap(char *filename) except +
        void map(void *ptr, int size) except +

        void load(char *filename) except +
        void read(int fd) except +

        void save(char *filename) except +
        void write(int fd) except +

        bint lookup(agent.Agent &agent) except +
        void reverse_lookup(agent.Agent &agent) except +KeyError
        bint common_prefix_search(agent.Agent &agent) except +
        bint predictive_search(agent.Agent &agent) except +

        int num_tries() except +
        int num_keys() except +
        int num_nodes() except +

        base.TailMode tail_mode()
        base.NodeOrder node_order()

        bint empty() except +
        int size() except +
        int total_size() except +
        int io_size() except +

        void clear() except +
        void swap(Trie &rhs) except +
