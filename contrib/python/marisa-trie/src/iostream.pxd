from std_iostream cimport istream, ostream
from trie cimport Trie

cdef extern from "<marisa/iostream.h>" namespace "marisa" nogil:

    istream &read(istream &stream, Trie *trie)
    ostream &write(ostream &stream, Trie &trie)
