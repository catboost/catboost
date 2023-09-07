from std_iostream cimport istream, ostream
from trie cimport Trie

cdef extern from "<contrib/python/marisa-trie/marisa/iostream.h>" namespace "marisa" nogil:

    istream &read(istream &stream, Trie *trie)
    ostream &write(ostream &stream, Trie &trie)
