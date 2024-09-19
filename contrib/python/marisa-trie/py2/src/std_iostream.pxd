from libcpp.string cimport string

cdef extern from "<istream>" namespace "std" nogil:
    cdef cppclass istream:
        istream() except +
        istream& read (char* s, int n) except +

    cdef cppclass ostream:
        ostream() except +
        ostream& write (char* s, int n) except +

cdef extern from "<sstream>" namespace "std" nogil:

    cdef cppclass stringstream:
        stringstream()
        stringstream(string s)
        string str ()

