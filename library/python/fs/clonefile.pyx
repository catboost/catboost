import six

cdef extern from "sys/clonefile.h" nogil:
     int clonefile(const char * src, const char * dst, int flags)

cdef extern from "Python.h":
    ctypedef struct PyObject
    cdef PyObject *PyExc_OSError
    PyObject *PyErr_SetFromErrno(PyObject *)

cdef int _macos_clone_file(const char* src, const char* dst) except? 0:
    if clonefile(src, dst, 0) == -1:
        PyErr_SetFromErrno(PyExc_OSError)
        return 0
    return 1

def macos_clone_file(src, dst):
    return _macos_clone_file(six.ensure_binary(src), six.ensure_binary(dst)) != 0
