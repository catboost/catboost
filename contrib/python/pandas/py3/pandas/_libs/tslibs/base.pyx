"""
We define base classes that will be inherited by Timestamp, Timedelta, etc
in order to allow for fast isinstance checks without circular dependency issues.

This is analogous to core.dtypes.generic.
"""

from cpython.datetime cimport datetime

cdef extern from "Python.h":
    void PyObject_GC_UnTrack(object op)


cdef class ABCTimestamp(datetime):
    def __dealloc__(ABCTimestamp self):
        PyObject_GC_UnTrack(self)
