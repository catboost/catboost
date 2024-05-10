cimport numpy as np
from numpy cimport (
    npy_float, npy_double, npy_longdouble,
    npy_cfloat, npy_cdouble, npy_clongdouble,
    npy_int, npy_long,
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
    NPY_INT, NPY_LONG)

ctypedef double complex double_complex

cdef extern from "numpy/ufuncobject.h":
    int PyUFunc_getfperr() nogil


cimport libc

from . cimport sf_error

np.import_array()
np.import_ufunc()

cdef void _set_action(sf_error.sf_error_t code,
                      sf_error.sf_action_t action) noexcept nogil:
    sf_error.set_action(code, action)
