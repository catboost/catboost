cimport numpy as np
from numpy cimport (
    npy_float, npy_double, npy_longdouble,
    npy_cfloat, npy_cdouble, npy_clongdouble,
    npy_int, npy_long,
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
    NPY_INT, NPY_LONG)

cdef extern from "numpy/ufuncobject.h":
    int PyUFunc_getfperr() nogil


from . cimport sf_error
from . cimport _complexstuff
cimport scipy.special._ufuncs_cxx
from scipy.special import _ufuncs

ctypedef long double long_double
ctypedef float complex float_complex
ctypedef double complex double_complex
ctypedef long double complex long_double_complex
