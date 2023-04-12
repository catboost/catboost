cimport numpy as cnp
from ...utils._typedefs cimport ITYPE_t, DTYPE_t

cnp.import_array()

from ._base cimport BaseDistancesReduction64
from ._middle_term_computer cimport MiddleTermComputer64

cdef class ArgKmin64(BaseDistancesReduction64):
    """float64 implementation of the ArgKmin."""

    cdef:
        ITYPE_t k

        ITYPE_t[:, ::1] argkmin_indices
        DTYPE_t[:, ::1] argkmin_distances

        # Used as array of pointers to private datastructures used in threads.
        DTYPE_t ** heaps_r_distances_chunks
        ITYPE_t ** heaps_indices_chunks


cdef class EuclideanArgKmin64(ArgKmin64):
    """EuclideanDistance-specialisation of ArgKmin64."""
    cdef:
        MiddleTermComputer64 middle_term_computer
        const DTYPE_t[::1] X_norm_squared
        const DTYPE_t[::1] Y_norm_squared

        bint use_squared_distances

from ._base cimport BaseDistancesReduction32
from ._middle_term_computer cimport MiddleTermComputer32

cdef class ArgKmin32(BaseDistancesReduction32):
    """float32 implementation of the ArgKmin."""

    cdef:
        ITYPE_t k

        ITYPE_t[:, ::1] argkmin_indices
        DTYPE_t[:, ::1] argkmin_distances

        # Used as array of pointers to private datastructures used in threads.
        DTYPE_t ** heaps_r_distances_chunks
        ITYPE_t ** heaps_indices_chunks


cdef class EuclideanArgKmin32(ArgKmin32):
    """EuclideanDistance-specialisation of ArgKmin32."""
    cdef:
        MiddleTermComputer32 middle_term_computer
        const DTYPE_t[::1] X_norm_squared
        const DTYPE_t[::1] Y_norm_squared

        bint use_squared_distances
