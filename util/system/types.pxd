from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cdef extern from "<util/system/types.h>" nogil:
    ctypedef int8_t i8
    ctypedef int16_t i16
    ctypedef int32_t i32
    ctypedef int64_t i64

    ctypedef uint8_t ui8
    ctypedef uint16_t ui16
    ctypedef uint32_t ui32
    ctypedef uint64_t ui64
