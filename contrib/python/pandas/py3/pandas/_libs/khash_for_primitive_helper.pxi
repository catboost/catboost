"""
Template for wrapping khash-tables for each primitive `dtype`

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

cdef extern from "khash_python.h":
    ctypedef struct kh_int64_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        size_t *vals

    kh_int64_t* kh_init_int64() nogil
    void kh_destroy_int64(kh_int64_t*) nogil
    void kh_clear_int64(kh_int64_t*) nogil
    khuint_t kh_get_int64(kh_int64_t*, int64_t) nogil
    void kh_resize_int64(kh_int64_t*, khuint_t) nogil
    khuint_t kh_put_int64(kh_int64_t*, int64_t, int*) nogil
    void kh_del_int64(kh_int64_t*, khuint_t) nogil

    bint kh_exist_int64(kh_int64_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_uint64_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        uint64_t *keys
        size_t *vals

    kh_uint64_t* kh_init_uint64() nogil
    void kh_destroy_uint64(kh_uint64_t*) nogil
    void kh_clear_uint64(kh_uint64_t*) nogil
    khuint_t kh_get_uint64(kh_uint64_t*, uint64_t) nogil
    void kh_resize_uint64(kh_uint64_t*, khuint_t) nogil
    khuint_t kh_put_uint64(kh_uint64_t*, uint64_t, int*) nogil
    void kh_del_uint64(kh_uint64_t*, khuint_t) nogil

    bint kh_exist_uint64(kh_uint64_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_float64_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float64_t *keys
        size_t *vals

    kh_float64_t* kh_init_float64() nogil
    void kh_destroy_float64(kh_float64_t*) nogil
    void kh_clear_float64(kh_float64_t*) nogil
    khuint_t kh_get_float64(kh_float64_t*, float64_t) nogil
    void kh_resize_float64(kh_float64_t*, khuint_t) nogil
    khuint_t kh_put_float64(kh_float64_t*, float64_t, int*) nogil
    void kh_del_float64(kh_float64_t*, khuint_t) nogil

    bint kh_exist_float64(kh_float64_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_int32_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int32_t *keys
        size_t *vals

    kh_int32_t* kh_init_int32() nogil
    void kh_destroy_int32(kh_int32_t*) nogil
    void kh_clear_int32(kh_int32_t*) nogil
    khuint_t kh_get_int32(kh_int32_t*, int32_t) nogil
    void kh_resize_int32(kh_int32_t*, khuint_t) nogil
    khuint_t kh_put_int32(kh_int32_t*, int32_t, int*) nogil
    void kh_del_int32(kh_int32_t*, khuint_t) nogil

    bint kh_exist_int32(kh_int32_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_uint32_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        uint32_t *keys
        size_t *vals

    kh_uint32_t* kh_init_uint32() nogil
    void kh_destroy_uint32(kh_uint32_t*) nogil
    void kh_clear_uint32(kh_uint32_t*) nogil
    khuint_t kh_get_uint32(kh_uint32_t*, uint32_t) nogil
    void kh_resize_uint32(kh_uint32_t*, khuint_t) nogil
    khuint_t kh_put_uint32(kh_uint32_t*, uint32_t, int*) nogil
    void kh_del_uint32(kh_uint32_t*, khuint_t) nogil

    bint kh_exist_uint32(kh_uint32_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_float32_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        float32_t *keys
        size_t *vals

    kh_float32_t* kh_init_float32() nogil
    void kh_destroy_float32(kh_float32_t*) nogil
    void kh_clear_float32(kh_float32_t*) nogil
    khuint_t kh_get_float32(kh_float32_t*, float32_t) nogil
    void kh_resize_float32(kh_float32_t*, khuint_t) nogil
    khuint_t kh_put_float32(kh_float32_t*, float32_t, int*) nogil
    void kh_del_float32(kh_float32_t*, khuint_t) nogil

    bint kh_exist_float32(kh_float32_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_int16_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int16_t *keys
        size_t *vals

    kh_int16_t* kh_init_int16() nogil
    void kh_destroy_int16(kh_int16_t*) nogil
    void kh_clear_int16(kh_int16_t*) nogil
    khuint_t kh_get_int16(kh_int16_t*, int16_t) nogil
    void kh_resize_int16(kh_int16_t*, khuint_t) nogil
    khuint_t kh_put_int16(kh_int16_t*, int16_t, int*) nogil
    void kh_del_int16(kh_int16_t*, khuint_t) nogil

    bint kh_exist_int16(kh_int16_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_uint16_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        uint16_t *keys
        size_t *vals

    kh_uint16_t* kh_init_uint16() nogil
    void kh_destroy_uint16(kh_uint16_t*) nogil
    void kh_clear_uint16(kh_uint16_t*) nogil
    khuint_t kh_get_uint16(kh_uint16_t*, uint16_t) nogil
    void kh_resize_uint16(kh_uint16_t*, khuint_t) nogil
    khuint_t kh_put_uint16(kh_uint16_t*, uint16_t, int*) nogil
    void kh_del_uint16(kh_uint16_t*, khuint_t) nogil

    bint kh_exist_uint16(kh_uint16_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_int8_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int8_t *keys
        size_t *vals

    kh_int8_t* kh_init_int8() nogil
    void kh_destroy_int8(kh_int8_t*) nogil
    void kh_clear_int8(kh_int8_t*) nogil
    khuint_t kh_get_int8(kh_int8_t*, int8_t) nogil
    void kh_resize_int8(kh_int8_t*, khuint_t) nogil
    khuint_t kh_put_int8(kh_int8_t*, int8_t, int*) nogil
    void kh_del_int8(kh_int8_t*, khuint_t) nogil

    bint kh_exist_int8(kh_int8_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_uint8_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        uint8_t *keys
        size_t *vals

    kh_uint8_t* kh_init_uint8() nogil
    void kh_destroy_uint8(kh_uint8_t*) nogil
    void kh_clear_uint8(kh_uint8_t*) nogil
    khuint_t kh_get_uint8(kh_uint8_t*, uint8_t) nogil
    void kh_resize_uint8(kh_uint8_t*, khuint_t) nogil
    khuint_t kh_put_uint8(kh_uint8_t*, uint8_t, int*) nogil
    void kh_del_uint8(kh_uint8_t*, khuint_t) nogil

    bint kh_exist_uint8(kh_uint8_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_complex64_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        khcomplex64_t *keys
        size_t *vals

    kh_complex64_t* kh_init_complex64() nogil
    void kh_destroy_complex64(kh_complex64_t*) nogil
    void kh_clear_complex64(kh_complex64_t*) nogil
    khuint_t kh_get_complex64(kh_complex64_t*, khcomplex64_t) nogil
    void kh_resize_complex64(kh_complex64_t*, khuint_t) nogil
    khuint_t kh_put_complex64(kh_complex64_t*, khcomplex64_t, int*) nogil
    void kh_del_complex64(kh_complex64_t*, khuint_t) nogil

    bint kh_exist_complex64(kh_complex64_t*, khiter_t) nogil

cdef extern from "khash_python.h":
    ctypedef struct kh_complex128_t:
        khuint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        khcomplex128_t *keys
        size_t *vals

    kh_complex128_t* kh_init_complex128() nogil
    void kh_destroy_complex128(kh_complex128_t*) nogil
    void kh_clear_complex128(kh_complex128_t*) nogil
    khuint_t kh_get_complex128(kh_complex128_t*, khcomplex128_t) nogil
    void kh_resize_complex128(kh_complex128_t*, khuint_t) nogil
    khuint_t kh_put_complex128(kh_complex128_t*, khcomplex128_t, int*) nogil
    void kh_del_complex128(kh_complex128_t*, khuint_t) nogil

    bint kh_exist_complex128(kh_complex128_t*, khiter_t) nogil
