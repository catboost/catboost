from libcpp.pair cimport pair

cdef extern from "util/system/types.h":
    ctypedef unsigned long ui64


cdef extern from "util/digest/city.h":
    ui64 CityHash64(const char* buf, size_t len) nogil
    pair[ui64, ui64] CityHash128(const char* buf, size_t len) nogil
    ui64 CityHash64WithSeed(const char* buf, size_t len, ui64 seed) nogil


cdef extern from "library/python/cityhash/hash.h":
    ui64 FileCityHash128WithSeedHigh64(const char* fpath) nogil except+
    ui64 FileCityHash64(const char* fpath) nogil except+


def hash64(content):
    cdef const char* s = content
    cdef size_t size = len(content)
    cdef ui64 res = 0

    if size > 128:
        with nogil:
            res = CityHash64(s, size)
    else:
        res = CityHash64(s, size)

    return res

def hash128(content):
    cdef const char* s = content
    cdef size_t size = len(content)
    cdef pair[ui64, ui64] res = pair[ui64, ui64](0, 0)

    if size > 128:
        with nogil:
            res = CityHash128(s, size)
    else:
        res = CityHash128(s, size)
    return res


def hash64seed(content, seed):
    cdef const char* s = content
    cdef size_t size = len(content)
    cdef ui64 _seed = seed;

    if size > 128:
        with nogil:
            res = CityHash64WithSeed(s, size, _seed)
    else:
        res = CityHash64WithSeed(s, size, _seed)

    return res


def filehash64(path):
    cdef const char* p = path
    cdef ui64 res = 0

    with nogil:
        res = FileCityHash64(p)

    return res


def filehash128high64(path):
    cdef const char* p = path
    cdef ui64 res = 0

    with nogil:
        res = FileCityHash128WithSeedHigh64(p)

    return res
