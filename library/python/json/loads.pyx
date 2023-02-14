from libcpp cimport bool

cdef extern from "library/python/json/loads.h":
    object LoadJsonFromString(const char*, size_t, bool internKeys, bool internVals, bool mayUnicode) except +


def loads(s, intern_keys = False, intern_vals = False, may_unicode = False):
    if isinstance(s, unicode):
        s = s.encode('utf-8')

    try:
        return LoadJsonFromString(s, len(s), intern_keys, intern_vals, may_unicode)
    except Exception as e:
        raise ValueError(str(e))
