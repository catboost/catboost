"""
Template for each `dtype` helper function for hashtable

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


# ----------------------------------------------------------------------
# VectorData
# ----------------------------------------------------------------------

ctypedef struct Float64VectorData:
    float64_t *data
    Py_ssize_t n, m


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void append_data_float64(Float64VectorData *data,
                                       float64_t x) nogil:

    data.data[data.n] = x
    data.n += 1


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void append_data_int64(Int64VectorData *data,
                                       int64_t x) nogil:

    data.data[data.n] = x
    data.n += 1

ctypedef struct StringVectorData:
    char * *data
    Py_ssize_t n, m


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void append_data_string(StringVectorData *data,
                                       char * x) nogil:

    data.data[data.n] = x
    data.n += 1

ctypedef struct UInt64VectorData:
    uint64_t *data
    Py_ssize_t n, m


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void append_data_uint64(UInt64VectorData *data,
                                       uint64_t x) nogil:

    data.data[data.n] = x
    data.n += 1

ctypedef fused vector_data:
    Int64VectorData
    UInt64VectorData
    Float64VectorData
    StringVectorData

cdef inline bint needs_resize(vector_data *data) nogil:
    return data.n == data.m

# ----------------------------------------------------------------------
# Vector
# ----------------------------------------------------------------------

cdef class Float64Vector:

    cdef:
        bint external_view_exists
        Float64VectorData *data
        ndarray ao

    def __cinit__(self):
        self.data = <Float64VectorData *>PyMem_Malloc(
            sizeof(Float64VectorData))
        if not self.data:
            raise MemoryError()
        self.external_view_exists = False
        self.data.n = 0
        self.data.m = _INIT_VEC_CAP
        self.ao = np.empty(self.data.m, dtype=np.float64)
        self.data.data = <float64_t*>self.ao.data

    cdef resize(self):
        self.data.m = max(self.data.m * 4, _INIT_VEC_CAP)
        self.ao.resize(self.data.m, refcheck=False)
        self.data.data = <float64_t*>self.ao.data

    def __dealloc__(self):
        if self.data is not NULL:
            PyMem_Free(self.data)
            self.data = NULL

    def __len__(self):
        return self.data.n

    cpdef to_array(self):
        if self.data.m != self.data.n:
            if self.external_view_exists:
                # should never happen
                raise ValueError("should have raised on append()")
            self.ao.resize(self.data.n, refcheck=False)
            self.data.m = self.data.n
        self.external_view_exists = True
        return self.ao

    cdef inline void append(self, float64_t x):

        if needs_resize(self.data):
            if self.external_view_exists:
                raise ValueError("external reference but "
                                 "Vector.resize() needed")
            self.resize()

        append_data_float64(self.data, x)

    cdef extend(self, const float64_t[:] x):
        for i in range(len(x)):
            self.append(x[i])

cdef class UInt64Vector:

    cdef:
        bint external_view_exists
        UInt64VectorData *data
        ndarray ao

    def __cinit__(self):
        self.data = <UInt64VectorData *>PyMem_Malloc(
            sizeof(UInt64VectorData))
        if not self.data:
            raise MemoryError()
        self.external_view_exists = False
        self.data.n = 0
        self.data.m = _INIT_VEC_CAP
        self.ao = np.empty(self.data.m, dtype=np.uint64)
        self.data.data = <uint64_t*>self.ao.data

    cdef resize(self):
        self.data.m = max(self.data.m * 4, _INIT_VEC_CAP)
        self.ao.resize(self.data.m, refcheck=False)
        self.data.data = <uint64_t*>self.ao.data

    def __dealloc__(self):
        if self.data is not NULL:
            PyMem_Free(self.data)
            self.data = NULL

    def __len__(self):
        return self.data.n

    cpdef to_array(self):
        if self.data.m != self.data.n:
            if self.external_view_exists:
                # should never happen
                raise ValueError("should have raised on append()")
            self.ao.resize(self.data.n, refcheck=False)
            self.data.m = self.data.n
        self.external_view_exists = True
        return self.ao

    cdef inline void append(self, uint64_t x):

        if needs_resize(self.data):
            if self.external_view_exists:
                raise ValueError("external reference but "
                                 "Vector.resize() needed")
            self.resize()

        append_data_uint64(self.data, x)

    cdef extend(self, const uint64_t[:] x):
        for i in range(len(x)):
            self.append(x[i])

cdef class Int64Vector:


    def __cinit__(self):
        self.data = <Int64VectorData *>PyMem_Malloc(
            sizeof(Int64VectorData))
        if not self.data:
            raise MemoryError()
        self.external_view_exists = False
        self.data.n = 0
        self.data.m = _INIT_VEC_CAP
        self.ao = np.empty(self.data.m, dtype=np.int64)
        self.data.data = <int64_t*>self.ao.data

    cdef resize(self):
        self.data.m = max(self.data.m * 4, _INIT_VEC_CAP)
        self.ao.resize(self.data.m, refcheck=False)
        self.data.data = <int64_t*>self.ao.data

    def __dealloc__(self):
        if self.data is not NULL:
            PyMem_Free(self.data)
            self.data = NULL

    def __len__(self):
        return self.data.n

    cpdef to_array(self):
        if self.data.m != self.data.n:
            if self.external_view_exists:
                # should never happen
                raise ValueError("should have raised on append()")
            self.ao.resize(self.data.n, refcheck=False)
            self.data.m = self.data.n
        self.external_view_exists = True
        return self.ao

    cdef inline void append(self, int64_t x):

        if needs_resize(self.data):
            if self.external_view_exists:
                raise ValueError("external reference but "
                                 "Vector.resize() needed")
            self.resize()

        append_data_int64(self.data, x)

    cdef extend(self, const int64_t[:] x):
        for i in range(len(x)):
            self.append(x[i])

cdef class StringVector:

    cdef:
        StringVectorData *data
        bint external_view_exists

    def __cinit__(self):
        self.data = <StringVectorData *>PyMem_Malloc(sizeof(StringVectorData))
        if not self.data:
            raise MemoryError()
        self.external_view_exists = False
        self.data.n = 0
        self.data.m = _INIT_VEC_CAP
        self.data.data = <char **>malloc(self.data.m * sizeof(char *))
        if not self.data.data:
            raise MemoryError()

    cdef resize(self):
        cdef:
            char **orig_data
            Py_ssize_t i, m

        m = self.data.m
        self.data.m = max(self.data.m * 4, _INIT_VEC_CAP)

        orig_data = self.data.data
        self.data.data = <char **>malloc(self.data.m * sizeof(char *))
        if not self.data.data:
            raise MemoryError()
        for i in range(m):
            self.data.data[i] = orig_data[i]

    def __dealloc__(self):
        if self.data is not NULL:
            if self.data.data is not NULL:
                free(self.data.data)
            PyMem_Free(self.data)
            self.data = NULL

    def __len__(self):
        return self.data.n

    def to_array(self):
        cdef:
            ndarray ao
            Py_ssize_t n
            object val

        ao = np.empty(self.data.n, dtype=np.object)
        for i in range(self.data.n):
            val = self.data.data[i]
            ao[i] = val
        self.external_view_exists = True
        self.data.m = self.data.n
        return ao

    cdef inline void append(self, char *x):

        if needs_resize(self.data):
            self.resize()

        append_data_string(self.data, x)

    cdef extend(self, ndarray[:] x):
        for i in range(len(x)):
            self.append(x[i])


cdef class ObjectVector:

    cdef:
        PyObject **data
        Py_ssize_t n, m
        ndarray ao
        bint external_view_exists

    def __cinit__(self):
        self.external_view_exists = False
        self.n = 0
        self.m = _INIT_VEC_CAP
        self.ao = np.empty(_INIT_VEC_CAP, dtype=object)
        self.data = <PyObject**>self.ao.data

    def __len__(self):
        return self.n

    cdef inline append(self, object obj):
        if self.n == self.m:
            if self.external_view_exists:
                raise ValueError("external reference but "
                                 "Vector.resize() needed")
            self.m = max(self.m * 2, _INIT_VEC_CAP)
            self.ao.resize(self.m, refcheck=False)
            self.data = <PyObject**>self.ao.data

        Py_INCREF(obj)
        self.data[self.n] = <PyObject*>obj
        self.n += 1

    def to_array(self):
        if self.m != self.n:
            if self.external_view_exists:
                raise ValueError("should have raised on append()")
            self.ao.resize(self.n, refcheck=False)
            self.m = self.n
        self.external_view_exists = True
        return self.ao

    cdef extend(self, ndarray[:] x):
        for i in range(len(x)):
            self.append(x[i])

# ----------------------------------------------------------------------
# HashTable
# ----------------------------------------------------------------------


cdef class HashTable:

    pass

cdef class Float64HashTable(HashTable):

    def __cinit__(self, int64_t size_hint=1):
        self.table = kh_init_float64()
        if size_hint is not None:
            size_hint = min(size_hint, _SIZE_HINT_LIMIT)
            kh_resize_float64(self.table, size_hint)

    def __len__(self):
        return self.table.size

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_float64(self.table)
            self.table = NULL

    def __contains__(self, object key):
        cdef khiter_t k
        k = kh_get_float64(self.table, key)
        return k != self.table.n_buckets

    def sizeof(self, deep=False):
        """ return the size of my table in bytes """
        return self.table.n_buckets * (sizeof(float64_t) +  # keys
                                       sizeof(Py_ssize_t) +  # vals
                                       sizeof(uint32_t))  # flags

    cpdef get_item(self, float64_t val):
        cdef khiter_t k
        k = kh_get_float64(self.table, val)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(val)

    cpdef set_item(self, float64_t key, Py_ssize_t val):
        cdef:
            khiter_t k
            int ret = 0

        k = kh_put_float64(self.table, key, &ret)
        self.table.keys[k] = key
        if kh_exist_float64(self.table, k):
            self.table.vals[k] = val
        else:
            raise KeyError(key)

    @cython.boundscheck(False)
    def map(self, const float64_t[:] keys, const int64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            float64_t key
            khiter_t k

        with nogil:
            for i in range(n):
                key = keys[i]
                k = kh_put_float64(self.table, key, &ret)
                self.table.vals[k] = <Py_ssize_t>values[i]

    @cython.boundscheck(False)
    def map_locations(self, ndarray[float64_t, ndim=1] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            float64_t val
            khiter_t k

        with nogil:
            for i in range(n):
                val = values[i]
                k = kh_put_float64(self.table, val, &ret)
                self.table.vals[k] = i

    @cython.boundscheck(False)
    def lookup(self, const float64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            float64_t val
            khiter_t k
            int64_t[:] locs = np.empty(n, dtype=np.int64)

        with nogil:
            for i in range(n):
                val = values[i]
                k = kh_get_float64(self.table, val)
                if k != self.table.n_buckets:
                    locs[i] = self.table.vals[k]
                else:
                    locs[i] = -1

        return np.asarray(locs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _unique(self, const float64_t[:] values, Float64Vector uniques,
                Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                object na_value=None, bint ignore_na=False,
                bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[float64]
            Array of values of which unique will be calculated
        uniques : Float64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : boolean, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[float64]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse=True)
            The labels from values to uniques
        """
        cdef:
            Py_ssize_t i, idx, count = count_prior, n = len(values)
            int64_t[:] labels
            int ret = 0
            float64_t val, na_value2
            khiter_t k
            Float64VectorData *ud
            bint use_na_value

        if return_inverse:
            labels = np.empty(n, dtype=np.int64)
        ud = uniques.data
        use_na_value = na_value is not None

        if use_na_value:
            # We need this na_value2 because we want to allow users
            # to *optionally* specify an NA sentinel *of the correct* type.
            # We use None, to make it optional, which requires `object` type
            # for the parameter. To please the compiler, we use na_value2,
            # which is only used if it's *specified*.
            na_value2 = <float64_t>na_value
        else:
            na_value2 = np.nan

        with nogil:
            for i in range(n):
                val = values[i]

                if ignore_na and (val != val
                                  or (use_na_value and val == na_value2)):
                    # if missing values do not count as unique values (i.e. if
                    # ignore_na is True), skip the hashtable entry for them,
                    # and replace the corresponding label with na_sentinel
                    labels[i] = na_sentinel
                    continue

                k = kh_get_float64(self.table, val)

                if k == self.table.n_buckets:
                    # k hasn't been seen yet
                    k = kh_put_float64(self.table, val, &ret)

                    if needs_resize(ud):
                        with gil:
                            if uniques.external_view_exists:
                                raise ValueError("external reference to "
                                                 "uniques held, but "
                                                 "Vector.resize() needed")
                            uniques.resize()
                    append_data_float64(ud, val)
                    if return_inverse:
                        self.table.vals[k] = count
                        labels[i] = count
                        count += 1
                elif return_inverse:
                    # k falls into a previous bucket
                    # only relevant in case we need to construct the inverse
                    idx = self.table.vals[k]
                    labels[i] = idx

        if return_inverse:
            return uniques.to_array(), np.asarray(labels)
        return uniques.to_array()

    def unique(self, const float64_t[:] values, bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[float64]
            Array of values of which unique will be calculated
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[float64]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse)
            The labels from values to uniques
        """
        uniques = Float64Vector()
        return self._unique(values, uniques, ignore_na=False,
                            return_inverse=return_inverse)

    def factorize(self, const float64_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None):
        """
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[float64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.

        Returns
        -------
        uniques : ndarray[float64]
            Unique values of input, not sorted
        labels : ndarray[int64]
            The labels from values to uniques
        """
        uniques_vector = Float64Vector()
        return self._unique(values, uniques_vector, na_sentinel=na_sentinel,
                            na_value=na_value, ignore_na=True,
                            return_inverse=True)

    def get_labels(self, const float64_t[:] values, Float64Vector uniques,
                   Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                   object na_value=None):
        _, labels = self._unique(values, uniques, count_prior=count_prior,
                                 na_sentinel=na_sentinel, na_value=na_value,
                                 ignore_na=True, return_inverse=True)
        return labels

    @cython.boundscheck(False)
    def get_labels_groupby(self, const float64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int64_t[:] labels
            Py_ssize_t idx, count = 0
            int ret = 0
            float64_t val
            khiter_t k
            Float64Vector uniques = Float64Vector()
            Float64VectorData *ud

        labels = np.empty(n, dtype=np.int64)
        ud = uniques.data

        with nogil:
            for i in range(n):
                val = values[i]

                # specific for groupby
                if val < 0:
                    labels[i] = -1
                    continue

                k = kh_get_float64(self.table, val)
                if k != self.table.n_buckets:
                    idx = self.table.vals[k]
                    labels[i] = idx
                else:
                    k = kh_put_float64(self.table, val, &ret)
                    self.table.vals[k] = count

                    if needs_resize(ud):
                        with gil:
                            uniques.resize()
                    append_data_float64(ud, val)
                    labels[i] = count
                    count += 1

        arr_uniques = uniques.to_array()

        return np.asarray(labels), arr_uniques

cdef class UInt64HashTable(HashTable):

    def __cinit__(self, int64_t size_hint=1):
        self.table = kh_init_uint64()
        if size_hint is not None:
            size_hint = min(size_hint, _SIZE_HINT_LIMIT)
            kh_resize_uint64(self.table, size_hint)

    def __len__(self):
        return self.table.size

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_uint64(self.table)
            self.table = NULL

    def __contains__(self, object key):
        cdef khiter_t k
        k = kh_get_uint64(self.table, key)
        return k != self.table.n_buckets

    def sizeof(self, deep=False):
        """ return the size of my table in bytes """
        return self.table.n_buckets * (sizeof(uint64_t) +  # keys
                                       sizeof(Py_ssize_t) +  # vals
                                       sizeof(uint32_t))  # flags

    cpdef get_item(self, uint64_t val):
        cdef khiter_t k
        k = kh_get_uint64(self.table, val)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(val)

    cpdef set_item(self, uint64_t key, Py_ssize_t val):
        cdef:
            khiter_t k
            int ret = 0

        k = kh_put_uint64(self.table, key, &ret)
        self.table.keys[k] = key
        if kh_exist_uint64(self.table, k):
            self.table.vals[k] = val
        else:
            raise KeyError(key)

    @cython.boundscheck(False)
    def map(self, const uint64_t[:] keys, const int64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            uint64_t key
            khiter_t k

        with nogil:
            for i in range(n):
                key = keys[i]
                k = kh_put_uint64(self.table, key, &ret)
                self.table.vals[k] = <Py_ssize_t>values[i]

    @cython.boundscheck(False)
    def map_locations(self, ndarray[uint64_t, ndim=1] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            uint64_t val
            khiter_t k

        with nogil:
            for i in range(n):
                val = values[i]
                k = kh_put_uint64(self.table, val, &ret)
                self.table.vals[k] = i

    @cython.boundscheck(False)
    def lookup(self, const uint64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            uint64_t val
            khiter_t k
            int64_t[:] locs = np.empty(n, dtype=np.int64)

        with nogil:
            for i in range(n):
                val = values[i]
                k = kh_get_uint64(self.table, val)
                if k != self.table.n_buckets:
                    locs[i] = self.table.vals[k]
                else:
                    locs[i] = -1

        return np.asarray(locs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _unique(self, const uint64_t[:] values, UInt64Vector uniques,
                Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                object na_value=None, bint ignore_na=False,
                bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint64]
            Array of values of which unique will be calculated
        uniques : UInt64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : boolean, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[uint64]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse=True)
            The labels from values to uniques
        """
        cdef:
            Py_ssize_t i, idx, count = count_prior, n = len(values)
            int64_t[:] labels
            int ret = 0
            uint64_t val, na_value2
            khiter_t k
            UInt64VectorData *ud
            bint use_na_value

        if return_inverse:
            labels = np.empty(n, dtype=np.int64)
        ud = uniques.data
        use_na_value = na_value is not None

        if use_na_value:
            # We need this na_value2 because we want to allow users
            # to *optionally* specify an NA sentinel *of the correct* type.
            # We use None, to make it optional, which requires `object` type
            # for the parameter. To please the compiler, we use na_value2,
            # which is only used if it's *specified*.
            na_value2 = <uint64_t>na_value
        else:
            na_value2 = 0

        with nogil:
            for i in range(n):
                val = values[i]

                if ignore_na and (val != val
                                  or (use_na_value and val == na_value2)):
                    # if missing values do not count as unique values (i.e. if
                    # ignore_na is True), skip the hashtable entry for them,
                    # and replace the corresponding label with na_sentinel
                    labels[i] = na_sentinel
                    continue

                k = kh_get_uint64(self.table, val)

                if k == self.table.n_buckets:
                    # k hasn't been seen yet
                    k = kh_put_uint64(self.table, val, &ret)

                    if needs_resize(ud):
                        with gil:
                            if uniques.external_view_exists:
                                raise ValueError("external reference to "
                                                 "uniques held, but "
                                                 "Vector.resize() needed")
                            uniques.resize()
                    append_data_uint64(ud, val)
                    if return_inverse:
                        self.table.vals[k] = count
                        labels[i] = count
                        count += 1
                elif return_inverse:
                    # k falls into a previous bucket
                    # only relevant in case we need to construct the inverse
                    idx = self.table.vals[k]
                    labels[i] = idx

        if return_inverse:
            return uniques.to_array(), np.asarray(labels)
        return uniques.to_array()

    def unique(self, const uint64_t[:] values, bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[uint64]
            Array of values of which unique will be calculated
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[uint64]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse)
            The labels from values to uniques
        """
        uniques = UInt64Vector()
        return self._unique(values, uniques, ignore_na=False,
                            return_inverse=return_inverse)

    def factorize(self, const uint64_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None):
        """
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[uint64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.

        Returns
        -------
        uniques : ndarray[uint64]
            Unique values of input, not sorted
        labels : ndarray[int64]
            The labels from values to uniques
        """
        uniques_vector = UInt64Vector()
        return self._unique(values, uniques_vector, na_sentinel=na_sentinel,
                            na_value=na_value, ignore_na=True,
                            return_inverse=True)

    def get_labels(self, const uint64_t[:] values, UInt64Vector uniques,
                   Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                   object na_value=None):
        _, labels = self._unique(values, uniques, count_prior=count_prior,
                                 na_sentinel=na_sentinel, na_value=na_value,
                                 ignore_na=True, return_inverse=True)
        return labels

    @cython.boundscheck(False)
    def get_labels_groupby(self, const uint64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int64_t[:] labels
            Py_ssize_t idx, count = 0
            int ret = 0
            uint64_t val
            khiter_t k
            UInt64Vector uniques = UInt64Vector()
            UInt64VectorData *ud

        labels = np.empty(n, dtype=np.int64)
        ud = uniques.data

        with nogil:
            for i in range(n):
                val = values[i]

                # specific for groupby

                k = kh_get_uint64(self.table, val)
                if k != self.table.n_buckets:
                    idx = self.table.vals[k]
                    labels[i] = idx
                else:
                    k = kh_put_uint64(self.table, val, &ret)
                    self.table.vals[k] = count

                    if needs_resize(ud):
                        with gil:
                            uniques.resize()
                    append_data_uint64(ud, val)
                    labels[i] = count
                    count += 1

        arr_uniques = uniques.to_array()

        return np.asarray(labels), arr_uniques

cdef class Int64HashTable(HashTable):

    def __cinit__(self, int64_t size_hint=1):
        self.table = kh_init_int64()
        if size_hint is not None:
            size_hint = min(size_hint, _SIZE_HINT_LIMIT)
            kh_resize_int64(self.table, size_hint)

    def __len__(self):
        return self.table.size

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_int64(self.table)
            self.table = NULL

    def __contains__(self, object key):
        cdef khiter_t k
        k = kh_get_int64(self.table, key)
        return k != self.table.n_buckets

    def sizeof(self, deep=False):
        """ return the size of my table in bytes """
        return self.table.n_buckets * (sizeof(int64_t) +  # keys
                                       sizeof(Py_ssize_t) +  # vals
                                       sizeof(uint32_t))  # flags

    cpdef get_item(self, int64_t val):
        cdef khiter_t k
        k = kh_get_int64(self.table, val)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(val)

    cpdef set_item(self, int64_t key, Py_ssize_t val):
        cdef:
            khiter_t k
            int ret = 0

        k = kh_put_int64(self.table, key, &ret)
        self.table.keys[k] = key
        if kh_exist_int64(self.table, k):
            self.table.vals[k] = val
        else:
            raise KeyError(key)

    @cython.boundscheck(False)
    def map(self, const int64_t[:] keys, const int64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            int64_t key
            khiter_t k

        with nogil:
            for i in range(n):
                key = keys[i]
                k = kh_put_int64(self.table, key, &ret)
                self.table.vals[k] = <Py_ssize_t>values[i]

    @cython.boundscheck(False)
    def map_locations(self, ndarray[int64_t, ndim=1] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            int64_t val
            khiter_t k

        with nogil:
            for i in range(n):
                val = values[i]
                k = kh_put_int64(self.table, val, &ret)
                self.table.vals[k] = i

    @cython.boundscheck(False)
    def lookup(self, const int64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            int64_t val
            khiter_t k
            int64_t[:] locs = np.empty(n, dtype=np.int64)

        with nogil:
            for i in range(n):
                val = values[i]
                k = kh_get_int64(self.table, val)
                if k != self.table.n_buckets:
                    locs[i] = self.table.vals[k]
                else:
                    locs[i] = -1

        return np.asarray(locs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _unique(self, const int64_t[:] values, Int64Vector uniques,
                Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                object na_value=None, bint ignore_na=False,
                bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        uniques : Int64Vector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : boolean, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse=True)
            The labels from values to uniques
        """
        cdef:
            Py_ssize_t i, idx, count = count_prior, n = len(values)
            int64_t[:] labels
            int ret = 0
            int64_t val, na_value2
            khiter_t k
            Int64VectorData *ud
            bint use_na_value

        if return_inverse:
            labels = np.empty(n, dtype=np.int64)
        ud = uniques.data
        use_na_value = na_value is not None

        if use_na_value:
            # We need this na_value2 because we want to allow users
            # to *optionally* specify an NA sentinel *of the correct* type.
            # We use None, to make it optional, which requires `object` type
            # for the parameter. To please the compiler, we use na_value2,
            # which is only used if it's *specified*.
            na_value2 = <int64_t>na_value
        else:
            na_value2 = NPY_NAT

        with nogil:
            for i in range(n):
                val = values[i]

                if ignore_na and (val != val
                                  or (use_na_value and val == na_value2)):
                    # if missing values do not count as unique values (i.e. if
                    # ignore_na is True), skip the hashtable entry for them,
                    # and replace the corresponding label with na_sentinel
                    labels[i] = na_sentinel
                    continue

                k = kh_get_int64(self.table, val)

                if k == self.table.n_buckets:
                    # k hasn't been seen yet
                    k = kh_put_int64(self.table, val, &ret)

                    if needs_resize(ud):
                        with gil:
                            if uniques.external_view_exists:
                                raise ValueError("external reference to "
                                                 "uniques held, but "
                                                 "Vector.resize() needed")
                            uniques.resize()
                    append_data_int64(ud, val)
                    if return_inverse:
                        self.table.vals[k] = count
                        labels[i] = count
                        count += 1
                elif return_inverse:
                    # k falls into a previous bucket
                    # only relevant in case we need to construct the inverse
                    idx = self.table.vals[k]
                    labels[i] = idx

        if return_inverse:
            return uniques.to_array(), np.asarray(labels)
        return uniques.to_array()

    def unique(self, const int64_t[:] values, bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse)
            The labels from values to uniques
        """
        uniques = Int64Vector()
        return self._unique(values, uniques, ignore_na=False,
                            return_inverse=return_inverse)

    def factorize(self, const int64_t[:] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None):
        """
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[int64]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.

        Returns
        -------
        uniques : ndarray[int64]
            Unique values of input, not sorted
        labels : ndarray[int64]
            The labels from values to uniques
        """
        uniques_vector = Int64Vector()
        return self._unique(values, uniques_vector, na_sentinel=na_sentinel,
                            na_value=na_value, ignore_na=True,
                            return_inverse=True)

    def get_labels(self, const int64_t[:] values, Int64Vector uniques,
                   Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                   object na_value=None):
        _, labels = self._unique(values, uniques, count_prior=count_prior,
                                 na_sentinel=na_sentinel, na_value=na_value,
                                 ignore_na=True, return_inverse=True)
        return labels

    @cython.boundscheck(False)
    def get_labels_groupby(self, const int64_t[:] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int64_t[:] labels
            Py_ssize_t idx, count = 0
            int ret = 0
            int64_t val
            khiter_t k
            Int64Vector uniques = Int64Vector()
            Int64VectorData *ud

        labels = np.empty(n, dtype=np.int64)
        ud = uniques.data

        with nogil:
            for i in range(n):
                val = values[i]

                # specific for groupby
                if val < 0:
                    labels[i] = -1
                    continue

                k = kh_get_int64(self.table, val)
                if k != self.table.n_buckets:
                    idx = self.table.vals[k]
                    labels[i] = idx
                else:
                    k = kh_put_int64(self.table, val, &ret)
                    self.table.vals[k] = count

                    if needs_resize(ud):
                        with gil:
                            uniques.resize()
                    append_data_int64(ud, val)
                    labels[i] = count
                    count += 1

        arr_uniques = uniques.to_array()

        return np.asarray(labels), arr_uniques


cdef class StringHashTable(HashTable):
    # these by-definition *must* be strings
    # or a sentinel np.nan / None missing value
    na_string_sentinel = '__nan__'

    def __init__(self, int64_t size_hint=1):
        self.table = kh_init_str()
        if size_hint is not None:
            size_hint = min(size_hint, _SIZE_HINT_LIMIT)
            kh_resize_str(self.table, size_hint)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_str(self.table)
            self.table = NULL

    def sizeof(self, deep=False):
        """ return the size of my table in bytes """
        return self.table.n_buckets * (sizeof(char *) + # keys
                                       sizeof(Py_ssize_t) + # vals
                                       sizeof(uint32_t)) # flags

    cpdef get_item(self, object val):
        cdef:
            khiter_t k
            const char *v
        v = util.get_c_string(val)

        k = kh_get_str(self.table, v)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(val)

    cpdef set_item(self, object key, Py_ssize_t val):
        cdef:
            khiter_t k
            int ret = 0
            const char *v

        v = util.get_c_string(val)

        k = kh_put_str(self.table, v, &ret)
        self.table.keys[k] = key
        if kh_exist_str(self.table, k):
            self.table.vals[k] = val
        else:
            raise KeyError(key)

    @cython.boundscheck(False)
    def get_indexer(self, ndarray[object] values):
        cdef:
            Py_ssize_t i, n = len(values)
            ndarray[int64_t] labels = np.empty(n, dtype=np.int64)
            int64_t *resbuf = <int64_t*>labels.data
            khiter_t k
            kh_str_t *table = self.table
            const char *v
            const char **vecs

        vecs = <const char **>malloc(n * sizeof(char *))
        for i in range(n):
            val = values[i]
            v = util.get_c_string(val)
            vecs[i] = v

        with nogil:
            for i in range(n):
                k = kh_get_str(table, vecs[i])
                if k != table.n_buckets:
                    resbuf[i] = table.vals[k]
                else:
                    resbuf[i] = -1

        free(vecs)
        return labels

    @cython.boundscheck(False)
    def lookup(self, ndarray[object] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            object val
            const char *v
            khiter_t k
            int64_t[:] locs = np.empty(n, dtype=np.int64)

        # these by-definition *must* be strings
        vecs = <char **>malloc(n * sizeof(char *))
        for i in range(n):
            val = values[i]

            if isinstance(val, (str, unicode)):
                v = util.get_c_string(val)
            else:
                v = util.get_c_string(self.na_string_sentinel)
            vecs[i] = v

        with nogil:
            for i in range(n):
                v = vecs[i]
                k = kh_get_str(self.table, v)
                if k != self.table.n_buckets:
                    locs[i] = self.table.vals[k]
                else:
                    locs[i] = -1

        free(vecs)
        return np.asarray(locs)

    @cython.boundscheck(False)
    def map_locations(self, ndarray[object] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            object val
            const char *v
            const char **vecs
            khiter_t k

        # these by-definition *must* be strings
        vecs = <const char **>malloc(n * sizeof(char *))
        for i in range(n):
            val = values[i]

            if isinstance(val, (str, unicode)):
                v = util.get_c_string(val)
            else:
                v = util.get_c_string(self.na_string_sentinel)
            vecs[i] = v

        with nogil:
            for i in range(n):
                v = vecs[i]
                k = kh_put_str(self.table, v, &ret)
                self.table.vals[k] = i
        free(vecs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _unique(self, ndarray[object] values, ObjectVector uniques,
                Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                object na_value=None, bint ignore_na=False,
                bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        uniques : ObjectVector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then any value
            that is not a string is considered missing. If na_value is
            not None, then _additionally_ any value "val" satisfying
            val == na_value is considered missing.
        ignore_na : boolean, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse=True)
            The labels from values to uniques
        """
        cdef:
            Py_ssize_t i, idx, count = count_prior, n = len(values)
            int64_t[:] labels
            int64_t[:] uindexer
            int ret = 0
            object val
            const char *v
            const char **vecs
            khiter_t k
            bint use_na_value

        if return_inverse:
            labels = np.zeros(n, dtype=np.int64)
        uindexer = np.empty(n, dtype=np.int64)
        use_na_value = na_value is not None

        # assign pointers and pre-filter out missing (if ignore_na)
        vecs = <const char **>malloc(n * sizeof(char *))
        for i in range(n):
            val = values[i]

            if (ignore_na
                and (not isinstance(val, (str, unicode))
                     or (use_na_value and val == na_value))):
                # if missing values do not count as unique values (i.e. if
                # ignore_na is True), we can skip the actual value, and
                # replace the label with na_sentinel directly
                labels[i] = na_sentinel
            else:
                # if ignore_na is False, we also stringify NaN/None/etc.
                v = util.get_c_string(val)
                vecs[i] = v

        # compute
        with nogil:
            for i in range(n):
                if ignore_na and labels[i] == na_sentinel:
                    # skip entries for ignored missing values (see above)
                    continue

                v = vecs[i]
                k = kh_get_str(self.table, v)
                if k == self.table.n_buckets:
                    # k hasn't been seen yet
                    k = kh_put_str(self.table, v, &ret)
                    uindexer[count] = i
                    if return_inverse:
                        self.table.vals[k] = count
                        labels[i] = <int64_t>count
                    count += 1
                elif return_inverse:
                    # k falls into a previous bucket
                    # only relevant in case we need to construct the inverse
                    idx = self.table.vals[k]
                    labels[i] = <int64_t>idx

        free(vecs)

        # uniques
        for i in range(count):
            uniques.append(values[uindexer[i]])

        if return_inverse:
            return uniques.to_array(), np.asarray(labels)
        return uniques.to_array()

    def unique(self, ndarray[object] values, bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse)
            The labels from values to uniques
        """
        uniques = ObjectVector()
        return self._unique(values, uniques, ignore_na=False,
                            return_inverse=return_inverse)

    def factorize(self, ndarray[object] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None):
        """
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then any value
            that is not a string is considered missing. If na_value is
            not None, then _additionally_ any value "val" satisfying
            val == na_value is considered missing.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[int64]
            The labels from values to uniques
        """
        uniques_vector = ObjectVector()
        return self._unique(values, uniques_vector, na_sentinel=na_sentinel,
                            na_value=na_value, ignore_na=True,
                            return_inverse=True)

    def get_labels(self, ndarray[object] values, ObjectVector uniques,
                   Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                   object na_value=None):
        _, labels = self._unique(values, uniques, count_prior=count_prior,
                                 na_sentinel=na_sentinel, na_value=na_value,
                                 ignore_na=True, return_inverse=True)
        return labels


cdef class PyObjectHashTable(HashTable):

    def __init__(self, int64_t size_hint=1):
        self.table = kh_init_pymap()
        if size_hint is not None:
            size_hint = min(size_hint, _SIZE_HINT_LIMIT)
            kh_resize_pymap(self.table, size_hint)

    def __dealloc__(self):
        if self.table is not NULL:
            kh_destroy_pymap(self.table)
            self.table = NULL

    def __len__(self):
        return self.table.size

    def __contains__(self, object key):
        cdef khiter_t k
        hash(key)

        k = kh_get_pymap(self.table, <PyObject*>key)
        return k != self.table.n_buckets

    def sizeof(self, deep=False):
        """ return the size of my table in bytes """
        return self.table.n_buckets * (sizeof(PyObject *) +  # keys
                                       sizeof(Py_ssize_t) +  # vals
                                       sizeof(uint32_t))  # flags

    cpdef get_item(self, object val):
        cdef khiter_t k

        k = kh_get_pymap(self.table, <PyObject*>val)
        if k != self.table.n_buckets:
            return self.table.vals[k]
        else:
            raise KeyError(val)

    cpdef set_item(self, object key, Py_ssize_t val):
        cdef:
            khiter_t k
            int ret = 0
            char* buf

        hash(key)

        k = kh_put_pymap(self.table, <PyObject*>key, &ret)
        # self.table.keys[k] = key
        if kh_exist_pymap(self.table, k):
            self.table.vals[k] = val
        else:
            raise KeyError(key)

    def map_locations(self, ndarray[object] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            object val
            khiter_t k

        for i in range(n):
            val = values[i]
            hash(val)

            k = kh_put_pymap(self.table, <PyObject*>val, &ret)
            self.table.vals[k] = i

    def lookup(self, ndarray[object] values):
        cdef:
            Py_ssize_t i, n = len(values)
            int ret = 0
            object val
            khiter_t k
            int64_t[:] locs = np.empty(n, dtype=np.int64)

        for i in range(n):
            val = values[i]
            hash(val)

            k = kh_get_pymap(self.table, <PyObject*>val)
            if k != self.table.n_buckets:
                locs[i] = self.table.vals[k]
            else:
                locs[i] = -1

        return np.asarray(locs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _unique(self, ndarray[object] values, ObjectVector uniques,
                Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                object na_value=None, bint ignore_na=False,
                bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        uniques : ObjectVector
            Vector into which uniques will be written
        count_prior : Py_ssize_t, default 0
            Number of existing entries in uniques
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then None _plus_
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.
        ignore_na : boolean, default False
            Whether NA-values should be ignored for calculating the uniques. If
            True, the labels corresponding to missing values will be set to
            na_sentinel.
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse=True)
            The labels from values to uniques
        """
        cdef:
            Py_ssize_t i, idx, count = count_prior, n = len(values)
            int64_t[:] labels
            int ret = 0
            object val
            khiter_t k
            bint use_na_value

        if return_inverse:
            labels = np.empty(n, dtype=np.int64)
        use_na_value = na_value is not None

        for i in range(n):
            val = values[i]
            hash(val)

            if ignore_na and ((val != val or val is None)
                              or (use_na_value and val == na_value)):
                # if missing values do not count as unique values (i.e. if
                # ignore_na is True), skip the hashtable entry for them, and
                # replace the corresponding label with na_sentinel
                labels[i] = na_sentinel
                continue

            k = kh_get_pymap(self.table, <PyObject*>val)
            if k == self.table.n_buckets:
                # k hasn't been seen yet
                k = kh_put_pymap(self.table, <PyObject*>val, &ret)
                uniques.append(val)
                if return_inverse:
                    self.table.vals[k] = count
                    labels[i] = count
                    count += 1
            elif return_inverse:
                # k falls into a previous bucket
                # only relevant in case we need to construct the inverse
                idx = self.table.vals[k]
                labels[i] = idx

        if return_inverse:
            return uniques.to_array(), np.asarray(labels)
        return uniques.to_array()

    def unique(self, ndarray[object] values, bint return_inverse=False):
        """
        Calculate unique values and labels (no sorting!)

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        return_inverse : boolean, default False
            Whether the mapping of the original array values to their location
            in the vector of uniques should be returned.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[int64] (if return_inverse)
            The labels from values to uniques
        """
        uniques = ObjectVector()
        return self._unique(values, uniques, ignore_na=False,
                            return_inverse=return_inverse)

    def factorize(self, ndarray[object] values, Py_ssize_t na_sentinel=-1,
                  object na_value=None):
        """
        Calculate unique values and labels (no sorting!)

        Missing values are not included in the "uniques" for this method.
        The labels for any missing values will be set to "na_sentinel"

        Parameters
        ----------
        values : ndarray[object]
            Array of values of which unique will be calculated
        na_sentinel : Py_ssize_t, default -1
            Sentinel value used for all NA-values in inverse
        na_value : object, default None
            Value to identify as missing. If na_value is None, then None _plus_
            any value "val" satisfying val != val is considered missing.
            If na_value is not None, then _additionally_, any value "val"
            satisfying val == na_value is considered missing.

        Returns
        -------
        uniques : ndarray[object]
            Unique values of input, not sorted
        labels : ndarray[int64]
            The labels from values to uniques
        """
        uniques_vector = ObjectVector()
        return self._unique(values, uniques_vector, na_sentinel=na_sentinel,
                            na_value=na_value, ignore_na=True,
                            return_inverse=True)

    def get_labels(self, ndarray[object] values, ObjectVector uniques,
                   Py_ssize_t count_prior=0, Py_ssize_t na_sentinel=-1,
                   object na_value=None):
        _, labels = self._unique(values, uniques, count_prior=count_prior,
                                 na_sentinel=na_sentinel, na_value=na_value,
                                 ignore_na=True, return_inverse=True)
        return labels
