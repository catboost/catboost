"""
Template for functions of IndexEngine subclasses.

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

# ----------------------------------------------------------------------
# IndexEngine Subclass Methods
# ----------------------------------------------------------------------


cdef class Float64Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.Float64HashTable(n)


    cpdef _call_map_locations(self, values):
        # self.mapping is of type Float64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_float64(values))

    cdef _get_index_values(self):
        return algos.ensure_float64(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[float64_t] values
            int count = 0


        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('float64')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class Float32Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.Float64HashTable(n)


    cpdef _call_map_locations(self, values):
        # self.mapping is of type Float64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_float64(values))

    cdef _get_index_values(self):
        return algos.ensure_float32(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[float32_t] values
            int count = 0


        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('float32')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class Int64Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.Int64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type Int64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_int64(values))

    cdef _get_index_values(self):
        return algos.ensure_int64(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int64_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('int64')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class Int32Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.Int64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type Int64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_int64(values))

    cdef _get_index_values(self):
        return algos.ensure_int32(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int32_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('int32')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class Int16Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.Int64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type Int64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_int64(values))

    cdef _get_index_values(self):
        return algos.ensure_int16(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int16_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('int16')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class Int8Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.Int64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type Int64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_int64(values))

    cdef _get_index_values(self):
        return algos.ensure_int8(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int8_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('int8')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class UInt64Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.UInt64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type UInt64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_uint64(values))

    cdef _get_index_values(self):
        return algos.ensure_uint64(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint64_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('uint64')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class UInt32Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.UInt64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type UInt64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_uint64(values))

    cdef _get_index_values(self):
        return algos.ensure_uint32(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint32_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('uint32')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class UInt16Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.UInt64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type UInt64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_uint64(values))

    cdef _get_index_values(self):
        return algos.ensure_uint16(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint16_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('uint16')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)


cdef class UInt8Engine(IndexEngine):

    cdef _make_hash_table(self, n):
        return _hash.UInt64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cpdef _call_map_locations(self, values):
        # self.mapping is of type UInt64HashTable,
        # so convert dtype of values
        self.mapping.map_locations(algos.ensure_uint64(values))

    cdef _get_index_values(self):
        return algos.ensure_uint8(self.vgetter())

    cdef _maybe_get_bool_indexer(self, object val):
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint8_t] values
            int count = 0

        if not util.is_integer_object(val):
            raise KeyError(val)

        # A view is needed for some subclasses, such as PeriodEngine:
        values = self._get_index_values().view('uint8')
        indexer = values == val
        found = np.where(indexer)[0]
        count = len(found)

        if count > 1:
            return indexer
        if count == 1:
            return int(found[0])

        raise KeyError(val)
