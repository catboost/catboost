"""
Template for functions of IndexEngine subclasses.

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

# ----------------------------------------------------------------------
# IndexEngine Subclass Methods
# ----------------------------------------------------------------------


cdef class Float64Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype float64_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Float64HashTable(n)

    cdef _check_type(self, object val):
        if util.is_bool_object(val):
            # avoid casting to True -> 1.0
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[float64_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[float64_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                if util.is_nan(val):
                    indexer = np.isnan(values)
                else:
                    indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class Float32Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype float32_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Float32HashTable(n)

    cdef _check_type(self, object val):
        if util.is_bool_object(val):
            # avoid casting to True -> 1.0
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[float32_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[float32_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                if util.is_nan(val):
                    indexer = np.isnan(values)
                else:
                    indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class Int64Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype int64_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[int64_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int64_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class Int32Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype int32_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int32HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[int32_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int32_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class Int16Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype int16_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int16HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[int16_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int16_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class Int8Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype int8_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int8HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[int8_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[int8_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class UInt64Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype uint64_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[uint64_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint64_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class UInt32Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype uint32_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt32HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[uint32_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint32_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class UInt16Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype uint16_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt16HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[uint16_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint16_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)


cdef class UInt8Engine(IndexEngine):
    # constructor-caller is responsible for ensuring that vgetter()
    #  returns an ndarray with dtype uint8_t

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt8HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            raise KeyError(val)

    cdef void _call_map_locations(self, ndarray[uint8_t] values):
        self.mapping.map_locations(values)

    cdef _maybe_get_bool_indexer(self, object val):
        # Returns ndarray[bool] or int
        cdef:
            ndarray[uint8_t, ndim=1, cast=True] indexer
            ndarray[intp_t, ndim=1] found
            ndarray[uint8_t, ndim=1] values
            int count = 0

        self._check_type(val)

        values = self._get_index_values()
        try:
            with warnings.catch_warnings():
                # e.g. if values is float64 and `val` is a str, suppress warning
                warnings.filterwarnings("ignore", category=FutureWarning)
                indexer = values == val
        except TypeError:
            # if the equality above returns a bool, cython will raise TypeError
            #  when trying to cast it to ndarray
            raise KeyError(val)

        return self._unpack_bool_indexer(indexer, val)
