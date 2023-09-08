"""
Template for functions of IndexEngine subclasses.

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

# ----------------------------------------------------------------------
# IndexEngine Subclass Methods
# ----------------------------------------------------------------------

cdef class Float64Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Float64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val) and not util.is_float_object(val):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class MaskedFloat64Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Float64HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val) and not util.is_float_object(val):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class Float32Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Float32HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val) and not util.is_float_object(val):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class MaskedFloat32Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Float32HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val) and not util.is_float_object(val):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class Int64Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class MaskedInt64Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int64HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class Int32Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int32HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class MaskedInt32Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int32HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class Int16Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int16HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class MaskedInt16Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int16HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class Int8Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int8HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class MaskedInt8Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Int8HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        return val

cdef class UInt64Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt64HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class MaskedUInt64Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt64HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class UInt32Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt32HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class MaskedUInt32Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt32HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class UInt16Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt16HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class MaskedUInt16Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt16HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class UInt8Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt8HashTable(n)

    cdef _check_type(self, object val):
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class MaskedUInt8Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.UInt8HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if not util.is_integer_object(val):
            if util.is_float_object(val):
                # Make sure Int64Index.get_loc(2.0) works
                if val.is_integer():
                    return int(val)
            raise KeyError(val)
        if val < 0:
            # cannot have negative values with unsigned int dtype
            raise KeyError(val)
        return val

cdef class Complex64Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Complex64HashTable(n)

    cdef _check_type(self, object val):
        if (not util.is_integer_object(val)
            and not util.is_float_object(val)
            and not util.is_complex_object(val)
        ):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class MaskedComplex64Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Complex64HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if (not util.is_integer_object(val)
            and not util.is_float_object(val)
            and not util.is_complex_object(val)
        ):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class Complex128Engine(IndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Complex128HashTable(n)

    cdef _check_type(self, object val):
        if (not util.is_integer_object(val)
            and not util.is_float_object(val)
            and not util.is_complex_object(val)
        ):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val

cdef class MaskedComplex128Engine(MaskedIndexEngine):

    cdef _make_hash_table(self, Py_ssize_t n):
        return _hash.Complex128HashTable(n, uses_mask=True)

    cdef _check_type(self, object val):
        if val is C_NA:
            return val
        if (not util.is_integer_object(val)
            and not util.is_float_object(val)
            and not util.is_complex_object(val)
        ):
            # in particular catch bool and avoid casting True -> 1.0
            raise KeyError(val)
        return val
