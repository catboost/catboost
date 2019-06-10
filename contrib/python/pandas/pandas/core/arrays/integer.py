import copy
import sys
import warnings

import numpy as np

from pandas._libs import lib
from pandas.compat import range, set_function_name, string_types
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import (
    is_bool_dtype, is_float, is_float_dtype, is_integer, is_integer_dtype,
    is_list_like, is_object_dtype, is_scalar)
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.generic import ABCIndexClass, ABCSeries
from pandas.core.dtypes.missing import isna, notna

from pandas.core import nanops
from pandas.core.arrays import ExtensionArray, ExtensionOpsMixin
from pandas.core.tools.numeric import to_numeric


class _IntegerDtype(ExtensionDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    _IntegerDtype. For example we have Int8Dtype to represnt signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """
    name = None
    base = None
    type = None
    na_value = np.nan

    def __repr__(self):
        sign = 'U' if self.is_unsigned_integer else ''
        return "{sign}Int{size}Dtype()".format(sign=sign,
                                               size=8 * self.itemsize)

    @cache_readonly
    def is_signed_integer(self):
        return self.kind == 'i'

    @cache_readonly
    def is_unsigned_integer(self):
        return self.kind == 'u'

    @property
    def _is_numeric(self):
        return True

    @cache_readonly
    def numpy_dtype(self):
        """ Return an instance of our numpy dtype """
        return np.dtype(self.type)

    @cache_readonly
    def kind(self):
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self):
        """ Return the number of bytes in this dtype """
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype

        Returns
        -------
        type
        """
        return IntegerArray

    @classmethod
    def construct_from_string(cls, string):
        """
        Construction from a string, raise a TypeError if not
        possible
        """
        if string == cls.name:
            return cls()
        raise TypeError("Cannot construct a '{}' from "
                        "'{}'".format(cls, string))


def integer_array(values, dtype=None, copy=False):
    """
    Infer and return an integer array of the values.

    Parameters
    ----------
    values : 1D list-like
    dtype : dtype, optional
        dtype to coerce
    copy : boolean, default False

    Returns
    -------
    IntegerArray

    Raises
    ------
    TypeError if incompatible types
    """
    values, mask = coerce_to_array(values, dtype=dtype, copy=copy)
    return IntegerArray(values, mask)


def safe_cast(values, dtype, copy):
    """
    Safely cast the values to the dtype if they
    are equivalent, meaning floats must be equivalent to the
    ints.

    """

    try:
        return values.astype(dtype, casting='safe', copy=copy)
    except TypeError:

        casted = values.astype(dtype, copy=copy)
        if (casted == values).all():
            return casted

        raise TypeError("cannot safely cast non-equivalent {} to {}".format(
            values.dtype, np.dtype(dtype)))


def coerce_to_array(values, dtype, mask=None, copy=False):
    """
    Coerce the input values array to numpy arrays with a mask

    Parameters
    ----------
    values : 1D list-like
    dtype : integer dtype
    mask : boolean 1D array, optional
    copy : boolean, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    # if values is integer numpy array, preserve it's dtype
    if dtype is None and hasattr(values, 'dtype'):
        if is_integer_dtype(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        if (isinstance(dtype, string_types) and
                (dtype.startswith("Int") or dtype.startswith("UInt"))):
            # Avoid DeprecationWarning from NumPy about np.dtype("Int64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not issubclass(type(dtype), _IntegerDtype):
            try:
                dtype = _dtypes[str(np.dtype(dtype))]
            except KeyError:
                raise ValueError("invalid dtype specified {}".format(dtype))

    if isinstance(values, IntegerArray):
        values, mask = values._data, values._mask
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    values = np.array(values, copy=copy)
    if is_object_dtype(values):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == 'empty':
            values = np.empty(len(values))
            values.fill(np.nan)
        elif inferred_type not in ['floating', 'integer',
                                   'mixed-integer', 'mixed-integer-float']:
            raise TypeError("{} cannot be converted to an IntegerDtype".format(
                values.dtype))

    elif not (is_integer_dtype(values) or is_float_dtype(values)):
        raise TypeError("{} cannot be converted to an IntegerDtype".format(
            values.dtype))

    if mask is None:
        mask = isna(values)
    else:
        assert len(mask) == len(values)

    if not values.ndim == 1:
        raise TypeError("values must be a 1D list-like")
    if not mask.ndim == 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = np.dtype('int64')
    else:
        dtype = dtype.type

    # if we are float, let's make sure that we can
    # safely cast

    # we copy as need to coerce here
    if mask.any():
        values = values.copy()
        values[mask] = 1
        values = safe_cast(values, dtype, copy=False)
    else:
        values = safe_cast(values, dtype, copy=False)

    return values, mask


class IntegerArray(ExtensionArray, ExtensionOpsMixin):
    """
    Array of integer (optional missing) values.

    .. versionadded:: 0.24.0

    .. warning::

       IntegerArray is currently experimental, and its API or internal
       implementation may change without warning.

    We represent an IntegerArray with 2 numpy arrays:

    - data: contains a numpy integer array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an IntegerArray from generic array-like input, use
    :func:`pandas.array` with one of the integer dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d integer-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Returns
    -------
    IntegerArray

    Examples
    --------
    Create an IntegerArray with :func:`pandas.array`.

    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())
    >>> int_array
    <IntegerArray>
    [1, NaN, 3]
    Length: 3, dtype: Int32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([1, None, 3], dtype='Int32')
    <IntegerArray>
    [1, NaN, 3]
    Length: 3, dtype: Int32

    >>> pd.array([1, None, 3], dtype='UInt16')
    <IntegerArray>
    [1, NaN, 3]
    Length: 3, dtype: UInt16
    """

    @cache_readonly
    def dtype(self):
        return _dtypes[str(self._data.dtype)]

    def __init__(self, values, mask, copy=False):
        if not (isinstance(values, np.ndarray)
                and is_integer_dtype(values.dtype)):
            raise TypeError("values should be integer numpy array. Use "
                            "the 'integer_array' function instead")
        if not (isinstance(mask, np.ndarray) and is_bool_dtype(mask.dtype)):
            raise TypeError("mask should be boolean numpy array. Use "
                            "the 'integer_array' function instead")

        if copy:
            values = values.copy()
            mask = mask.copy()

        self._data = values
        self._mask = mask

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return integer_array(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        scalars = to_numeric(strings, errors="raise")
        return cls._from_sequence(scalars, dtype, copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return integer_array(values, dtype=original.dtype)

    def _formatter(self, boxed=False):
        def fmt(x):
            if isna(x):
                return 'NaN'
            return str(x)
        return fmt

    def __getitem__(self, item):
        if is_integer(item):
            if self._mask[item]:
                return self.dtype.na_value
            return self._data[item]
        return type(self)(self._data[item], self._mask[item])

    def _coerce_to_ndarray(self):
        """
        coerce to an ndarary of object dtype
        """

        # TODO(jreback) make this better
        data = self._data.astype(object)
        data[self._mask] = self._na_value
        return data

    __array_priority__ = 1000  # higher than ndarray so ops dispatch to us

    def __array__(self, dtype=None):
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
        return self._coerce_to_ndarray()

    def __iter__(self):
        for i in range(len(self)):
            if self._mask[i]:
                yield self.dtype.na_value
            else:
                yield self._data[i]

    def take(self, indexer, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        # we always fill with 1 internally
        # to avoid upcasting
        data_fill_value = 1 if isna(fill_value) else fill_value
        result = take(self._data, indexer, fill_value=data_fill_value,
                      allow_fill=allow_fill)

        mask = take(self._mask, indexer, fill_value=True,
                    allow_fill=allow_fill)

        # if we are filling
        # we only fill where the indexer is null
        # not existing missing values
        # TODO(jreback) what if we have a non-na float as a fill value?
        if allow_fill and notna(fill_value):
            fill_mask = np.asarray(indexer) == -1
            result[fill_mask] = fill_value
            mask = mask ^ fill_mask

        return type(self)(result, mask, copy=False)

    def copy(self, deep=False):
        data, mask = self._data, self._mask
        if deep:
            data = copy.deepcopy(data)
            mask = copy.deepcopy(mask)
        else:
            data = data.copy()
            mask = mask.copy()
        return type(self)(data, mask, copy=False)

    def __setitem__(self, key, value):
        _is_scalar = is_scalar(value)
        if _is_scalar:
            value = [value]
        value, mask = coerce_to_array(value, dtype=self.dtype)

        if _is_scalar:
            value = value[0]
            mask = mask[0]

        self._data[key] = value
        self._mask[key] = mask

    def __len__(self):
        return len(self._data)

    @property
    def nbytes(self):
        return self._data.nbytes + self._mask.nbytes

    def isna(self):
        return self._mask

    @property
    def _na_value(self):
        return np.nan

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([x._data for x in to_concat])
        mask = np.concatenate([x._mask for x in to_concat])
        return cls(data, mask)

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array or IntegerArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray or IntegerArray
            NumPy ndarray or IntergerArray with 'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an IntegerDtype, equivalent of same_kind
            casting
        """

        # if we are astyping to an existing IntegerDtype we can fastpath
        if isinstance(dtype, _IntegerDtype):
            result = self._data.astype(dtype.numpy_dtype, copy=False)
            return type(self)(result, mask=self._mask, copy=False)

        # coerce
        data = self._coerce_to_ndarray()
        return astype_nansafe(data, dtype, copy=None)

    @property
    def _ndarray_values(self):
        # type: () -> np.ndarray
        """Internal pandas method for lossy conversion to a NumPy ndarray.

        This method is not part of the pandas interface.

        The expectation is that this is cheap to compute, and is primarily
        used for interacting with our indexers.
        """
        return self._data

    def value_counts(self, dropna=True):
        """
        Returns a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts

        """

        from pandas import Index, Series

        # compute counts on the data with no nans
        data = self._data[~self._mask]
        value_counts = Index(data).value_counts()
        array = value_counts.values

        # TODO(extension)
        # if we have allow Index to hold an ExtensionArray
        # this is easier
        index = value_counts.index.astype(object)

        # if we want nans, count the mask
        if not dropna:

            # TODO(extension)
            # appending to an Index *always* infers
            # w/o passing the dtype
            array = np.append(array, [self._mask.sum()])
            index = Index(np.concatenate(
                [index.values,
                 np.array([np.nan], dtype=object)]), dtype=object)

        return Series(array, index=index)

    def _values_for_argsort(self):
        # type: () -> ndarray
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        data = self._data.copy()
        data[self._mask] = data.min() - 1
        return data

    @classmethod
    def _create_comparison_method(cls, op):
        def cmp_method(self, other):

            op_name = op.__name__
            mask = None

            if isinstance(other, (ABCSeries, ABCIndexClass)):
                # Rely on pandas to unbox and dispatch to us.
                return NotImplemented

            if isinstance(other, IntegerArray):
                other, mask = other._data, other._mask

            elif is_list_like(other):
                other = np.asarray(other)
                if other.ndim > 0 and len(self) != len(other):
                    raise ValueError('Lengths must match to compare')

            other = lib.item_from_zerodim(other)

            # numpy will show a DeprecationWarning on invalid elementwise
            # comparisons, this will raise in the future
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "elementwise", FutureWarning)
                with np.errstate(all='ignore'):
                    result = op(self._data, other)

            # nans propagate
            if mask is None:
                mask = self._mask
            else:
                mask = self._mask | mask

            result[mask] = True if op_name == 'ne' else False
            return result

        name = '__{name}__'.format(name=op.__name__)
        return set_function_name(cmp_method, name, cls)

    def _reduce(self, name, skipna=True, **kwargs):
        data = self._data
        mask = self._mask

        # coerce to a nan-aware float if needed
        if mask.any():
            data = self._data.astype('float64')
            data[mask] = self._na_value

        op = getattr(nanops, 'nan' + name)
        result = op(data, axis=0, skipna=skipna, mask=mask)

        # if we have a boolean op, don't coerce
        if name in ['any', 'all']:
            pass

        # if we have a preservable numeric op,
        # provide coercion back to an integer type if possible
        elif name in ['sum', 'min', 'max', 'prod'] and notna(result):
            int_result = int(result)
            if int_result == result:
                result = int_result

        return result

    def _maybe_mask_result(self, result, mask, other, op_name):
        """
        Parameters
        ----------
        result : array-like
        mask : array-like bool
        other : scalar or array-like
        op_name : str
        """

        # may need to fill infs
        # and mask wraparound
        if is_float_dtype(result):
            mask |= (result == np.inf) | (result == -np.inf)

        # if we have a float operand we are by-definition
        # a float result
        # or our op is a divide
        if ((is_float_dtype(other) or is_float(other)) or
                (op_name in ['rtruediv', 'truediv', 'rdiv', 'div'])):
            result[mask] = np.nan
            return result

        return type(self)(result, mask, copy=False)

    @classmethod
    def _create_arithmetic_method(cls, op):
        def integer_arithmetic_method(self, other):

            op_name = op.__name__
            mask = None

            if isinstance(other, (ABCSeries, ABCIndexClass)):
                # Rely on pandas to unbox and dispatch to us.
                return NotImplemented

            if getattr(other, 'ndim', 0) > 1:
                raise NotImplementedError(
                    "can only perform ops with 1-d structures")

            if isinstance(other, IntegerArray):
                other, mask = other._data, other._mask

            elif getattr(other, 'ndim', None) == 0:
                other = other.item()

            elif is_list_like(other):
                other = np.asarray(other)
                if not other.ndim:
                    other = other.item()
                elif other.ndim == 1:
                    if not (is_float_dtype(other) or is_integer_dtype(other)):
                        raise TypeError(
                            "can only perform ops with numeric values")
            else:
                if not (is_float(other) or is_integer(other)):
                    raise TypeError("can only perform ops with numeric values")

            # nans propagate
            if mask is None:
                mask = self._mask
            else:
                mask = self._mask | mask

            # 1 ** np.nan is 1. So we have to unmask those.
            if op_name == 'pow':
                mask = np.where(self == 1, False, mask)

            elif op_name == 'rpow':
                mask = np.where(other == 1, False, mask)

            with np.errstate(all='ignore'):
                result = op(self._data, other)

            # divmod returns a tuple
            if op_name == 'divmod':
                div, mod = result
                return (self._maybe_mask_result(div, mask, other, 'floordiv'),
                        self._maybe_mask_result(mod, mask, other, 'mod'))

            return self._maybe_mask_result(result, mask, other, op_name)

        name = '__{name}__'.format(name=op.__name__)
        return set_function_name(integer_arithmetic_method, name, cls)


IntegerArray._add_arithmetic_ops()
IntegerArray._add_comparison_ops()


module = sys.modules[__name__]


# create the Dtype
_dtypes = {}
for dtype in ['int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64']:

    if dtype.startswith('u'):
        name = "U{}".format(dtype[1:].capitalize())
    else:
        name = dtype.capitalize()
    classname = "{}Dtype".format(name)
    numpy_dtype = getattr(np, dtype)
    attributes_dict = {'type': numpy_dtype,
                       'name': name}
    dtype_type = register_extension_dtype(
        type(classname, (_IntegerDtype, ), attributes_dict)
    )
    setattr(module, classname, dtype_type)

    _dtypes[dtype] = dtype_type()
