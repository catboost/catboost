from datetime import datetime, timedelta
import operator
from textwrap import dedent
import warnings

import numpy as np

from pandas._libs import (
    Timedelta, algos as libalgos, index as libindex, join as libjoin, lib,
    tslibs)
from pandas._libs.lib import is_datetime_array
import pandas.compat as compat
from pandas.compat import range, set_function_name, u
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, Substitution, cache_readonly

from pandas.core.dtypes.cast import maybe_cast_to_integer_array
from pandas.core.dtypes.common import (
    ensure_categorical, ensure_int64, ensure_object, ensure_platform_int,
    is_bool, is_bool_dtype, is_categorical, is_categorical_dtype,
    is_datetime64_any_dtype, is_datetime64tz_dtype, is_dtype_equal,
    is_dtype_union_equal, is_extension_array_dtype, is_float, is_float_dtype,
    is_hashable, is_integer, is_integer_dtype, is_interval_dtype, is_iterator,
    is_list_like, is_object_dtype, is_period_dtype, is_scalar,
    is_signed_integer_dtype, is_timedelta64_dtype, is_unsigned_integer_dtype,
    pandas_dtype)
import pandas.core.dtypes.concat as _concat
from pandas.core.dtypes.generic import (
    ABCDataFrame, ABCDateOffset, ABCDatetimeArray, ABCIndexClass,
    ABCMultiIndex, ABCPandasArray, ABCPeriodIndex, ABCSeries,
    ABCTimedeltaArray, ABCTimedeltaIndex)
from pandas.core.dtypes.missing import array_equivalent, isna

from pandas.core import ops
from pandas.core.accessor import CachedAccessor, DirNamesMixin
import pandas.core.algorithms as algos
from pandas.core.arrays import ExtensionArray
from pandas.core.base import IndexOpsMixin, PandasObject
import pandas.core.common as com
from pandas.core.indexes.frozen import FrozenList
import pandas.core.missing as missing
from pandas.core.ops import get_op_result_name, make_invalid_op
import pandas.core.sorting as sorting
from pandas.core.strings import StringMethods

from pandas.io.formats.printing import (
    default_pprint, format_object_attrs, format_object_summary, pprint_thing)

__all__ = ['Index']

_unsortable_types = frozenset(('mixed', 'mixed-integer'))

_index_doc_kwargs = dict(klass='Index', inplace='',
                         target_klass='Index',
                         unique='Index', duplicated='np.ndarray')
_index_shared_docs = dict()


def _try_get_item(x):
    try:
        return x.item()
    except AttributeError:
        return x


def _make_comparison_op(op, cls):
    def cmp_method(self, other):
        if isinstance(other, (np.ndarray, Index, ABCSeries)):
            if other.ndim > 0 and len(self) != len(other):
                raise ValueError('Lengths must match to compare')

        if is_object_dtype(self) and not isinstance(self, ABCMultiIndex):
            # don't pass MultiIndex
            with np.errstate(all='ignore'):
                result = ops._comp_method_OBJECT_ARRAY(op, self.values, other)

        else:

            # numpy will show a DeprecationWarning on invalid elementwise
            # comparisons, this will raise in the future
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings("ignore", "elementwise", FutureWarning)
                with np.errstate(all='ignore'):
                    result = op(self.values, np.asarray(other))

        # technically we could support bool dtyped Index
        # for now just return the indexing array directly
        if is_bool_dtype(result):
            return result
        try:
            return Index(result)
        except TypeError:
            return result

    name = '__{name}__'.format(name=op.__name__)
    # TODO: docstring?
    return set_function_name(cmp_method, name, cls)


def _make_arithmetic_op(op, cls):
    def index_arithmetic_method(self, other):
        if isinstance(other, (ABCSeries, ABCDataFrame)):
            return NotImplemented
        elif isinstance(other, ABCTimedeltaIndex):
            # Defer to subclass implementation
            return NotImplemented
        elif (isinstance(other, (np.ndarray, ABCTimedeltaArray)) and
              is_timedelta64_dtype(other)):
            # GH#22390; wrap in Series for op, this will in turn wrap in
            # TimedeltaIndex, but will correctly raise TypeError instead of
            # NullFrequencyError for add/sub ops
            from pandas import Series
            other = Series(other)
            out = op(self, other)
            return Index(out, name=self.name)

        other = self._validate_for_numeric_binop(other, op)

        # handle time-based others
        if isinstance(other, (ABCDateOffset, np.timedelta64, timedelta)):
            return self._evaluate_with_timedelta_like(other, op)
        elif isinstance(other, (datetime, np.datetime64)):
            return self._evaluate_with_datetime_like(other, op)

        values = self.values
        with np.errstate(all='ignore'):
            result = op(values, other)

        result = missing.dispatch_missing(op, values, other, result)

        attrs = self._get_attributes_dict()
        attrs = self._maybe_update_attributes(attrs)
        if op is divmod:
            result = (Index(result[0], **attrs), Index(result[1], **attrs))
        else:
            result = Index(result, **attrs)
        return result

    name = '__{name}__'.format(name=op.__name__)
    # TODO: docstring?
    return set_function_name(index_arithmetic_method, name, cls)


class InvalidIndexError(Exception):
    pass


_o_dtype = np.dtype(object)
_Identity = object


def _new_Index(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__.
    """
    # required for backward compat, because PI can't be instantiated with
    # ordinals through __new__ GH #13277
    if issubclass(cls, ABCPeriodIndex):
        from pandas.core.indexes.period import _new_PeriodIndex
        return _new_PeriodIndex(cls, **d)
    return cls.__new__(cls, **d)


class Index(IndexOpsMixin, PandasObject):
    """
    Immutable ndarray implementing an ordered, sliceable set. The basic object
    storing axis labels for all pandas objects.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype (default: object)
        If dtype is None, we find the dtype that best fits the data.
        If an actual dtype is provided, we coerce to that dtype if it's safe.
        Otherwise, an error will be raised.
    copy : bool
        Make a copy of input ndarray
    name : object
        Name to be stored in the index
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible

    See Also
    ---------
    RangeIndex : Index implementing a monotonic integer range.
    CategoricalIndex : Index of :class:`Categorical` s.
    MultiIndex : A multi-level, or hierarchical, Index.
    IntervalIndex : An Index of :class:`Interval` s.
    DatetimeIndex, TimedeltaIndex, PeriodIndex
    Int64Index, UInt64Index,  Float64Index

    Notes
    -----
    An Index instance can **only** contain hashable objects

    Examples
    --------
    >>> pd.Index([1, 2, 3])
    Int64Index([1, 2, 3], dtype='int64')

    >>> pd.Index(list('abc'))
    Index(['a', 'b', 'c'], dtype='object')
    """
    # tolist is not actually deprecated, just suppressed in the __dir__
    _deprecations = DirNamesMixin._deprecations | frozenset(['tolist'])

    # To hand over control to subclasses
    _join_precedence = 1

    # Cython methods; see github.com/cython/cython/issues/2647
    #  for why we need to wrap these instead of making them class attributes
    # Moreover, cython will choose the appropriate-dtyped sub-function
    #  given the dtypes of the passed arguments
    def _left_indexer_unique(self, left, right):
        return libjoin.left_join_indexer_unique(left, right)

    def _left_indexer(self, left, right):
        return libjoin.left_join_indexer(left, right)

    def _inner_indexer(self, left, right):
        return libjoin.inner_join_indexer(left, right)

    def _outer_indexer(self, left, right):
        return libjoin.outer_join_indexer(left, right)

    _typ = 'index'
    _data = None
    _id = None
    name = None
    asi8 = None
    _comparables = ['name']
    _attributes = ['name']
    _is_numeric_dtype = False
    _can_hold_na = True

    # would we like our indexing holder to defer to us
    _defer_to_indexing = False

    # prioritize current class for _shallow_copy_with_infer,
    # used to infer integers as datetime-likes
    _infer_as_myclass = False

    _engine_type = libindex.ObjectEngine

    _accessors = {'str'}

    str = CachedAccessor("str", StringMethods)

    # --------------------------------------------------------------------
    # Constructors

    def __new__(cls, data=None, dtype=None, copy=False, name=None,
                fastpath=None, tupleize_cols=True, **kwargs):

        if name is None and hasattr(data, 'name'):
            name = data.name

        if fastpath is not None:
            warnings.warn("The 'fastpath' keyword is deprecated, and will be "
                          "removed in a future version.",
                          FutureWarning, stacklevel=2)
            if fastpath:
                return cls._simple_new(data, name)

        from .range import RangeIndex
        if isinstance(data, ABCPandasArray):
            # ensure users don't accidentally put a PandasArray in an index.
            data = data.to_numpy()

        # range
        if isinstance(data, RangeIndex):
            return RangeIndex(start=data, copy=copy, dtype=dtype, name=name)
        elif isinstance(data, range):
            return RangeIndex.from_range(data, copy=copy, dtype=dtype,
                                         name=name)

        # categorical
        elif is_categorical_dtype(data) or is_categorical_dtype(dtype):
            from .category import CategoricalIndex
            return CategoricalIndex(data, dtype=dtype, copy=copy, name=name,
                                    **kwargs)

        # interval
        elif ((is_interval_dtype(data) or is_interval_dtype(dtype)) and
              not is_object_dtype(dtype)):
            from .interval import IntervalIndex
            closed = kwargs.get('closed', None)
            return IntervalIndex(data, dtype=dtype, name=name, copy=copy,
                                 closed=closed)

        elif (is_datetime64_any_dtype(data) or
              (dtype is not None and is_datetime64_any_dtype(dtype)) or
                'tz' in kwargs):
            from pandas import DatetimeIndex

            if dtype is not None and is_dtype_equal(_o_dtype, dtype):
                # GH#23524 passing `dtype=object` to DatetimeIndex is invalid,
                #  will raise in the where `data` is already tz-aware.  So
                #  we leave it out of this step and cast to object-dtype after
                #  the DatetimeIndex construction.
                # Note we can pass copy=False because the .astype below
                #  will always make a copy
                result = DatetimeIndex(data, copy=False, name=name, **kwargs)
                return result.astype(object)
            else:
                result = DatetimeIndex(data, copy=copy, name=name,
                                       dtype=dtype, **kwargs)
                return result

        elif (is_timedelta64_dtype(data) or
              (dtype is not None and is_timedelta64_dtype(dtype))):
            from pandas import TimedeltaIndex
            if dtype is not None and is_dtype_equal(_o_dtype, dtype):
                # Note we can pass copy=False because the .astype below
                #  will always make a copy
                result = TimedeltaIndex(data, copy=False, name=name, **kwargs)
                return result.astype(object)
            else:
                result = TimedeltaIndex(data, copy=copy, name=name,
                                        dtype=dtype, **kwargs)
                return result

        elif is_period_dtype(data) and not is_object_dtype(dtype):
            from pandas import PeriodIndex
            result = PeriodIndex(data, copy=copy, name=name, **kwargs)
            return result

        # extension dtype
        elif is_extension_array_dtype(data) or is_extension_array_dtype(dtype):
            data = np.asarray(data)
            if not (dtype is None or is_object_dtype(dtype)):

                # coerce to the provided dtype
                data = dtype.construct_array_type()._from_sequence(
                    data, dtype=dtype, copy=False)

            # coerce to the object dtype
            data = data.astype(object)
            return Index(data, dtype=object, copy=copy, name=name,
                         **kwargs)

        # index-like
        elif isinstance(data, (np.ndarray, Index, ABCSeries)):
            if dtype is not None:
                try:

                    # we need to avoid having numpy coerce
                    # things that look like ints/floats to ints unless
                    # they are actually ints, e.g. '0' and 0.0
                    # should not be coerced
                    # GH 11836
                    if is_integer_dtype(dtype):
                        inferred = lib.infer_dtype(data, skipna=False)
                        if inferred == 'integer':
                            data = maybe_cast_to_integer_array(data, dtype,
                                                               copy=copy)
                        elif inferred in ['floating', 'mixed-integer-float']:
                            if isna(data).any():
                                raise ValueError('cannot convert float '
                                                 'NaN to integer')

                            if inferred == "mixed-integer-float":
                                data = maybe_cast_to_integer_array(data, dtype)

                            # If we are actually all equal to integers,
                            # then coerce to integer.
                            try:
                                return cls._try_convert_to_int_index(
                                    data, copy, name, dtype)
                            except ValueError:
                                pass

                            # Return an actual float index.
                            from .numeric import Float64Index
                            return Float64Index(data, copy=copy, dtype=dtype,
                                                name=name)

                        elif inferred == 'string':
                            pass
                        else:
                            data = data.astype(dtype)
                    elif is_float_dtype(dtype):
                        inferred = lib.infer_dtype(data, skipna=False)
                        if inferred == 'string':
                            pass
                        else:
                            data = data.astype(dtype)
                    else:
                        data = np.array(data, dtype=dtype, copy=copy)

                except (TypeError, ValueError) as e:
                    msg = str(e)
                    if ("cannot convert float" in msg or
                            "Trying to coerce float values to integer" in msg):
                        raise

            # maybe coerce to a sub-class
            from pandas.core.indexes.period import (
                PeriodIndex, IncompatibleFrequency)

            if is_signed_integer_dtype(data.dtype):
                from .numeric import Int64Index
                return Int64Index(data, copy=copy, dtype=dtype, name=name)
            elif is_unsigned_integer_dtype(data.dtype):
                from .numeric import UInt64Index
                return UInt64Index(data, copy=copy, dtype=dtype, name=name)
            elif is_float_dtype(data.dtype):
                from .numeric import Float64Index
                return Float64Index(data, copy=copy, dtype=dtype, name=name)
            elif issubclass(data.dtype.type, np.bool) or is_bool_dtype(data):
                subarr = data.astype('object')
            else:
                subarr = com.asarray_tuplesafe(data, dtype=object)

            # asarray_tuplesafe does not always copy underlying data,
            # so need to make sure that this happens
            if copy:
                subarr = subarr.copy()

            if dtype is None:
                inferred = lib.infer_dtype(subarr, skipna=False)
                if inferred == 'integer':
                    try:
                        return cls._try_convert_to_int_index(
                            subarr, copy, name, dtype)
                    except ValueError:
                        pass

                    return Index(subarr, copy=copy,
                                 dtype=object, name=name)
                elif inferred in ['floating', 'mixed-integer-float']:
                    from .numeric import Float64Index
                    return Float64Index(subarr, copy=copy, name=name)
                elif inferred == 'interval':
                    from .interval import IntervalIndex
                    return IntervalIndex(subarr, name=name, copy=copy)
                elif inferred == 'boolean':
                    # don't support boolean explicitly ATM
                    pass
                elif inferred != 'string':
                    if inferred.startswith('datetime'):
                        if (lib.is_datetime_with_singletz_array(subarr) or
                                'tz' in kwargs):
                            # only when subarr has the same tz
                            from pandas import DatetimeIndex
                            try:
                                return DatetimeIndex(subarr, copy=copy,
                                                     name=name, **kwargs)
                            except tslibs.OutOfBoundsDatetime:
                                pass

                    elif inferred.startswith('timedelta'):
                        from pandas import TimedeltaIndex
                        return TimedeltaIndex(subarr, copy=copy, name=name,
                                              **kwargs)
                    elif inferred == 'period':
                        try:
                            return PeriodIndex(subarr, name=name, **kwargs)
                        except IncompatibleFrequency:
                            pass
            return cls._simple_new(subarr, name)

        elif hasattr(data, '__array__'):
            return Index(np.asarray(data), dtype=dtype, copy=copy, name=name,
                         **kwargs)
        elif data is None or is_scalar(data):
            cls._scalar_data_error(data)
        else:
            if tupleize_cols and is_list_like(data):
                # GH21470: convert iterable to list before determining if empty
                if is_iterator(data):
                    data = list(data)

                if data and all(isinstance(e, tuple) for e in data):
                    # we must be all tuples, otherwise don't construct
                    # 10697
                    from .multi import MultiIndex
                    return MultiIndex.from_tuples(
                        data, names=name or kwargs.get('names'))
            # other iterable of some kind
            subarr = com.asarray_tuplesafe(data, dtype=object)
            return Index(subarr, dtype=dtype, copy=copy, name=name, **kwargs)

    """
    NOTE for new Index creation:

    - _simple_new: It returns new Index with the same type as the caller.
      All metadata (such as name) must be provided by caller's responsibility.
      Using _shallow_copy is recommended because it fills these metadata
      otherwise specified.

    - _shallow_copy: It returns new Index with the same type (using
      _simple_new), but fills caller's metadata otherwise specified. Passed
      kwargs will overwrite corresponding metadata.

    - _shallow_copy_with_infer: It returns new Index inferring its type
      from passed values. It fills caller's metadata otherwise specified as the
      same as _shallow_copy.

    See each method's docstring.
    """

    @classmethod
    def _simple_new(cls, values, name=None, dtype=None, **kwargs):
        """
        We require that we have a dtype compat for the values. If we are passed
        a non-dtype compat, then coerce using the constructor.

        Must be careful not to recurse.
        """
        if not hasattr(values, 'dtype'):
            if (values is None or not len(values)) and dtype is not None:
                values = np.empty(0, dtype=dtype)
            else:
                values = np.array(values, copy=False)
                if is_object_dtype(values):
                    values = cls(values, name=name, dtype=dtype,
                                 **kwargs)._ndarray_values

        if isinstance(values, (ABCSeries, ABCIndexClass)):
            # Index._data must always be an ndarray.
            # This is no-copy for when _values is an ndarray,
            # which should be always at this point.
            values = np.asarray(values._values)

        result = object.__new__(cls)
        result._data = values
        # _index_data is a (temporary?) fix to ensure that the direct data
        # manipulation we do in `_libs/reduction.pyx` continues to work.
        # We need access to the actual ndarray, since we're messing with
        # data buffers and strides. We don't re-use `_ndarray_values`, since
        # we actually set this value too.
        result._index_data = values
        result.name = name
        for k, v in compat.iteritems(kwargs):
            setattr(result, k, v)
        return result._reset_identity()

    @cache_readonly
    def _constructor(self):
        return type(self)

    # --------------------------------------------------------------------
    # Index Internals Methods

    def _get_attributes_dict(self):
        """
        Return an attributes dict for my class.
        """
        return {k: getattr(self, k, None) for k in self._attributes}

    _index_shared_docs['_shallow_copy'] = """
        Create a new Index with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        kwargs : updates the default attributes for this Index
        """

    @Appender(_index_shared_docs['_shallow_copy'])
    def _shallow_copy(self, values=None, **kwargs):
        if values is None:
            values = self.values
        attributes = self._get_attributes_dict()
        attributes.update(kwargs)
        if not len(values) and 'dtype' not in kwargs:
            attributes['dtype'] = self.dtype

        # _simple_new expects an the type of self._data
        values = getattr(values, '_values', values)
        if isinstance(values, ABCDatetimeArray):
            # `self.values` returns `self` for tz-aware, so we need to unwrap
            #  more specifically
            values = values.asi8

        return self._simple_new(values, **attributes)

    def _shallow_copy_with_infer(self, values, **kwargs):
        """
        Create a new Index inferring the class with passed value, don't copy
        the data, use the same object attributes with passed in attributes
        taking precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        kwargs : updates the default attributes for this Index
        """
        attributes = self._get_attributes_dict()
        attributes.update(kwargs)
        attributes['copy'] = False
        if not len(values) and 'dtype' not in kwargs:
            attributes['dtype'] = self.dtype
        if self._infer_as_myclass:
            try:
                return self._constructor(values, **attributes)
            except (TypeError, ValueError):
                pass
        return Index(values, **attributes)

    def _update_inplace(self, result, **kwargs):
        # guard when called from IndexOpsMixin
        raise TypeError("Index can't be updated inplace")

    def is_(self, other):
        """
        More flexible, faster check like ``is`` but that works through views.

        Note: this is *not* the same as ``Index.identical()``, which checks
        that metadata is also the same.

        Parameters
        ----------
        other : object
            other object to compare against.

        Returns
        -------
        True if both have same underlying data, False otherwise : bool
        """
        # use something other than None to be clearer
        return self._id is getattr(
            other, '_id', Ellipsis) and self._id is not None

    def _reset_identity(self):
        """
        Initializes or resets ``_id`` attribute with new object.
        """
        self._id = _Identity()
        return self

    def _cleanup(self):
        self._engine.clear_mapping()

    @cache_readonly
    def _engine(self):
        # property, for now, slow to look up
        return self._engine_type(lambda: self._ndarray_values, len(self))

    # --------------------------------------------------------------------
    # Array-Like Methods

    # ndarray compat
    def __len__(self):
        """
        Return the length of the Index.
        """
        return len(self._data)

    def __array__(self, dtype=None):
        """
        The array interface, return my values.
        """
        return np.asarray(self._data, dtype=dtype)

    def __array_wrap__(self, result, context=None):
        """
        Gets called after a ufunc.
        """
        result = lib.item_from_zerodim(result)
        if is_bool_dtype(result) or lib.is_scalar(result):
            return result

        attrs = self._get_attributes_dict()
        attrs = self._maybe_update_attributes(attrs)
        return Index(result, **attrs)

    @cache_readonly
    def dtype(self):
        """
        Return the dtype object of the underlying data.
        """
        return self._data.dtype

    @cache_readonly
    def dtype_str(self):
        """
        Return the dtype str of the underlying data.
        """
        return str(self.dtype)

    def ravel(self, order='C'):
        """
        Return an ndarray of the flattened values of the underlying data.

        See Also
        --------
        numpy.ndarray.ravel
        """
        return self._ndarray_values.ravel(order=order)

    def view(self, cls=None):

        # we need to see if we are subclassing an
        # index type here
        if cls is not None and not hasattr(cls, '_typ'):
            result = self._data.view(cls)
        else:
            result = self._shallow_copy()
        if isinstance(result, Index):
            result._id = self._id
        return result

    _index_shared_docs['astype'] = """
        Create an Index with values cast to dtypes. The class of a new Index
        is determined by dtype. When conversion is impossible, a ValueError
        exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

            .. versionadded:: 0.19.0
        """

    @Appender(_index_shared_docs['astype'])
    def astype(self, dtype, copy=True):
        if is_dtype_equal(self.dtype, dtype):
            return self.copy() if copy else self

        elif is_categorical_dtype(dtype):
            from .category import CategoricalIndex
            return CategoricalIndex(self.values, name=self.name, dtype=dtype,
                                    copy=copy)
        elif is_datetime64tz_dtype(dtype):
            # TODO(GH-24559): Remove this block, use the following elif.
            # avoid FutureWarning from DatetimeIndex constructor.
            from pandas import DatetimeIndex
            tz = pandas_dtype(dtype).tz
            return (DatetimeIndex(np.asarray(self))
                    .tz_localize("UTC").tz_convert(tz))

        elif is_extension_array_dtype(dtype):
            return Index(np.asarray(self), dtype=dtype, copy=copy)

        try:
            if is_datetime64tz_dtype(dtype):
                from pandas import DatetimeIndex
                return DatetimeIndex(self.values, name=self.name, dtype=dtype,
                                     copy=copy)
            return Index(self.values.astype(dtype, copy=copy), name=self.name,
                         dtype=dtype)
        except (TypeError, ValueError):
            msg = 'Cannot cast {name} to dtype {dtype}'
            raise TypeError(msg.format(name=type(self).__name__, dtype=dtype))

    _index_shared_docs['take'] = """
        Return a new %(klass)s of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : list
            Indices to be taken
        axis : int, optional
            The axis over which to select values, always 0.
        allow_fill : bool, default True
        fill_value : bool, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 is regarded as NA. If Index doesn't hold NA, raise ValueError

        See Also
        --------
        numpy.ndarray.take
        """

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self, indices, axis=0, allow_fill=True,
             fill_value=None, **kwargs):
        if kwargs:
            nv.validate_take(tuple(), kwargs)
        indices = ensure_platform_int(indices)
        if self._can_hold_na:
            taken = self._assert_take_fillable(self.values, indices,
                                               allow_fill=allow_fill,
                                               fill_value=fill_value,
                                               na_value=self._na_value)
        else:
            if allow_fill and fill_value is not None:
                msg = 'Unable to fill values because {0} cannot contain NA'
                raise ValueError(msg.format(self.__class__.__name__))
            taken = self.values.take(indices)
        return self._shallow_copy(taken)

    def _assert_take_fillable(self, values, indices, allow_fill=True,
                              fill_value=None, na_value=np.nan):
        """
        Internal method to handle NA filling of take.
        """
        indices = ensure_platform_int(indices)

        # only fill if we are passing a non-None fill_value
        if allow_fill and fill_value is not None:
            if (indices < -1).any():
                msg = ('When allow_fill=True and fill_value is not None, '
                       'all indices must be >= -1')
                raise ValueError(msg)
            taken = algos.take(values,
                               indices,
                               allow_fill=allow_fill,
                               fill_value=na_value)
        else:
            taken = values.take(indices)
        return taken

    _index_shared_docs['repeat'] = """
        Repeat elements of a %(klass)s.

        Returns a new %(klass)s where each element of the current %(klass)s
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            %(klass)s.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        repeated_index : %(klass)s
            Newly created %(klass)s with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.repeat(2)
        Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')
        >>> idx.repeat([1, 2, 3])
        Index(['a', 'b', 'b', 'c', 'c', 'c'], dtype='object')
        """

    @Appender(_index_shared_docs['repeat'] % _index_doc_kwargs)
    def repeat(self, repeats, axis=None):
        nv.validate_repeat(tuple(), dict(axis=axis))
        return self._shallow_copy(self._values.repeat(repeats))

    # --------------------------------------------------------------------
    # Copying Methods

    _index_shared_docs['copy'] = """
        Make a copy of this object.  Name and dtype sets those attributes on
        the new object.

        Parameters
        ----------
        name : string, optional
        deep : boolean, default False
        dtype : numpy dtype or pandas type

        Returns
        -------
        copy : Index

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        """

    @Appender(_index_shared_docs['copy'])
    def copy(self, name=None, deep=False, dtype=None, **kwargs):
        if deep:
            new_index = self._shallow_copy(self._data.copy())
        else:
            new_index = self._shallow_copy()

        names = kwargs.get('names')
        names = self._validate_names(name=name, names=names, deep=deep)
        new_index = new_index.set_names(names)

        if dtype:
            new_index = new_index.astype(dtype)
        return new_index

    def __copy__(self, **kwargs):
        return self.copy(**kwargs)

    def __deepcopy__(self, memo=None):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        if memo is None:
            memo = {}
        return self.copy(deep=True)

    # --------------------------------------------------------------------
    # Rendering Methods

    def __unicode__(self):
        """
        Return a string representation for this object.

        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """
        klass = self.__class__.__name__
        data = self._format_data()
        attrs = self._format_attrs()
        space = self._format_space()

        prepr = (u(",%s") %
                 space).join(u("%s=%s") % (k, v) for k, v in attrs)

        # no data provided, just attributes
        if data is None:
            data = ''

        res = u("%s(%s%s)") % (klass, data, prepr)

        return res

    def _format_space(self):

        # using space here controls if the attributes
        # are line separated or not (the default)

        # max_seq_items = get_option('display.max_seq_items')
        # if len(self) > max_seq_items:
        #    space = "\n%s" % (' ' * (len(klass) + 1))
        return " "

    @property
    def _formatter_func(self):
        """
        Return the formatter function.
        """
        return default_pprint

    def _format_data(self, name=None):
        """
        Return the formatted data as a unicode string.
        """

        # do we want to justify (only do so for non-objects)
        is_justify = not (self.inferred_type in ('string', 'unicode') or
                          (self.inferred_type == 'categorical' and
                           is_object_dtype(self.categories)))

        return format_object_summary(self, self._formatter_func,
                                     is_justify=is_justify, name=name)

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        return format_object_attrs(self)

    def _mpl_repr(self):
        # how to represent ourselves to matplotlib
        return self.values

    def format(self, name=False, formatter=None, **kwargs):
        """
        Render a string representation of the Index.
        """
        header = []
        if name:
            header.append(pprint_thing(self.name,
                                       escape_chars=('\t', '\r', '\n')) if
                          self.name is not None else '')

        if formatter is not None:
            return header + list(self.map(formatter))

        return self._format_with_header(header, **kwargs)

    def _format_with_header(self, header, na_rep='NaN', **kwargs):
        values = self.values

        from pandas.io.formats.format import format_array

        if is_categorical_dtype(values.dtype):
            values = np.array(values)

        elif is_object_dtype(values.dtype):
            values = lib.maybe_convert_objects(values, safe=1)

        if is_object_dtype(values.dtype):
            result = [pprint_thing(x, escape_chars=('\t', '\r', '\n'))
                      for x in values]

            # could have nans
            mask = isna(values)
            if mask.any():
                result = np.array(result)
                result[mask] = na_rep
                result = result.tolist()

        else:
            result = _trim_front(format_array(values, None, justify='left'))
        return header + result

    def to_native_types(self, slicer=None, **kwargs):
        """
        Format specified values of `self` and return them.

        Parameters
        ----------
        slicer : int, array-like
            An indexer into `self` that specifies which values
            are used in the formatting process.
        kwargs : dict
            Options for specifying how the values should be formatted.
            These options include the following:

            1) na_rep : str
                The value that serves as a placeholder for NULL values
            2) quoting : bool or None
                Whether or not there are quoted values in `self`
            3) date_format : str
                The format used to represent date-like values
        """

        values = self
        if slicer is not None:
            values = values[slicer]
        return values._format_native_types(**kwargs)

    def _format_native_types(self, na_rep='', quoting=None, **kwargs):
        """
        Actually format specific types of the index.
        """
        mask = isna(self)
        if not self.is_object() and not quoting:
            values = np.asarray(self).astype(str)
        else:
            values = np.array(self, dtype=object, copy=True)

        values[mask] = na_rep
        return values

    def _summary(self, name=None):
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
        if len(self) > 0:
            head = self[0]
            if (hasattr(head, 'format') and
                    not isinstance(head, compat.string_types)):
                head = head.format()
            tail = self[-1]
            if (hasattr(tail, 'format') and
                    not isinstance(tail, compat.string_types)):
                tail = tail.format()
            index_summary = ', %s to %s' % (pprint_thing(head),
                                            pprint_thing(tail))
        else:
            index_summary = ''

        if name is None:
            name = type(self).__name__
        return '%s: %s entries%s' % (name, len(self), index_summary)

    def summary(self, name=None):
        """
        Return a summarized representation.

        .. deprecated:: 0.23.0
        """
        warnings.warn("'summary' is deprecated and will be removed in a "
                      "future version.", FutureWarning, stacklevel=2)
        return self._summary(name)

    # --------------------------------------------------------------------
    # Conversion Methods

    def to_flat_index(self):
        """
        Identity method.

        .. versionadded:: 0.24.0

        This is implemented for compatability with subclass implementations
        when chaining.

        Returns
        -------
        pd.Index
            Caller.

        See Also
        --------
        MultiIndex.to_flat_index : Subclass implementation.
        """
        return self

    def to_series(self, index=None, name=None):
        """
        Create a Series with both index and values equal to the index keys
        useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            index of resulting Series. If None, defaults to original index
        name : string, optional
            name of resulting Series. If None, defaults to name of original
            index

        Returns
        -------
        Series : dtype will be based on the type of the Index values.
        """

        from pandas import Series

        if index is None:
            index = self._shallow_copy()
        if name is None:
            name = self.name

        return Series(self.values.copy(), index=index, name=name)

    def to_frame(self, index=True, name=None):
        """
        Create a DataFrame with a column containing the Index.

        .. versionadded:: 0.24.0

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index.

        name : object, default None
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """

        from pandas import DataFrame
        if name is None:
            name = self.name or 0
        result = DataFrame({name: self.values.copy()})

        if index:
            result.index = self
        return result

    # --------------------------------------------------------------------
    # Name-Centric Methods

    def _validate_names(self, name=None, names=None, deep=False):
        """
        Handles the quirks of having a singular 'name' parameter for general
        Index and plural 'names' parameter for MultiIndex.
        """
        from copy import deepcopy
        if names is not None and name is not None:
            raise TypeError("Can only provide one of `names` and `name`")
        elif names is None and name is None:
            return deepcopy(self.names) if deep else self.names
        elif names is not None:
            if not is_list_like(names):
                raise TypeError("Must pass list-like as `names`.")
            return names
        else:
            if not is_list_like(name):
                return [name]
            return name

    def _get_names(self):
        return FrozenList((self.name, ))

    def _set_names(self, values, level=None):
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None

        Raises
        ------
        TypeError if each name is not hashable.
        """
        if not is_list_like(values):
            raise ValueError('Names must be a list-like')
        if len(values) != 1:
            raise ValueError('Length of new names must be 1, got %d' %
                             len(values))

        # GH 20527
        # All items in 'name' need to be hashable:
        for name in values:
            if not is_hashable(name):
                raise TypeError('{}.name must be a hashable type'
                                .format(self.__class__.__name__))
        self.name = values[0]

    names = property(fset=_set_names, fget=_get_names)

    def set_names(self, names, level=None, inplace=False):
        """
        Set Index or MultiIndex name.

        Able to set new names partially and by level.

        Parameters
        ----------
        names : label or list of label
            Name(s) to set.
        level : int, label or list of int or label, optional
            If the index is a MultiIndex, level(s) to set (None for all
            levels). Otherwise level must be None.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index
            The same type as the caller or None if inplace is True.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]])
        >>> idx
        MultiIndex(levels=[['cobra', 'python'], [2018, 2019]],
                   codes=[[1, 1, 0, 0], [0, 1, 0, 1]])
        >>> idx.set_names(['kind', 'year'], inplace=True)
        >>> idx
        MultiIndex(levels=[['cobra', 'python'], [2018, 2019]],
                   codes=[[1, 1, 0, 0], [0, 1, 0, 1]],
                   names=['kind', 'year'])
        >>> idx.set_names('species', level=0)
        MultiIndex(levels=[['cobra', 'python'], [2018, 2019]],
                   codes=[[1, 1, 0, 0], [0, 1, 0, 1]],
                   names=['species', 'year'])
        """

        if level is not None and not isinstance(self, ABCMultiIndex):
            raise ValueError('Level must be None for non-MultiIndex')

        if level is not None and not is_list_like(level) and is_list_like(
                names):
            msg = "Names must be a string when a single level is provided."
            raise TypeError(msg)

        if not is_list_like(names) and level is None and self.nlevels > 1:
            raise TypeError("Must pass list-like as `names`.")

        if not is_list_like(names):
            names = [names]
        if level is not None and not is_list_like(level):
            level = [level]

        if inplace:
            idx = self
        else:
            idx = self._shallow_copy()
        idx._set_names(names, level=level)
        if not inplace:
            return idx

    def rename(self, name, inplace=False):
        """
        Alter Index or MultiIndex name.

        Able to set new names without level. Defaults to returning new index.
        Length of names must match number of levels in MultiIndex.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : boolean, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index
            The same type as the caller or None if inplace is True.

        See Also
        --------
        Index.set_names : Able to set new names partially and by level.

        Examples
        --------
        >>> idx = pd.Index(['A', 'C', 'A', 'B'], name='score')
        >>> idx.rename('grade')
        Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]],
        ...                                   names=['kind', 'year'])
        >>> idx
        MultiIndex(levels=[['cobra', 'python'], [2018, 2019]],
                   codes=[[1, 1, 0, 0], [0, 1, 0, 1]],
                   names=['kind', 'year'])
        >>> idx.rename(['species', 'year'])
        MultiIndex(levels=[['cobra', 'python'], [2018, 2019]],
                   codes=[[1, 1, 0, 0], [0, 1, 0, 1]],
                   names=['species', 'year'])
        >>> idx.rename('species')
        Traceback (most recent call last):
        TypeError: Must pass list-like as `names`.
        """
        return self.set_names([name], inplace=inplace)

    # --------------------------------------------------------------------
    # Level-Centric Methods

    @property
    def nlevels(self):
        return 1

    def _sort_levels_monotonic(self):
        """
        Compat with MultiIndex.
        """
        return self

    def _validate_index_level(self, level):
        """
        Validate index level.

        For single-level Index getting level number is a no-op, but some
        verification must be done like in MultiIndex.

        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError("Too many levels: Index has only 1 level,"
                                 " %d is not a valid level number" % (level, ))
            elif level > 0:
                raise IndexError("Too many levels:"
                                 " Index has only 1 level, not %d" %
                                 (level + 1))
        elif level != self.name:
            raise KeyError('Level %s must be same as name (%s)' %
                           (level, self.name))

    def _get_level_number(self, level):
        self._validate_index_level(level)
        return 0

    def sortlevel(self, level=None, ascending=True, sort_remaining=None):
        """
        For internal compatibility with with the Index API.

        Sort the Index. This is for compat with MultiIndex

        Parameters
        ----------
        ascending : boolean, default True
            False to sort in descending order

        level, sort_remaining are compat parameters

        Returns
        -------
        sorted_index : Index
        """
        return self.sort_values(return_indexer=True, ascending=ascending)

    def _get_level_values(self, level):
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatability.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        values : Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        MultiIndex.get_level_values : Get values for a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------

        >>> idx = pd.Index(list('abc'))
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        Get level values by supplying `level` as integer:

        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object')
        """
        self._validate_index_level(level)
        return self

    get_level_values = _get_level_values

    def droplevel(self, level=0):
        """
        Return index with requested level(s) removed.

        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex.

        .. versionadded:: 0.23.1 (support for non-MultiIndex)

        Parameters
        ----------
        level : int, str, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        index : Index or MultiIndex
        """
        if not isinstance(level, (tuple, list)):
            level = [level]

        levnums = sorted(self._get_level_number(lev) for lev in level)[::-1]

        if len(level) == 0:
            return self
        if len(level) >= self.nlevels:
            raise ValueError("Cannot remove {} levels from an index with {} "
                             "levels: at least one level must be "
                             "left.".format(len(level), self.nlevels))
        # The two checks above guarantee that here self is a MultiIndex

        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)

        for i in levnums:
            new_levels.pop(i)
            new_codes.pop(i)
            new_names.pop(i)

        if len(new_levels) == 1:

            # set nan if needed
            mask = new_codes[0] == -1
            result = new_levels[0].take(new_codes[0])
            if mask.any():
                result = result.putmask(mask, np.nan)

            result.name = new_names[0]
            return result
        else:
            from .multi import MultiIndex
            return MultiIndex(levels=new_levels, codes=new_codes,
                              names=new_names, verify_integrity=False)

    _index_shared_docs['_get_grouper_for_level'] = """
        Get index grouper corresponding to an index level

        Parameters
        ----------
        mapper: Group mapping function or None
            Function mapping index values to groups
        level : int or None
            Index level

        Returns
        -------
        grouper : Index
            Index of values to group on
        labels : ndarray of int or None
            Array of locations in level_index
        uniques : Index or None
            Index of unique values for level
        """

    @Appender(_index_shared_docs['_get_grouper_for_level'])
    def _get_grouper_for_level(self, mapper, level=None):
        assert level is None or level == 0
        if mapper is None:
            grouper = self
        else:
            grouper = self.map(mapper)

        return grouper, None, None

    # --------------------------------------------------------------------
    # Introspection Methods

    @property
    def is_monotonic(self):
        """
        Alias for is_monotonic_increasing.
        """
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing (only equal or
        increasing) values.

        Examples
        --------
        >>> Index([1, 2, 3]).is_monotonic_increasing
        True
        >>> Index([1, 2, 2]).is_monotonic_increasing
        True
        >>> Index([1, 3, 2]).is_monotonic_increasing
        False
        """
        return self._engine.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing (only equal or
        decreasing) values.

        Examples
        --------
        >>> Index([3, 2, 1]).is_monotonic_decreasing
        True
        >>> Index([3, 2, 2]).is_monotonic_decreasing
        True
        >>> Index([3, 1, 2]).is_monotonic_decreasing
        False
        """
        return self._engine.is_monotonic_decreasing

    @property
    def _is_strictly_monotonic_increasing(self):
        """
        Return if the index is strictly monotonic increasing
        (only increasing) values.

        Examples
        --------
        >>> Index([1, 2, 3])._is_strictly_monotonic_increasing
        True
        >>> Index([1, 2, 2])._is_strictly_monotonic_increasing
        False
        >>> Index([1, 3, 2])._is_strictly_monotonic_increasing
        False
        """
        return self.is_unique and self.is_monotonic_increasing

    @property
    def _is_strictly_monotonic_decreasing(self):
        """
        Return if the index is strictly monotonic decreasing
        (only decreasing) values.

        Examples
        --------
        >>> Index([3, 2, 1])._is_strictly_monotonic_decreasing
        True
        >>> Index([3, 2, 2])._is_strictly_monotonic_decreasing
        False
        >>> Index([3, 1, 2])._is_strictly_monotonic_decreasing
        False
        """
        return self.is_unique and self.is_monotonic_decreasing

    def is_lexsorted_for_tuple(self, tup):
        return True

    @cache_readonly
    def is_unique(self):
        """
        Return if the index has unique values.
        """
        return self._engine.is_unique

    @property
    def has_duplicates(self):
        return not self.is_unique

    def is_boolean(self):
        return self.inferred_type in ['boolean']

    def is_integer(self):
        return self.inferred_type in ['integer']

    def is_floating(self):
        return self.inferred_type in ['floating', 'mixed-integer-float']

    def is_numeric(self):
        return self.inferred_type in ['integer', 'floating']

    def is_object(self):
        return is_object_dtype(self.dtype)

    def is_categorical(self):
        """
        Check if the Index holds categorical data.

        Returns
        -------
        boolean
            True if the Index is categorical.

        See Also
        --------
        CategoricalIndex : Index for categorical data.

        Examples
        --------
        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_categorical()
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_categorical()
        False

        >>> s = pd.Series(["Peter", "Victor", "Elisabeth", "Mar"])
        >>> s
        0        Peter
        1       Victor
        2    Elisabeth
        3          Mar
        dtype: object
        >>> s.index.is_categorical()
        False
        """
        return self.inferred_type in ['categorical']

    def is_interval(self):
        return self.inferred_type in ['interval']

    def is_mixed(self):
        return self.inferred_type in ['mixed']

    def holds_integer(self):
        return self.inferred_type in ['integer', 'mixed-integer']

    @cache_readonly
    def inferred_type(self):
        """
        Return a string of the type inferred from the values.
        """
        return lib.infer_dtype(self, skipna=False)

    @cache_readonly
    def is_all_dates(self):
        if self._data is None:
            return False
        return is_datetime_array(ensure_object(self.values))

    # --------------------------------------------------------------------
    # Pickle Methods

    def __reduce__(self):
        d = dict(data=self._data)
        d.update(self._get_attributes_dict())
        return _new_Index, (self.__class__, d), None

    def __setstate__(self, state):
        """
        Necessary for making this object picklable.
        """

        if isinstance(state, dict):
            self._data = state.pop('data')
            for k, v in compat.iteritems(state):
                setattr(self, k, v)

        elif isinstance(state, tuple):

            if len(state) == 2:
                nd_state, own_state = state
                data = np.empty(nd_state[1], dtype=nd_state[2])
                np.ndarray.__setstate__(data, nd_state)
                self.name = own_state[0]

            else:  # pragma: no cover
                data = np.empty(state)
                np.ndarray.__setstate__(data, state)

            self._data = data
            self._reset_identity()
        else:
            raise Exception("invalid pickle state")

    _unpickle_compat = __setstate__

    # --------------------------------------------------------------------
    # Null Handling Methods

    _na_value = np.nan
    """The expected NA value to use with this index."""

    @cache_readonly
    def _isnan(self):
        """
        Return if each value is NaN.
        """
        if self._can_hold_na:
            return isna(self)
        else:
            # shouldn't reach to this condition by checking hasnans beforehand
            values = np.empty(len(self), dtype=np.bool_)
            values.fill(False)
            return values

    @cache_readonly
    def _nan_idxs(self):
        if self._can_hold_na:
            w, = self._isnan.nonzero()
            return w
        else:
            return np.array([], dtype=np.int64)

    @cache_readonly
    def hasnans(self):
        """
        Return if I have any nans; enables various perf speedups.
        """
        if self._can_hold_na:
            return bool(self._isnan.any())
        else:
            return False

    def isna(self):
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values. Characters such as
        empty strings `''` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).

        .. versionadded:: 0.20.0

        Returns
        -------
        numpy.ndarray
            A boolean array of whether my values are NA

        See Also
        --------
        pandas.Index.notna : Boolean inverse of isna.
        pandas.Index.dropna : Omit entries with missing values.
        pandas.isna : Top-level isna.
        Series.isna : Detect missing values in Series object.

        Examples
        --------
        Show which entries in a pandas.Index are NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.NaN])
        >>> idx
        Float64Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.isna()
        array([False, False,  True], dtype=bool)

        Empty strings are not considered NA values. None is considered an NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.isna()
        array([False, False, False,  True], dtype=bool)

        For datetimes, `NaT` (Not a Time) is considered as an NA value.

        >>> idx = pd.DatetimeIndex([pd.Timestamp('1940-04-25'),
        ...                         pd.Timestamp(''), None, pd.NaT])
        >>> idx
        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.isna()
        array([False,  True,  True,  True], dtype=bool)
        """
        return self._isnan
    isnull = isna

    def notna(self):
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values
        (unless you set ``pandas.options.mode.use_inf_as_na = True``).
        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``
        values.

        .. versionadded:: 0.20.0

        Returns
        -------
        numpy.ndarray
            Boolean array to indicate which entries are not NA.

        See Also
        --------
        Index.notnull : Alias of notna.
        Index.isna: Inverse of notna.
        pandas.notna : Top-level notna.

        Examples
        --------
        Show which entries in an Index are not NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.NaN])
        >>> idx
        Float64Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False])

        Empty strings are not considered NA values. None is considered a NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.notna()
        array([ True,  True,  True, False])
        """
        return ~self.isna()
    notnull = notna

    _index_shared_docs['fillna'] = """
        Fill NA/NaN values with the specified value

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.
        downcast : dict, default is None
            a dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible)

        Returns
        -------
        filled : Index
        """

    @Appender(_index_shared_docs['fillna'])
    def fillna(self, value=None, downcast=None):
        self._assert_can_do_op(value)
        if self.hasnans:
            result = self.putmask(self._isnan, value)
            if downcast is None:
                # no need to care metadata other than name
                # because it can't have freq if
                return Index(result, name=self.name)
        return self._shallow_copy()

    _index_shared_docs['dropna'] = """
        Return Index without NA/NaN values

        Parameters
        ----------
        how :  {'any', 'all'}, default 'any'
            If the Index is a MultiIndex, drop the value when any or all levels
            are NaN.

        Returns
        -------
        valid : Index
        """

    @Appender(_index_shared_docs['dropna'])
    def dropna(self, how='any'):
        if how not in ('any', 'all'):
            raise ValueError("invalid how option: {0}".format(how))

        if self.hasnans:
            return self._shallow_copy(self.values[~self._isnan])
        return self._shallow_copy()

    # --------------------------------------------------------------------
    # Uniqueness Methods

    _index_shared_docs['index_unique'] = (
        """
        Return unique values in the index. Uniques are returned in order
        of appearance, this does NOT sort.

        Parameters
        ----------
        level : int or str, optional, default None
            Only return values from specified level (for MultiIndex)

            .. versionadded:: 0.23.0

        Returns
        -------
        Index without duplicates

        See Also
        --------
        unique
        Series.unique
        """)

    @Appender(_index_shared_docs['index_unique'] % _index_doc_kwargs)
    def unique(self, level=None):
        if level is not None:
            self._validate_index_level(level)
        result = super(Index, self).unique()
        return self._shallow_copy(result)

    def drop_duplicates(self, keep='first'):
        """
        Return Index with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        Returns
        -------
        deduplicated : Index

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Index.duplicated : Related method on Index, indicating duplicate
            Index values.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

        The `keep` parameter controls  which duplicate values are removed.
        The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> idx.drop_duplicates(keep='first')
        Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')

        The value 'last' keeps the last occurrence for each set of duplicated
        entries.

        >>> idx.drop_duplicates(keep='last')
        Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')

        The value ``False`` discards all sets of duplicated entries.

        >>> idx.drop_duplicates(keep=False)
        Index(['cow', 'beetle', 'hippo'], dtype='object')
        """
        return super(Index, self).drop_duplicates(keep=keep)

    def duplicated(self, keep='first'):
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        pandas.Series.duplicated : Equivalent method on pandas.Series.
        pandas.DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        pandas.Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep='first')
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> idx.duplicated(keep='last')
        array([ True, False,  True, False, False])

        By setting keep on ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
        return super(Index, self).duplicated(keep=keep)

    def get_duplicates(self):
        """
        Extract duplicated index elements.

        .. deprecated:: 0.23.0
            Use idx[idx.duplicated()].unique() instead

        Returns a sorted list of index elements which appear more than once in
        the index.

        Returns
        -------
        array-like
            List of duplicated indexes.

        See Also
        --------
        Index.duplicated : Return boolean array denoting duplicates.
        Index.drop_duplicates : Return Index with duplicates removed.

        Examples
        --------

        Works on different Index of types.

        >>> pd.Index([1, 2, 2, 3, 3, 3, 4]).get_duplicates()  # doctest: +SKIP
        [2, 3]

        Note that for a DatetimeIndex, it does not return a list but a new
        DatetimeIndex:

        >>> dates = pd.to_datetime(['2018-01-01', '2018-01-02', '2018-01-03',
        ...                         '2018-01-03', '2018-01-04', '2018-01-04'],
        ...                        format='%Y-%m-%d')
        >>> pd.Index(dates).get_duplicates()  # doctest: +SKIP
        DatetimeIndex(['2018-01-03', '2018-01-04'],
                      dtype='datetime64[ns]', freq=None)

        Sorts duplicated elements even when indexes are unordered.

        >>> pd.Index([1, 2, 3, 2, 3, 4, 3]).get_duplicates()  # doctest: +SKIP
        [2, 3]

        Return empty array-like structure when all elements are unique.

        >>> pd.Index([1, 2, 3, 4]).get_duplicates()  # doctest: +SKIP
        []
        >>> dates = pd.to_datetime(['2018-01-01', '2018-01-02', '2018-01-03'],
        ...                        format='%Y-%m-%d')
        >>> pd.Index(dates).get_duplicates()  # doctest: +SKIP
        DatetimeIndex([], dtype='datetime64[ns]', freq=None)
        """
        warnings.warn("'get_duplicates' is deprecated and will be removed in "
                      "a future release. You can use "
                      "idx[idx.duplicated()].unique() instead",
                      FutureWarning, stacklevel=2)

        return self[self.duplicated()].unique()

    def _get_unique_index(self, dropna=False):
        """
        Returns an index containing unique values.

        Parameters
        ----------
        dropna : bool
            If True, NaN values are dropped.

        Returns
        -------
        uniques : index
        """
        if self.is_unique and not dropna:
            return self

        values = self.values

        if not self.is_unique:
            values = self.unique()

        if dropna:
            try:
                if self.hasnans:
                    values = values[~isna(values)]
            except NotImplementedError:
                pass

        return self._shallow_copy(values)

    # --------------------------------------------------------------------
    # Arithmetic & Logical Methods

    def __add__(self, other):
        if isinstance(other, (ABCSeries, ABCDataFrame)):
            return NotImplemented
        return Index(np.array(self) + other)

    def __radd__(self, other):
        return Index(other + np.array(self))

    def __iadd__(self, other):
        # alias for __add__
        return self + other

    def __sub__(self, other):
        return Index(np.array(self) - other)

    def __rsub__(self, other):
        return Index(other - np.array(self))

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __nonzero__(self):
        raise ValueError("The truth value of a {0} is ambiguous. "
                         "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
                         .format(self.__class__.__name__))

    __bool__ = __nonzero__

    # --------------------------------------------------------------------
    # Set Operation Methods

    def _get_reconciled_name_object(self, other):
        """
        If the result of a set operation will be self,
        return self, unless the name changes, in which
        case make a shallow copy of self.
        """
        name = get_op_result_name(self, other)
        if self.name != name:
            return self._shallow_copy(name=name)
        return self

    def _validate_sort_keyword(self, sort):
        if sort not in [None, False]:
            raise ValueError("The 'sort' keyword only takes the values of "
                             "None or False; {0} was passed.".format(sort))

    def union(self, other, sort=None):
        """
        Form the union of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.
              3. Some values in `self` or `other` cannot be compared.
                 A RuntimeWarning is issued in this case.

            * False : do not sort the result.

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default value from ``True`` to ``None``
               (without change in behaviour).

        Returns
        -------
        union : Index

        Examples
        --------

        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Int64Index([1, 2, 3, 4, 5, 6], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other = ensure_index(other)

        if len(other) == 0 or self.equals(other):
            return self._get_reconciled_name_object(other)

        if len(self) == 0:
            return other._get_reconciled_name_object(self)

        # TODO: is_dtype_union_equal is a hack around
        # 1. buggy set ops with duplicates (GH #13432)
        # 2. CategoricalIndex lacking setops (GH #10186)
        # Once those are fixed, this workaround can be removed
        if not is_dtype_union_equal(self.dtype, other.dtype):
            this = self.astype('O')
            other = other.astype('O')
            return this.union(other, sort=sort)

        # TODO(EA): setops-refactor, clean all this up
        if is_period_dtype(self) or is_datetime64tz_dtype(self):
            lvals = self._ndarray_values
        else:
            lvals = self._values
        if is_period_dtype(other) or is_datetime64tz_dtype(other):
            rvals = other._ndarray_values
        else:
            rvals = other._values

        if self.is_monotonic and other.is_monotonic:
            try:
                result = self._outer_indexer(lvals, rvals)[0]
            except TypeError:
                # incomparable objects
                result = list(lvals)

                # worth making this faster? a very unusual case
                value_set = set(lvals)
                result.extend([x for x in rvals if x not in value_set])
        else:
            indexer = self.get_indexer(other)
            indexer, = (indexer == -1).nonzero()

            if len(indexer) > 0:
                other_diff = algos.take_nd(rvals, indexer,
                                           allow_fill=False)
                result = _concat._concat_compat((lvals, other_diff))

            else:
                result = lvals

            if sort is None:
                try:
                    result = sorting.safe_sort(result)
                except TypeError as e:
                    warnings.warn("{}, sort order is undefined for "
                                  "incomparable objects".format(e),
                                  RuntimeWarning, stacklevel=3)

        # for subclasses
        return self._wrap_setop_result(other, result)

    def _wrap_setop_result(self, other, result):
        return self._constructor(result, name=get_op_result_name(self, other))

    def intersection(self, other, sort=False):
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default from ``True`` to ``False``, to match
               the behaviour of 0.23.4 and earlier.

        Returns
        -------
        intersection : Index

        Examples
        --------

        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Int64Index([3, 4], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other = ensure_index(other)

        if self.equals(other):
            return self._get_reconciled_name_object(other)

        if not is_dtype_equal(self.dtype, other.dtype):
            this = self.astype('O')
            other = other.astype('O')
            return this.intersection(other, sort=sort)

        # TODO(EA): setops-refactor, clean all this up
        if is_period_dtype(self):
            lvals = self._ndarray_values
        else:
            lvals = self._values
        if is_period_dtype(other):
            rvals = other._ndarray_values
        else:
            rvals = other._values

        if self.is_monotonic and other.is_monotonic:
            try:
                result = self._inner_indexer(lvals, rvals)[0]
                return self._wrap_setop_result(other, result)
            except TypeError:
                pass

        try:
            indexer = Index(rvals).get_indexer(lvals)
            indexer = indexer.take((indexer != -1).nonzero()[0])
        except Exception:
            # duplicates
            indexer = algos.unique1d(
                Index(rvals).get_indexer_non_unique(lvals)[0])
            indexer = indexer[indexer != -1]

        taken = other.take(indexer)

        if sort is None:
            taken = sorting.safe_sort(taken.values)
            if self.name != other.name:
                name = None
            else:
                name = self.name
            return self._shallow_copy(taken, name=name)

        if self.name != other.name:
            taken.name = None

        return taken

    def difference(self, other, sort=None):
        """
        Return a new Index with elements from the index that are not in
        `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default value from ``True`` to ``None``
               (without change in behaviour).

        Returns
        -------
        difference : Index

        Examples
        --------

        >>> idx1 = pd.Index([2, 1, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2)
        Int64Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Int64Index([2, 1], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)

        if self.equals(other):
            # pass an empty np.ndarray with the appropriate dtype
            return self._shallow_copy(self._data[:0])

        other, result_name = self._convert_can_do_setop(other)

        this = self._get_unique_index()

        indexer = this.get_indexer(other)
        indexer = indexer.take((indexer != -1).nonzero()[0])

        label_diff = np.setdiff1d(np.arange(this.size), indexer,
                                  assume_unique=True)
        the_diff = this.values.take(label_diff)
        if sort is None:
            try:
                the_diff = sorting.safe_sort(the_diff)
            except TypeError:
                pass

        return this._shallow_copy(the_diff, name=result_name, freq=None)

    def symmetric_difference(self, other, result_name=None, sort=None):
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : str
        sort : False or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default value from ``True`` to ``None``
               (without change in behaviour).

        Returns
        -------
        symmetric_difference : Index

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([2, 3, 4, 5])
        >>> idx1.symmetric_difference(idx2)
        Int64Index([1, 5], dtype='int64')

        You can also use the ``^`` operator:

        >>> idx1 ^ idx2
        Int64Index([1, 5], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name_update = self._convert_can_do_setop(other)
        if result_name is None:
            result_name = result_name_update

        this = self._get_unique_index()
        other = other._get_unique_index()
        indexer = this.get_indexer(other)

        # {this} minus {other}
        common_indexer = indexer.take((indexer != -1).nonzero()[0])
        left_indexer = np.setdiff1d(np.arange(this.size), common_indexer,
                                    assume_unique=True)
        left_diff = this.values.take(left_indexer)

        # {other} minus {this}
        right_indexer = (indexer == -1).nonzero()[0]
        right_diff = other.values.take(right_indexer)

        the_diff = _concat._concat_compat([left_diff, right_diff])
        if sort is None:
            try:
                the_diff = sorting.safe_sort(the_diff)
            except TypeError:
                pass

        attribs = self._get_attributes_dict()
        attribs['name'] = result_name
        if 'freq' in attribs:
            attribs['freq'] = None
        return self._shallow_copy_with_infer(the_diff, **attribs)

    def _assert_can_do_setop(self, other):
        if not is_list_like(other):
            raise TypeError('Input must be Index or array-like')
        return True

    def _convert_can_do_setop(self, other):
        if not isinstance(other, Index):
            other = Index(other, name=self.name)
            result_name = self.name
        else:
            result_name = get_op_result_name(self, other)
        return other, result_name

    # --------------------------------------------------------------------
    # Indexing Methods

    _index_shared_docs['get_loc'] = """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        tolerance : optional
            Maximum distance from index value for inexact matches. The value of
            the index at the matching location most satisfy the equation
            ``abs(index[loc] - key) <= tolerance``.

            Tolerance may be a scalar
            value, which applies the same tolerance to all values, or
            list-like, which applies variable tolerance per element. List-like
            includes list, tuple, array, Series, and must be the same size as
            the index and its dtype must exactly match the index's type.

            .. versionadded:: 0.21.0 (list-like tolerance)

        Returns
        -------
        loc : int if unique index, slice if monotonic index, else mask

        Examples
        ---------
        >>> unique_index = pd.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1

        >>> monotonic_index = pd.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)

        >>> non_monotonic_index = pd.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True], dtype=bool)
        """

    @Appender(_index_shared_docs['get_loc'])
    def get_loc(self, key, method=None, tolerance=None):
        if method is None:
            if tolerance is not None:
                raise ValueError('tolerance argument only valid if using pad, '
                                 'backfill or nearest lookups')
            try:
                return self._engine.get_loc(key)
            except KeyError:
                return self._engine.get_loc(self._maybe_cast_indexer(key))
        indexer = self.get_indexer([key], method=method, tolerance=tolerance)
        if indexer.ndim > 1 or indexer.size > 1:
            raise TypeError('get_loc requires scalar valued input')
        loc = indexer.item()
        if loc == -1:
            raise KeyError(key)
        return loc

    _index_shared_docs['get_indexer'] = """
        Compute indexer and mask for new index given the current index. The
        indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : %(target_klass)s
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations most
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

            .. versionadded:: 0.21.0 (list-like tolerance)

        Returns
        -------
        indexer : ndarray of int
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.

        Examples
        --------
        >>> index = pd.Index(['c', 'a', 'b'])
        >>> index.get_indexer(['a', 'b', 'x'])
        array([ 1,  2, -1])

        Notice that the return value is an array of locations in ``index``
        and ``x`` is marked by -1, as it is not in ``index``.
        """

    @Appender(_index_shared_docs['get_indexer'] % _index_doc_kwargs)
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        method = missing.clean_reindex_fill_method(method)
        target = ensure_index(target)
        if tolerance is not None:
            tolerance = self._convert_tolerance(tolerance, target)

        # Treat boolean labels passed to a numeric index as not found. Without
        # this fix False and True would be treated as 0 and 1 respectively.
        # (GH #16877)
        if target.is_boolean() and self.is_numeric():
            return ensure_platform_int(np.repeat(-1, target.size))

        pself, ptarget = self._maybe_promote(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer(ptarget, method=method, limit=limit,
                                     tolerance=tolerance)

        if not is_dtype_equal(self.dtype, target.dtype):
            this = self.astype(object)
            target = target.astype(object)
            return this.get_indexer(target, method=method, limit=limit,
                                    tolerance=tolerance)

        if not self.is_unique:
            raise InvalidIndexError('Reindexing only valid with uniquely'
                                    ' valued Index objects')

        if method == 'pad' or method == 'backfill':
            indexer = self._get_fill_indexer(target, method, limit, tolerance)
        elif method == 'nearest':
            indexer = self._get_nearest_indexer(target, limit, tolerance)
        else:
            if tolerance is not None:
                raise ValueError('tolerance argument only valid if doing pad, '
                                 'backfill or nearest reindexing')
            if limit is not None:
                raise ValueError('limit argument only valid if doing pad, '
                                 'backfill or nearest reindexing')

            indexer = self._engine.get_indexer(target._ndarray_values)

        return ensure_platform_int(indexer)

    def _convert_tolerance(self, tolerance, target):
        # override this method on subclasses
        tolerance = np.asarray(tolerance)
        if target.size != tolerance.size and tolerance.size > 1:
            raise ValueError('list-like tolerance size must match '
                             'target index size')
        return tolerance

    def _get_fill_indexer(self, target, method, limit=None, tolerance=None):
        if self.is_monotonic_increasing and target.is_monotonic_increasing:
            method = (self._engine.get_pad_indexer if method == 'pad' else
                      self._engine.get_backfill_indexer)
            indexer = method(target._ndarray_values, limit)
        else:
            indexer = self._get_fill_indexer_searchsorted(target, method,
                                                          limit)
        if tolerance is not None:
            indexer = self._filter_indexer_tolerance(target._ndarray_values,
                                                     indexer,
                                                     tolerance)
        return indexer

    def _get_fill_indexer_searchsorted(self, target, method, limit=None):
        """
        Fallback pad/backfill get_indexer that works for monotonic decreasing
        indexes and non-monotonic targets.
        """
        if limit is not None:
            raise ValueError('limit argument for %r method only well-defined '
                             'if index and target are monotonic' % method)

        side = 'left' if method == 'pad' else 'right'

        # find exact matches first (this simplifies the algorithm)
        indexer = self.get_indexer(target)
        nonexact = (indexer == -1)
        indexer[nonexact] = self._searchsorted_monotonic(target[nonexact],
                                                         side)
        if side == 'left':
            # searchsorted returns "indices into a sorted array such that,
            # if the corresponding elements in v were inserted before the
            # indices, the order of a would be preserved".
            # Thus, we need to subtract 1 to find values to the left.
            indexer[nonexact] -= 1
            # This also mapped not found values (values of 0 from
            # np.searchsorted) to -1, which conveniently is also our
            # sentinel for missing values
        else:
            # Mark indices to the right of the largest value as not found
            indexer[indexer == len(self)] = -1
        return indexer

    def _get_nearest_indexer(self, target, limit, tolerance):
        """
        Get the indexer for the nearest index labels; requires an index with
        values that can be subtracted from each other (e.g., not strings or
        tuples).
        """
        left_indexer = self.get_indexer(target, 'pad', limit=limit)
        right_indexer = self.get_indexer(target, 'backfill', limit=limit)

        target = np.asarray(target)
        left_distances = abs(self.values[left_indexer] - target)
        right_distances = abs(self.values[right_indexer] - target)

        op = operator.lt if self.is_monotonic_increasing else operator.le
        indexer = np.where(op(left_distances, right_distances) |
                           (right_indexer == -1), left_indexer, right_indexer)
        if tolerance is not None:
            indexer = self._filter_indexer_tolerance(target, indexer,
                                                     tolerance)
        return indexer

    def _filter_indexer_tolerance(self, target, indexer, tolerance):
        distance = abs(self.values[indexer] - target)
        indexer = np.where(distance <= tolerance, indexer, -1)
        return indexer

    # --------------------------------------------------------------------
    # Indexer Conversion Methods

    _index_shared_docs['_convert_scalar_indexer'] = """
        Convert a scalar indexer.

        Parameters
        ----------
        key : label of the slice bound
        kind : {'ix', 'loc', 'getitem', 'iloc'} or None
    """

    @Appender(_index_shared_docs['_convert_scalar_indexer'])
    def _convert_scalar_indexer(self, key, kind=None):
        assert kind in ['ix', 'loc', 'getitem', 'iloc', None]

        if kind == 'iloc':
            return self._validate_indexer('positional', key, kind)

        if len(self) and not isinstance(self, ABCMultiIndex,):

            # we can raise here if we are definitive that this
            # is positional indexing (eg. .ix on with a float)
            # or label indexing if we are using a type able
            # to be represented in the index

            if kind in ['getitem', 'ix'] and is_float(key):
                if not self.is_floating():
                    return self._invalid_indexer('label', key)

            elif kind in ['loc'] and is_float(key):

                # we want to raise KeyError on string/mixed here
                # technically we *could* raise a TypeError
                # on anything but mixed though
                if self.inferred_type not in ['floating',
                                              'mixed-integer-float',
                                              'string',
                                              'unicode',
                                              'mixed']:
                    return self._invalid_indexer('label', key)

            elif kind in ['loc'] and is_integer(key):
                if not self.holds_integer():
                    return self._invalid_indexer('label', key)

        return key

    _index_shared_docs['_convert_slice_indexer'] = """
        Convert a slice indexer.

        By definition, these are labels unless 'iloc' is passed in.
        Floats are not allowed as the start, step, or stop of the slice.

        Parameters
        ----------
        key : label of the slice bound
        kind : {'ix', 'loc', 'getitem', 'iloc'} or None
    """

    @Appender(_index_shared_docs['_convert_slice_indexer'])
    def _convert_slice_indexer(self, key, kind=None):
        assert kind in ['ix', 'loc', 'getitem', 'iloc', None]

        # if we are not a slice, then we are done
        if not isinstance(key, slice):
            return key

        # validate iloc
        if kind == 'iloc':
            return slice(self._validate_indexer('slice', key.start, kind),
                         self._validate_indexer('slice', key.stop, kind),
                         self._validate_indexer('slice', key.step, kind))

        # potentially cast the bounds to integers
        start, stop, step = key.start, key.stop, key.step

        # figure out if this is a positional indexer
        def is_int(v):
            return v is None or is_integer(v)

        is_null_slicer = start is None and stop is None
        is_index_slice = is_int(start) and is_int(stop)
        is_positional = is_index_slice and not self.is_integer()

        if kind == 'getitem':
            """
            called from the getitem slicers, validate that we are in fact
            integers
            """
            if self.is_integer() or is_index_slice:
                return slice(self._validate_indexer('slice', key.start, kind),
                             self._validate_indexer('slice', key.stop, kind),
                             self._validate_indexer('slice', key.step, kind))

        # convert the slice to an indexer here

        # if we are mixed and have integers
        try:
            if is_positional and self.is_mixed():
                # Validate start & stop
                if start is not None:
                    self.get_loc(start)
                if stop is not None:
                    self.get_loc(stop)
                is_positional = False
        except KeyError:
            if self.inferred_type == 'mixed-integer-float':
                raise

        if is_null_slicer:
            indexer = key
        elif is_positional:
            indexer = key
        else:
            try:
                indexer = self.slice_indexer(start, stop, step, kind=kind)
            except Exception:
                if is_index_slice:
                    if self.is_integer():
                        raise
                    else:
                        indexer = key
                else:
                    raise

        return indexer

    def _convert_listlike_indexer(self, keyarr, kind=None):
        """
        Parameters
        ----------
        keyarr : list-like
            Indexer to convert.

        Returns
        -------
        tuple (indexer, keyarr)
            indexer is an ndarray or None if cannot convert
            keyarr are tuple-safe keys
        """
        if isinstance(keyarr, Index):
            keyarr = self._convert_index_indexer(keyarr)
        else:
            keyarr = self._convert_arr_indexer(keyarr)

        indexer = self._convert_list_indexer(keyarr, kind=kind)
        return indexer, keyarr

    _index_shared_docs['_convert_arr_indexer'] = """
        Convert an array-like indexer to the appropriate dtype.

        Parameters
        ----------
        keyarr : array-like
            Indexer to convert.

        Returns
        -------
        converted_keyarr : array-like
    """

    @Appender(_index_shared_docs['_convert_arr_indexer'])
    def _convert_arr_indexer(self, keyarr):
        keyarr = com.asarray_tuplesafe(keyarr)
        return keyarr

    _index_shared_docs['_convert_index_indexer'] = """
        Convert an Index indexer to the appropriate dtype.

        Parameters
        ----------
        keyarr : Index (or sub-class)
            Indexer to convert.

        Returns
        -------
        converted_keyarr : Index (or sub-class)
    """

    @Appender(_index_shared_docs['_convert_index_indexer'])
    def _convert_index_indexer(self, keyarr):
        return keyarr

    _index_shared_docs['_convert_list_indexer'] = """
        Convert a list-like indexer to the appropriate dtype.

        Parameters
        ----------
        keyarr : Index (or sub-class)
            Indexer to convert.
        kind : iloc, ix, loc, optional

        Returns
        -------
        positional indexer or None
    """

    @Appender(_index_shared_docs['_convert_list_indexer'])
    def _convert_list_indexer(self, keyarr, kind=None):
        if (kind in [None, 'iloc', 'ix'] and
                is_integer_dtype(keyarr) and not self.is_floating() and
                not isinstance(keyarr, ABCPeriodIndex)):

            if self.inferred_type == 'mixed-integer':
                indexer = self.get_indexer(keyarr)
                if (indexer >= 0).all():
                    return indexer
                # missing values are flagged as -1 by get_indexer and negative
                # indices are already converted to positive indices in the
                # above if-statement, so the negative flags are changed to
                # values outside the range of indices so as to trigger an
                # IndexError in maybe_convert_indices
                indexer[indexer < 0] = len(self)
                from pandas.core.indexing import maybe_convert_indices
                return maybe_convert_indices(indexer, len(self))

            elif not self.inferred_type == 'integer':
                keyarr = np.where(keyarr < 0, len(self) + keyarr, keyarr)
                return keyarr

        return None

    def _invalid_indexer(self, form, key):
        """
        Consistent invalid indexer message.
        """
        raise TypeError("cannot do {form} indexing on {klass} with these "
                        "indexers [{key}] of {kind}".format(
                            form=form, klass=type(self), key=key,
                            kind=type(key)))

    # --------------------------------------------------------------------
    # Reindex Methods

    def _can_reindex(self, indexer):
        """
        Check if we are allowing reindexing with this particular indexer.

        Parameters
        ----------
        indexer : an integer indexer

        Raises
        ------
        ValueError if its a duplicate axis
        """

        # trying to reindex on an axis with duplicates
        if not self.is_unique and len(indexer):
            raise ValueError("cannot reindex from a duplicate axis")

    def reindex(self, target, method=None, level=None, limit=None,
                tolerance=None):
        """
        Create index with target's values (move/add/delete values
        as necessary).

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index
        indexer : np.ndarray or None
            Indices of output values in original index

        """
        # GH6552: preserve names when reindexing to non-named target
        # (i.e. neither Index nor Series).
        preserve_names = not hasattr(target, 'name')

        # GH7774: preserve dtype/tz if target is empty and not an Index.
        target = _ensure_has_len(target)  # target may be an iterator

        if not isinstance(target, Index) and len(target) == 0:
            attrs = self._get_attributes_dict()
            attrs.pop('freq', None)  # don't preserve freq
            values = self._data[:0]  # appropriately-dtyped empty array
            target = self._simple_new(values, dtype=self.dtype, **attrs)
        else:
            target = ensure_index(target)

        if level is not None:
            if method is not None:
                raise TypeError('Fill method not supported if level passed')
            _, indexer, _ = self._join_level(target, level, how='right',
                                             return_indexers=True)
        else:
            if self.equals(target):
                indexer = None
            else:

                if self.is_unique:
                    indexer = self.get_indexer(target, method=method,
                                               limit=limit,
                                               tolerance=tolerance)
                else:
                    if method is not None or limit is not None:
                        raise ValueError("cannot reindex a non-unique index "
                                         "with a method or limit")
                    indexer, missing = self.get_indexer_non_unique(target)

        if preserve_names and target.nlevels == 1 and target.name != self.name:
            target = target.copy()
            target.name = self.name

        return target, indexer

    def _reindex_non_unique(self, target):
        """
        Create a new index with target's values (move/add/delete values as
        necessary) use with non-unique Index and a possibly non-unique target.

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index
        indexer : np.ndarray or None
            Indices of output values in original index

        """

        target = ensure_index(target)
        indexer, missing = self.get_indexer_non_unique(target)
        check = indexer != -1
        new_labels = self.take(indexer[check])
        new_indexer = None

        if len(missing):
            length = np.arange(len(indexer))

            missing = ensure_platform_int(missing)
            missing_labels = target.take(missing)
            missing_indexer = ensure_int64(length[~check])
            cur_labels = self.take(indexer[check]).values
            cur_indexer = ensure_int64(length[check])

            new_labels = np.empty(tuple([len(indexer)]), dtype=object)
            new_labels[cur_indexer] = cur_labels
            new_labels[missing_indexer] = missing_labels

            # a unique indexer
            if target.is_unique:

                # see GH5553, make sure we use the right indexer
                new_indexer = np.arange(len(indexer))
                new_indexer[cur_indexer] = np.arange(len(cur_labels))
                new_indexer[missing_indexer] = -1

            # we have a non_unique selector, need to use the original
            # indexer here
            else:

                # need to retake to have the same size as the indexer
                indexer[~check] = -1

                # reset the new indexer to account for the new size
                new_indexer = np.arange(len(self.take(indexer)))
                new_indexer[~check] = -1

        new_index = self._shallow_copy_with_infer(new_labels, freq=None)
        return new_index, indexer, new_indexer

    # --------------------------------------------------------------------
    # Join Methods

    _index_shared_docs['join'] = """
        Compute join_index and indexers to conform data
        structures to the new index.

        Parameters
        ----------
        other : Index
        how : {'left', 'right', 'inner', 'outer'}
        level : int or level name, default None
        return_indexers : boolean, default False
        sort : boolean, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword)

            .. versionadded:: 0.20.0

        Returns
        -------
        join_index, (left_indexer, right_indexer)
        """

    @Appender(_index_shared_docs['join'])
    def join(self, other, how='left', level=None, return_indexers=False,
             sort=False):
        self_is_mi = isinstance(self, ABCMultiIndex)
        other_is_mi = isinstance(other, ABCMultiIndex)

        # try to figure out the join level
        # GH3662
        if level is None and (self_is_mi or other_is_mi):

            # have the same levels/names so a simple join
            if self.names == other.names:
                pass
            else:
                return self._join_multi(other, how=how,
                                        return_indexers=return_indexers)

        # join on the level
        if level is not None and (self_is_mi or other_is_mi):
            return self._join_level(other, level, how=how,
                                    return_indexers=return_indexers)

        other = ensure_index(other)

        if len(other) == 0 and how in ('left', 'outer'):
            join_index = self._shallow_copy()
            if return_indexers:
                rindexer = np.repeat(-1, len(join_index))
                return join_index, None, rindexer
            else:
                return join_index

        if len(self) == 0 and how in ('right', 'outer'):
            join_index = other._shallow_copy()
            if return_indexers:
                lindexer = np.repeat(-1, len(join_index))
                return join_index, lindexer, None
            else:
                return join_index

        if self._join_precedence < other._join_precedence:
            how = {'right': 'left', 'left': 'right'}.get(how, how)
            result = other.join(self, how=how, level=level,
                                return_indexers=return_indexers)
            if return_indexers:
                x, y, z = result
                result = x, z, y
            return result

        if not is_dtype_equal(self.dtype, other.dtype):
            this = self.astype('O')
            other = other.astype('O')
            return this.join(other, how=how, return_indexers=return_indexers)

        _validate_join_method(how)

        if not self.is_unique and not other.is_unique:
            return self._join_non_unique(other, how=how,
                                         return_indexers=return_indexers)
        elif not self.is_unique or not other.is_unique:
            if self.is_monotonic and other.is_monotonic:
                return self._join_monotonic(other, how=how,
                                            return_indexers=return_indexers)
            else:
                return self._join_non_unique(other, how=how,
                                             return_indexers=return_indexers)
        elif self.is_monotonic and other.is_monotonic:
            try:
                return self._join_monotonic(other, how=how,
                                            return_indexers=return_indexers)
            except TypeError:
                pass

        if how == 'left':
            join_index = self
        elif how == 'right':
            join_index = other
        elif how == 'inner':
            # TODO: sort=False here for backwards compat. It may
            # be better to use the sort parameter passed into join
            join_index = self.intersection(other, sort=False)
        elif how == 'outer':
            # TODO: sort=True here for backwards compat. It may
            # be better to use the sort parameter passed into join
            join_index = self.union(other)

        if sort:
            join_index = join_index.sort_values()

        if return_indexers:
            if join_index is self:
                lindexer = None
            else:
                lindexer = self.get_indexer(join_index)
            if join_index is other:
                rindexer = None
            else:
                rindexer = other.get_indexer(join_index)
            return join_index, lindexer, rindexer
        else:
            return join_index

    def _join_multi(self, other, how, return_indexers=True):
        from .multi import MultiIndex
        from pandas.core.reshape.merge import _restore_dropped_levels_multijoin

        # figure out join names
        self_names = set(com._not_none(*self.names))
        other_names = set(com._not_none(*other.names))
        overlap = self_names & other_names

        # need at least 1 in common
        if not overlap:
            raise ValueError("cannot join with no overlapping index names")

        self_is_mi = isinstance(self, MultiIndex)
        other_is_mi = isinstance(other, MultiIndex)

        if self_is_mi and other_is_mi:

            # Drop the non-matching levels from left and right respectively
            ldrop_names = list(self_names - overlap)
            rdrop_names = list(other_names - overlap)

            self_jnlevels = self.droplevel(ldrop_names)
            other_jnlevels = other.droplevel(rdrop_names)

            # Join left and right
            # Join on same leveled multi-index frames is supported
            join_idx, lidx, ridx = self_jnlevels.join(other_jnlevels, how,
                                                      return_indexers=True)

            # Restore the dropped levels
            # Returned index level order is
            # common levels, ldrop_names, rdrop_names
            dropped_names = ldrop_names + rdrop_names

            levels, codes, names = (
                _restore_dropped_levels_multijoin(self, other,
                                                  dropped_names,
                                                  join_idx,
                                                  lidx, ridx))

            # Re-create the multi-index
            multi_join_idx = MultiIndex(levels=levels, codes=codes,
                                        names=names, verify_integrity=False)

            multi_join_idx = multi_join_idx.remove_unused_levels()

            return multi_join_idx, lidx, ridx

        jl = list(overlap)[0]

        # Case where only one index is multi
        # make the indices into mi's that match
        flip_order = False
        if self_is_mi:
            self, other = other, self
            flip_order = True
            # flip if join method is right or left
            how = {'right': 'left', 'left': 'right'}.get(how, how)

        level = other.names.index(jl)
        result = self._join_level(other, level, how=how,
                                  return_indexers=return_indexers)

        if flip_order:
            if isinstance(result, tuple):
                return result[0], result[2], result[1]
        return result

    def _join_non_unique(self, other, how='left', return_indexers=False):
        from pandas.core.reshape.merge import _get_join_indexers

        left_idx, right_idx = _get_join_indexers([self._ndarray_values],
                                                 [other._ndarray_values],
                                                 how=how,
                                                 sort=True)

        left_idx = ensure_platform_int(left_idx)
        right_idx = ensure_platform_int(right_idx)

        join_index = np.asarray(self._ndarray_values.take(left_idx))
        mask = left_idx == -1
        np.putmask(join_index, mask, other._ndarray_values.take(right_idx))

        join_index = self._wrap_joined_index(join_index, other)

        if return_indexers:
            return join_index, left_idx, right_idx
        else:
            return join_index

    def _join_level(self, other, level, how='left', return_indexers=False,
                    keep_order=True):
        """
        The join method *only* affects the level of the resulting
        MultiIndex. Otherwise it just exactly aligns the Index data to the
        labels of the level in the MultiIndex.

        If ```keep_order == True```, the order of the data indexed by the
        MultiIndex will not be changed; otherwise, it will tie out
        with `other`.
        """
        from .multi import MultiIndex

        def _get_leaf_sorter(labels):
            """
            Returns sorter for the inner most level while preserving the
            order of higher levels.
            """
            if labels[0].size == 0:
                return np.empty(0, dtype='int64')

            if len(labels) == 1:
                lab = ensure_int64(labels[0])
                sorter, _ = libalgos.groupsort_indexer(lab, 1 + lab.max())
                return sorter

            # find indexers of beginning of each set of
            # same-key labels w.r.t all but last level
            tic = labels[0][:-1] != labels[0][1:]
            for lab in labels[1:-1]:
                tic |= lab[:-1] != lab[1:]

            starts = np.hstack(([True], tic, [True])).nonzero()[0]
            lab = ensure_int64(labels[-1])
            return lib.get_level_sorter(lab, ensure_int64(starts))

        if isinstance(self, MultiIndex) and isinstance(other, MultiIndex):
            raise TypeError('Join on level between two MultiIndex objects '
                            'is ambiguous')

        left, right = self, other

        flip_order = not isinstance(self, MultiIndex)
        if flip_order:
            left, right = right, left
            how = {'right': 'left', 'left': 'right'}.get(how, how)

        level = left._get_level_number(level)
        old_level = left.levels[level]

        if not right.is_unique:
            raise NotImplementedError('Index._join_level on non-unique index '
                                      'is not implemented')

        new_level, left_lev_indexer, right_lev_indexer = \
            old_level.join(right, how=how, return_indexers=True)

        if left_lev_indexer is None:
            if keep_order or len(left) == 0:
                left_indexer = None
                join_index = left
            else:  # sort the leaves
                left_indexer = _get_leaf_sorter(left.codes[:level + 1])
                join_index = left[left_indexer]

        else:
            left_lev_indexer = ensure_int64(left_lev_indexer)
            rev_indexer = lib.get_reverse_indexer(left_lev_indexer,
                                                  len(old_level))

            new_lev_codes = algos.take_nd(rev_indexer, left.codes[level],
                                          allow_fill=False)

            new_codes = list(left.codes)
            new_codes[level] = new_lev_codes

            new_levels = list(left.levels)
            new_levels[level] = new_level

            if keep_order:  # just drop missing values. o.w. keep order
                left_indexer = np.arange(len(left), dtype=np.intp)
                mask = new_lev_codes != -1
                if not mask.all():
                    new_codes = [lab[mask] for lab in new_codes]
                    left_indexer = left_indexer[mask]

            else:  # tie out the order with other
                if level == 0:  # outer most level, take the fast route
                    ngroups = 1 + new_lev_codes.max()
                    left_indexer, counts = libalgos.groupsort_indexer(
                        new_lev_codes, ngroups)

                    # missing values are placed first; drop them!
                    left_indexer = left_indexer[counts[0]:]
                    new_codes = [lab[left_indexer] for lab in new_codes]

                else:  # sort the leaves
                    mask = new_lev_codes != -1
                    mask_all = mask.all()
                    if not mask_all:
                        new_codes = [lab[mask] for lab in new_codes]

                    left_indexer = _get_leaf_sorter(new_codes[:level + 1])
                    new_codes = [lab[left_indexer] for lab in new_codes]

                    # left_indexers are w.r.t masked frame.
                    # reverse to original frame!
                    if not mask_all:
                        left_indexer = mask.nonzero()[0][left_indexer]

            join_index = MultiIndex(levels=new_levels, codes=new_codes,
                                    names=left.names, verify_integrity=False)

        if right_lev_indexer is not None:
            right_indexer = algos.take_nd(right_lev_indexer,
                                          join_index.codes[level],
                                          allow_fill=False)
        else:
            right_indexer = join_index.codes[level]

        if flip_order:
            left_indexer, right_indexer = right_indexer, left_indexer

        if return_indexers:
            left_indexer = (None if left_indexer is None
                            else ensure_platform_int(left_indexer))
            right_indexer = (None if right_indexer is None
                             else ensure_platform_int(right_indexer))
            return join_index, left_indexer, right_indexer
        else:
            return join_index

    def _join_monotonic(self, other, how='left', return_indexers=False):
        if self.equals(other):
            ret_index = other if how == 'right' else self
            if return_indexers:
                return ret_index, None, None
            else:
                return ret_index

        sv = self._ndarray_values
        ov = other._ndarray_values

        if self.is_unique and other.is_unique:
            # We can perform much better than the general case
            if how == 'left':
                join_index = self
                lidx = None
                ridx = self._left_indexer_unique(sv, ov)
            elif how == 'right':
                join_index = other
                lidx = self._left_indexer_unique(ov, sv)
                ridx = None
            elif how == 'inner':
                join_index, lidx, ridx = self._inner_indexer(sv, ov)
                join_index = self._wrap_joined_index(join_index, other)
            elif how == 'outer':
                join_index, lidx, ridx = self._outer_indexer(sv, ov)
                join_index = self._wrap_joined_index(join_index, other)
        else:
            if how == 'left':
                join_index, lidx, ridx = self._left_indexer(sv, ov)
            elif how == 'right':
                join_index, ridx, lidx = self._left_indexer(ov, sv)
            elif how == 'inner':
                join_index, lidx, ridx = self._inner_indexer(sv, ov)
            elif how == 'outer':
                join_index, lidx, ridx = self._outer_indexer(sv, ov)
            join_index = self._wrap_joined_index(join_index, other)

        if return_indexers:
            lidx = None if lidx is None else ensure_platform_int(lidx)
            ridx = None if ridx is None else ensure_platform_int(ridx)
            return join_index, lidx, ridx
        else:
            return join_index

    def _wrap_joined_index(self, joined, other):
        name = get_op_result_name(self, other)
        return Index(joined, name=name)

    # --------------------------------------------------------------------
    # Uncategorized Methods

    @property
    def values(self):
        """
        Return an array representing the data in the Index.

        .. warning::

           We recommend using :attr:`Index.array` or
           :meth:`Index.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        array: numpy.ndarray or ExtensionArray

        See Also
        --------
        Index.array : Reference to the underlying data.
        Index.to_numpy : A NumPy array representing the underlying data.

        Return the underlying data as an ndarray.
        """
        return self._data.view(np.ndarray)

    @property
    def _values(self):
        # type: () -> Union[ExtensionArray, Index, np.ndarray]
        # TODO(EA): remove index types as they become extension arrays
        """
        The best array representation.

        This is an ndarray, ExtensionArray, or Index subclass. This differs
        from ``_ndarray_values``, which always returns an ndarray.

        Both ``_values`` and ``_ndarray_values`` are consistent between
        ``Series`` and ``Index``.

        It may differ from the public '.values' method.

        index             | values          | _values       | _ndarray_values |
        ----------------- | --------------- | ------------- | --------------- |
        Index             | ndarray         | ndarray       | ndarray         |
        CategoricalIndex  | Categorical     | Categorical   | ndarray[int]    |
        DatetimeIndex     | ndarray[M8ns]   | ndarray[M8ns] | ndarray[M8ns]   |
        DatetimeIndex[tz] | ndarray[M8ns]   | DTI[tz]       | ndarray[M8ns]   |
        PeriodIndex       | ndarray[object] | PeriodArray   | ndarray[int]    |
        IntervalIndex     | IntervalArray   | IntervalArray | ndarray[object] |

        See Also
        --------
        values
        _ndarray_values
        """
        return self._data

    def get_values(self):
        """
        Return `Index` data as an `numpy.ndarray`.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of the `Index` values.

        See Also
        --------
        Index.values : The attribute that get_values wraps.

        Examples
        --------
        Getting the `Index` values of a `DataFrame`:

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...                    index=['a', 'b', 'c'], columns=['A', 'B', 'C'])
        >>> df
           A  B  C
        a  1  2  3
        b  4  5  6
        c  7  8  9
        >>> df.index.get_values()
        array(['a', 'b', 'c'], dtype=object)

        Standalone `Index` values:

        >>> idx = pd.Index(['1', '2', '3'])
        >>> idx.get_values()
        array(['1', '2', '3'], dtype=object)

        `MultiIndex` arrays also have only one dimension:

        >>> midx = pd.MultiIndex.from_arrays([[1, 2, 3], ['a', 'b', 'c']],
        ...                                  names=('number', 'letter'))
        >>> midx.get_values()
        array([(1, 'a'), (2, 'b'), (3, 'c')], dtype=object)
        >>> midx.get_values().ndim
        1
        """
        return self.values

    @Appender(IndexOpsMixin.memory_usage.__doc__)
    def memory_usage(self, deep=False):
        result = super(Index, self).memory_usage(deep=deep)

        # include our engine hashtable
        result += self._engine.sizeof(deep=deep)
        return result

    _index_shared_docs['where'] = """
        Return an Index of same shape as self and whose corresponding
        entries are from self where cond is True and otherwise are from
        other.

        .. versionadded:: 0.19.0

        Parameters
        ----------
        cond : boolean array-like with the same length as self
        other : scalar, or array-like
        """

    @Appender(_index_shared_docs['where'])
    def where(self, cond, other=None):
        if other is None:
            other = self._na_value

        dtype = self.dtype
        values = self.values

        if is_bool(other) or is_bool_dtype(other):

            # bools force casting
            values = values.astype(object)
            dtype = None

        values = np.where(cond, values, other)

        if self._is_numeric_dtype and np.any(isna(values)):
            # We can't coerce to the numeric dtype of "self" (unless
            # it's float) if there are NaN values in our output.
            dtype = None

        return self._shallow_copy_with_infer(values, dtype=dtype)

    # construction helpers
    @classmethod
    def _try_convert_to_int_index(cls, data, copy, name, dtype):
        """
        Attempt to convert an array of data into an integer index.

        Parameters
        ----------
        data : The data to convert.
        copy : Whether to copy the data or not.
        name : The name of the index returned.

        Returns
        -------
        int_index : data converted to either an Int64Index or a
                    UInt64Index

        Raises
        ------
        ValueError if the conversion was not successful.
        """

        from .numeric import Int64Index, UInt64Index
        if not is_unsigned_integer_dtype(dtype):
            # skip int64 conversion attempt if uint-like dtype is passed, as
            # this could return Int64Index when UInt64Index is what's desrired
            try:
                res = data.astype('i8', copy=False)
                if (res == data).all():
                    return Int64Index(res, copy=copy, name=name)
            except (OverflowError, TypeError, ValueError):
                pass

        # Conversion to int64 failed (possibly due to overflow) or was skipped,
        # so let's try now with uint64.
        try:
            res = data.astype('u8', copy=False)
            if (res == data).all():
                return UInt64Index(res, copy=copy, name=name)
        except (OverflowError, TypeError, ValueError):
            pass

        raise ValueError

    @classmethod
    def _scalar_data_error(cls, data):
        raise TypeError('{0}(...) must be called with a collection of some '
                        'kind, {1} was passed'.format(cls.__name__,
                                                      repr(data)))

    @classmethod
    def _string_data_error(cls, data):
        raise TypeError('String dtype not supported, you may need '
                        'to explicitly cast to a numeric type')

    @classmethod
    def _coerce_to_ndarray(cls, data):
        """
        Coerces data to ndarray.

        Converts other iterables to list first and then to array.
        Does not touch ndarrays.

        Raises
        ------
        TypeError
            When the data passed in is a scalar.
        """

        if not isinstance(data, (np.ndarray, Index)):
            if data is None or is_scalar(data):
                cls._scalar_data_error(data)

            # other iterable of some kind
            if not isinstance(data, (ABCSeries, list, tuple)):
                data = list(data)
            data = np.asarray(data)
        return data

    def _coerce_scalar_to_index(self, item):
        """
        We need to coerce a scalar to a compat for our index type.

        Parameters
        ----------
        item : scalar item to coerce
        """
        dtype = self.dtype

        if self._is_numeric_dtype and isna(item):
            # We can't coerce to the numeric dtype of "self" (unless
            # it's float) if there are NaN values in our output.
            dtype = None

        return Index([item], dtype=dtype, **self._get_attributes_dict())

    def _to_safe_for_reshape(self):
        """
        Convert to object if we are a categorical.
        """
        return self

    def _convert_for_op(self, value):
        """
        Convert value to be insertable to ndarray.
        """
        return value

    def _assert_can_do_op(self, value):
        """
        Check value is valid for scalar op.
        """
        if not is_scalar(value):
            msg = "'value' must be a scalar, passed: {0}"
            raise TypeError(msg.format(type(value).__name__))

    @property
    def _has_complex_internals(self):
        # to disable groupby tricks in MultiIndex
        return False

    def _is_memory_usage_qualified(self):
        """
        Return a boolean if we need a qualified .info display.
        """
        return self.is_object()

    def is_type_compatible(self, kind):
        return kind == self.inferred_type

    _index_shared_docs['contains'] = """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')

        >>> idx.contains(2)
        True
        >>> idx.contains(6)
        False

        This is equivalent to:

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """

    @Appender(_index_shared_docs['contains'] % _index_doc_kwargs)
    def __contains__(self, key):
        hash(key)
        try:
            return key in self._engine
        except (OverflowError, TypeError, ValueError):
            return False

    @Appender(_index_shared_docs['contains'] % _index_doc_kwargs)
    def contains(self, key):
        hash(key)
        try:
            return key in self._engine
        except (TypeError, ValueError):
            return False

    def __hash__(self):
        raise TypeError("unhashable type: %r" % type(self).__name__)

    def __setitem__(self, key, value):
        raise TypeError("Index does not support mutable operations")

    def __getitem__(self, key):
        """
        Override numpy.ndarray's __getitem__ method to work as desired.

        This function adds lists and Series as valid boolean indexers
        (ndarrays only supports ndarray with dtype=bool).

        If resulting ndim != 1, plain ndarray is returned instead of
        corresponding `Index` subclass.

        """
        # There's no custom logic to be implemented in __getslice__, so it's
        # not overloaded intentionally.
        getitem = self._data.__getitem__
        promote = self._shallow_copy

        if is_scalar(key):
            key = com.cast_scalar_indexer(key)
            return getitem(key)

        if isinstance(key, slice):
            # This case is separated from the conditional above to avoid
            # pessimization of basic indexing.
            return promote(getitem(key))

        if com.is_bool_indexer(key):
            key = np.asarray(key, dtype=bool)

        key = com.values_from_object(key)
        result = getitem(key)
        if not is_scalar(result):
            return promote(result)
        else:
            return result

    def _can_hold_identifiers_and_holds_name(self, name):
        """
        Faster check for ``name in self`` when we know `name` is a Python
        identifier (e.g. in NDFrame.__getattr__, which hits this to support
        . key lookup). For indexes that can't hold identifiers (everything
        but object & categorical) we just return False.

        https://github.com/pandas-dev/pandas/issues/19764
        """
        if self.is_object() or self.is_categorical():
            return name in self
        return False

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        appended : Index
        """

        to_concat = [self]

        if isinstance(other, (list, tuple)):
            to_concat = to_concat + list(other)
        else:
            to_concat.append(other)

        for obj in to_concat:
            if not isinstance(obj, Index):
                raise TypeError('all inputs must be Index')

        names = {obj.name for obj in to_concat}
        name = None if len(names) > 1 else self.name

        return self._concat(to_concat, name)

    def _concat(self, to_concat, name):

        typs = _concat.get_dtype_kinds(to_concat)

        if len(typs) == 1:
            return self._concat_same_dtype(to_concat, name=name)
        return _concat._concat_index_asobject(to_concat, name=name)

    def _concat_same_dtype(self, to_concat, name):
        """
        Concatenate to_concat which has the same class.
        """
        # must be overridden in specific classes
        return _concat._concat_index_asobject(to_concat, name)

    def putmask(self, mask, value):
        """
        Return a new Index of the values set with the mask.

        See Also
        --------
        numpy.ndarray.putmask
        """
        values = self.values.copy()
        try:
            np.putmask(values, mask, self._convert_for_op(value))
            return self._shallow_copy(values)
        except (ValueError, TypeError) as err:
            if is_object_dtype(self):
                raise err

            # coerces to object
            return self.astype(object).putmask(mask, value)

    def equals(self, other):
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        if is_object_dtype(self) and not is_object_dtype(other):
            # if other is not object, use other's logic for coercion
            return other.equals(self)

        try:
            return array_equivalent(com.values_from_object(self),
                                    com.values_from_object(other))
        except Exception:
            return False

    def identical(self, other):
        """
        Similar to equals, but check that other comparable attributes are
        also equal.
        """
        return (self.equals(other) and
                all((getattr(self, c, None) == getattr(other, c, None)
                     for c in self._comparables)) and
                type(self) == type(other))

    def asof(self, label):
        """
        Return the label from the index, or, if not present, the previous one.

        Assuming that the index is sorted, return the passed index label if it
        is in the index, or return the previous index label if the passed one
        is not in the index.

        Parameters
        ----------
        label : object
            The label up to which the method returns the latest index label.

        Returns
        -------
        object
            The passed label if it is in the index. The previous label if the
            passed label is not in the sorted index or `NaN` if there is no
            such label.

        See Also
        --------
        Series.asof : Return the latest value in a Series up to the
            passed index.
        merge_asof : Perform an asof merge (similar to left join but it
            matches on nearest key rather than equal key).
        Index.get_loc : An `asof` is a thin wrapper around `get_loc`
            with method='pad'.

        Examples
        --------
        `Index.asof` returns the latest index label up to the passed label.

        >>> idx = pd.Index(['2013-12-31', '2014-01-02', '2014-01-03'])
        >>> idx.asof('2014-01-01')
        '2013-12-31'

        If the label is in the index, the method returns the passed label.

        >>> idx.asof('2014-01-02')
        '2014-01-02'

        If all of the labels in the index are later than the passed label,
        NaN is returned.

        >>> idx.asof('1999-01-02')
        nan

        If the index is not sorted, an error is raised.

        >>> idx_not_sorted = pd.Index(['2013-12-31', '2015-01-02',
        ...                            '2014-01-03'])
        >>> idx_not_sorted.asof('2013-12-31')
        Traceback (most recent call last):
        ValueError: index must be monotonic increasing or decreasing
        """
        try:
            loc = self.get_loc(label, method='pad')
        except KeyError:
            return self._na_value
        else:
            if isinstance(loc, slice):
                loc = loc.indices(len(self))[-1]
            return self[loc]

    def asof_locs(self, where, mask):
        """
        Finds the locations (indices) of the labels from the index for
        every entry in the `where` argument.

        As in the `asof` function, if the label (a particular entry in
        `where`) is not in the index, the latest index label upto the
        passed label is chosen and its index returned.

        If all of the labels in the index are later than a label in `where`,
        -1 is returned.

        `mask` is used to ignore NA values in the index during calculation.

        Parameters
        ----------
        where : Index
            An Index consisting of an array of timestamps.
        mask : array-like
            Array of booleans denoting where values in the original
            data are not NA.

        Returns
        -------
        numpy.ndarray
            An array of locations (indices) of the labels from the Index
            which correspond to the return values of the `asof` function
            for every element in `where`.
        """
        locs = self.values[mask].searchsorted(where.values, side='right')
        locs = np.where(locs > 0, locs - 1, 0)

        result = np.arange(len(self))[mask].take(locs)

        first = mask.argmax()
        result[(locs == 0) & (where.values < self.values[first])] = -1

        return result

    def sort_values(self, return_indexer=False, ascending=True):
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.

        Returns
        -------
        sorted_index : pandas.Index
            Sorted copy of the index.
        indexer : numpy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        pandas.Series.sort_values : Sort values of a Series.
        pandas.DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([10, 100, 1, 1000])
        >>> idx
        Int64Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Int64Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Int64Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))
        """
        _as = self.argsort()
        if not ascending:
            _as = _as[::-1]

        sorted_index = self.take(_as)

        if return_indexer:
            return sorted_index, _as
        else:
            return sorted_index

    def sort(self, *args, **kwargs):
        raise TypeError("cannot sort an Index object in-place, use "
                        "sort_values instead")

    def shift(self, periods=1, freq=None):
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.Index
            shifted index

        See Also
        --------
        Series.shift : Shift values of Series.

        Notes
        -----
        This method is only implemented for datetime-like index classes,
        i.e., DatetimeIndex, PeriodIndex and TimedeltaIndex.

        Examples
        --------
        Put the first 5 month starts of 2011 into an index.

        >>> month_starts = pd.date_range('1/1/2011', periods=5, freq='MS')
        >>> month_starts
        DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',
                       '2011-05-01'],
                      dtype='datetime64[ns]', freq='MS')

        Shift the index by 10 days.

        >>> month_starts.shift(10, freq='D')
        DatetimeIndex(['2011-01-11', '2011-02-11', '2011-03-11', '2011-04-11',
                       '2011-05-11'],
                      dtype='datetime64[ns]', freq=None)

        The default value of `freq` is the `freq` attribute of the index,
        which is 'MS' (month start) in this example.

        >>> month_starts.shift(10)
        DatetimeIndex(['2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01',
                       '2012-03-01'],
                      dtype='datetime64[ns]', freq='MS')
        """
        raise NotImplementedError("Not supported for type %s" %
                                  type(self).__name__)

    def argsort(self, *args, **kwargs):
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        *args
            Passed to `numpy.ndarray.argsort`.
        **kwargs
            Passed to `numpy.ndarray.argsort`.

        Returns
        -------
        numpy.ndarray
            Integer indices that would sort the index if used as
            an indexer.

        See Also
        --------
        numpy.argsort : Similar method for NumPy arrays.
        Index.sort_values : Return sorted copy of Index.

        Examples
        --------
        >>> idx = pd.Index(['b', 'a', 'd', 'c'])
        >>> idx
        Index(['b', 'a', 'd', 'c'], dtype='object')

        >>> order = idx.argsort()
        >>> order
        array([1, 0, 3, 2])

        >>> idx[order]
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        result = self.asi8
        if result is None:
            result = np.array(self)
        return result.argsort(*args, **kwargs)

    def get_value(self, series, key):
        """
        Fast lookup of value from 1-dimensional ndarray. Only use this if you
        know what you're doing.
        """

        # if we have something that is Index-like, then
        # use this, e.g. DatetimeIndex
        # Things like `Series._get_value` (via .at) pass the EA directly here.
        s = getattr(series, '_values', series)
        if isinstance(s, (ExtensionArray, Index)) and is_scalar(key):
            # GH 20882, 21257
            # Unify Index and ExtensionArray treatment
            # First try to convert the key to a location
            # If that fails, raise a KeyError if an integer
            # index, otherwise, see if key is an integer, and
            # try that
            try:
                iloc = self.get_loc(key)
                return s[iloc]
            except KeyError:
                if (len(self) > 0 and
                        (self.holds_integer() or self.is_boolean())):
                    raise
                elif is_integer(key):
                    return s[key]

        s = com.values_from_object(series)
        k = com.values_from_object(key)

        k = self._convert_scalar_indexer(k, kind='getitem')
        try:
            return self._engine.get_value(s, k,
                                          tz=getattr(series.dtype, 'tz', None))
        except KeyError as e1:
            if len(self) > 0 and (self.holds_integer() or self.is_boolean()):
                raise

            try:
                return libindex.get_value_box(s, key)
            except IndexError:
                raise
            except TypeError:
                # generator/iterator-like
                if is_iterator(key):
                    raise InvalidIndexError(key)
                else:
                    raise e1
            except Exception:  # pragma: no cover
                raise e1
        except TypeError:
            # python 3
            if is_scalar(key):  # pragma: no cover
                raise IndexError(key)
            raise InvalidIndexError(key)

    def set_value(self, arr, key, value):
        """
        Fast lookup of value from 1-dimensional ndarray.

        Notes
        -----
        Only use this if you know what you're doing.
        """
        self._engine.set_value(com.values_from_object(arr),
                               com.values_from_object(key), value)

    _index_shared_docs['get_indexer_non_unique'] = """
        Compute indexer and mask for new index given the current index. The
        indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : %(target_klass)s

        Returns
        -------
        indexer : ndarray of int
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.
        missing : ndarray of int
            An indexer into the target of the values not found.
            These correspond to the -1 in the indexer array
        """

    @Appender(_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target):
        target = ensure_index(target)
        if is_categorical(target):
            target = target.astype(target.dtype.categories.dtype)
        pself, ptarget = self._maybe_promote(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer_non_unique(ptarget)

        if self.is_all_dates:
            self = Index(self.asi8)
            tgt_values = target.asi8
        else:
            tgt_values = target._ndarray_values

        indexer, missing = self._engine.get_indexer_non_unique(tgt_values)
        return ensure_platform_int(indexer), missing

    def get_indexer_for(self, target, **kwargs):
        """
        Guaranteed return of an indexer even when non-unique.

        This dispatches to get_indexer or get_indexer_nonunique
        as appropriate.
        """
        if self.is_unique:
            return self.get_indexer(target, **kwargs)
        indexer, _ = self.get_indexer_non_unique(target, **kwargs)
        return indexer

    def _maybe_promote(self, other):
        # A hack, but it works
        from pandas import DatetimeIndex
        if self.inferred_type == 'date' and isinstance(other, DatetimeIndex):
            return DatetimeIndex(self), other
        elif self.inferred_type == 'boolean':
            if not is_object_dtype(self.dtype):
                return self.astype('object'), other.astype('object')
        return self, other

    def groupby(self, values):
        """
        Group the index labels by a given array of values.

        Parameters
        ----------
        values : array
            Values used to determine the groups.

        Returns
        -------
        groups : dict
            {group name -> group labels}
        """

        # TODO: if we are a MultiIndex, we can do better
        # that converting to tuples
        if isinstance(values, ABCMultiIndex):
            values = values.values
        values = ensure_categorical(values)
        result = values._reverse_indexer()

        # map to the label
        result = {k: self.take(v) for k, v in compat.iteritems(result)}

        return result

    def map(self, mapper, na_action=None):
        """
        Map values using input correspondence (a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        applied : Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """

        from .multi import MultiIndex
        new_values = super(Index, self)._map_values(
            mapper, na_action=na_action)

        attributes = self._get_attributes_dict()

        # we can return a MultiIndex
        if new_values.size and isinstance(new_values[0], tuple):
            if isinstance(self, MultiIndex):
                names = self.names
            elif attributes.get('name'):
                names = [attributes.get('name')] * len(new_values[0])
            else:
                names = None
            return MultiIndex.from_tuples(new_values,
                                          names=names)

        attributes['copy'] = False
        if not new_values.size:
            # empty
            attributes['dtype'] = self.dtype

        return Index(new_values, **attributes)

    def isin(self, values, level=None):
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.

            .. versionadded:: 0.18.1

               Support for values as a set.

        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        is_contained : ndarray
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1,2,3])
        >>> idx
        Int64Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.
        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> midx = pd.MultiIndex.from_arrays([[1,2,3],
        ...                                  ['red', 'blue', 'green']],
        ...                                  names=('number', 'color'))
        >>> midx
        MultiIndex(levels=[[1, 2, 3], ['blue', 'green', 'red']],
                   codes=[[0, 1, 2], [2, 0, 1]],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(['red', 'orange', 'yellow'], level='color')
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, 'red'), (3, 'red')])
        array([ True, False, False])

        For a DatetimeIndex, string values in `values` are converted to
        Timestamps.

        >>> dates = ['2000-03-11', '2000-03-12', '2000-03-13']
        >>> dti = pd.to_datetime(dates)
        >>> dti
        DatetimeIndex(['2000-03-11', '2000-03-12', '2000-03-13'],
        dtype='datetime64[ns]', freq=None)

        >>> dti.isin(['2000-03-11'])
        array([ True, False, False])
        """
        if level is not None:
            self._validate_index_level(level)
        return algos.isin(self, values)

    def _get_string_slice(self, key, use_lhs=True, use_rhs=True):
        # this is for partial string indexing,
        # overridden in DatetimeIndex, TimedeltaIndex and PeriodIndex
        raise NotImplementedError

    def slice_indexer(self, start=None, end=None, step=None, kind=None):
        """
        For an ordered or unique index, compute the slice indexer for input
        labels and step.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning
        end : label, default None
            If None, defaults to the end
        step : int, default None
        kind : string, default None

        Returns
        -------
        indexer : slice

        Raises
        ------
        KeyError : If key does not exist, or key is not unique and index is
            not ordered.

        Notes
        -----
        This function assumes that the data is sorted, so use at your own peril

        Examples
        ---------
        This is a method on all index types. For example you can do:

        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_indexer(start='b', end='c')
        slice(1, 3)

        >>> idx = pd.MultiIndex.from_arrays([list('abcd'), list('efgh')])
        >>> idx.slice_indexer(start='b', end=('c', 'g'))
        slice(1, 3)
        """
        start_slice, end_slice = self.slice_locs(start, end, step=step,
                                                 kind=kind)

        # return a slice
        if not is_scalar(start_slice):
            raise AssertionError("Start slice bound is non-scalar")
        if not is_scalar(end_slice):
            raise AssertionError("End slice bound is non-scalar")

        return slice(start_slice, end_slice, step)

    def _maybe_cast_indexer(self, key):
        """
        If we have a float key and are not a floating index, then try to cast
        to an int if equivalent.
        """

        if is_float(key) and not self.is_floating():
            try:
                ckey = int(key)
                if ckey == key:
                    key = ckey
            except (OverflowError, ValueError, TypeError):
                pass
        return key

    def _validate_indexer(self, form, key, kind):
        """
        If we are positional indexer, validate that we have appropriate
        typed bounds must be an integer.
        """
        assert kind in ['ix', 'loc', 'getitem', 'iloc']

        if key is None:
            pass
        elif is_integer(key):
            pass
        elif kind in ['iloc', 'getitem']:
            self._invalid_indexer(form, key)
        return key

    _index_shared_docs['_maybe_cast_slice_bound'] = """
        This function should be overloaded in subclasses that allow non-trivial
        casting on label-slice bounds, e.g. datetime-like indices allowing
        strings containing formatted datetimes.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}

        Returns
        -------
        label :  object

        Notes
        -----
        Value of `side` parameter should be validated in caller.

        """

    @Appender(_index_shared_docs['_maybe_cast_slice_bound'])
    def _maybe_cast_slice_bound(self, label, side, kind):
        assert kind in ['ix', 'loc', 'getitem', None]

        # We are a plain index here (sub-class override this method if they
        # wish to have special treatment for floats/ints, e.g. Float64Index and
        # datetimelike Indexes
        # reject them
        if is_float(label):
            if not (kind in ['ix'] and (self.holds_integer() or
                                        self.is_floating())):
                self._invalid_indexer('slice', label)

        # we are trying to find integer bounds on a non-integer based index
        # this is rejected (generally .loc gets you here)
        elif is_integer(label):
            self._invalid_indexer('slice', label)

        return label

    def _searchsorted_monotonic(self, label, side='left'):
        if self.is_monotonic_increasing:
            return self.searchsorted(label, side=side)
        elif self.is_monotonic_decreasing:
            # np.searchsorted expects ascending sort order, have to reverse
            # everything for it to work (element ordering, search side and
            # resulting value).
            pos = self[::-1].searchsorted(label, side='right' if side == 'left'
                                          else 'left')
            return len(self) - pos

        raise ValueError('index must be monotonic increasing or decreasing')

    def _get_loc_only_exact_matches(self, key):
        """
        This is overridden on subclasses (namely, IntervalIndex) to control
        get_slice_bound.
        """
        return self.get_loc(key)

    def get_slice_bound(self, label, side, kind):
        """
        Calculate slice bound that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}
        """
        assert kind in ['ix', 'loc', 'getitem', None]

        if side not in ('left', 'right'):
            raise ValueError("Invalid value for side kwarg,"
                             " must be either 'left' or 'right': %s" %
                             (side, ))

        original_label = label

        # For datetime indices label may be a string that has to be converted
        # to datetime boundary according to its resolution.
        label = self._maybe_cast_slice_bound(label, side, kind)

        # we need to look up the label
        try:
            slc = self._get_loc_only_exact_matches(label)
        except KeyError as err:
            try:
                return self._searchsorted_monotonic(label, side)
            except ValueError:
                # raise the original KeyError
                raise err

        if isinstance(slc, np.ndarray):
            # get_loc may return a boolean array or an array of indices, which
            # is OK as long as they are representable by a slice.
            if is_bool_dtype(slc):
                slc = lib.maybe_booleans_to_slice(slc.view('u1'))
            else:
                slc = lib.maybe_indices_to_slice(slc.astype('i8'), len(self))
            if isinstance(slc, np.ndarray):
                raise KeyError("Cannot get %s slice bound for non-unique "
                               "label: %r" % (side, original_label))

        if isinstance(slc, slice):
            if side == 'left':
                return slc.start
            else:
                return slc.stop
        else:
            if side == 'right':
                return slc + 1
            else:
                return slc

    def slice_locs(self, start=None, end=None, step=None, kind=None):
        """
        Compute slice locations for input labels.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning
        end : label, default None
            If None, defaults to the end
        step : int, defaults None
            If None, defaults to 1
        kind : {'ix', 'loc', 'getitem'} or None

        Returns
        -------
        start, end : int

        See Also
        --------
        Index.get_loc : Get location for a single label.

        Notes
        -----
        This method only works if the index is monotonic or unique.

        Examples
        ---------
        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_locs(start='b', end='c')
        (1, 3)
        """
        inc = (step is None or step >= 0)

        if not inc:
            # If it's a reverse slice, temporarily swap bounds.
            start, end = end, start

        start_slice = None
        if start is not None:
            start_slice = self.get_slice_bound(start, 'left', kind)
        if start_slice is None:
            start_slice = 0

        end_slice = None
        if end is not None:
            end_slice = self.get_slice_bound(end, 'right', kind)
        if end_slice is None:
            end_slice = len(self)

        if not inc:
            # Bounds at this moment are swapped, swap them back and shift by 1.
            #
            # slice_locs('B', 'A', step=-1): s='B', e='A'
            #
            #              s='A'                 e='B'
            # AFTER SWAP:    |                     |
            #                v ------------------> V
            #           -----------------------------------
            #           | | |A|A|A|A| | | | | |B|B| | | | |
            #           -----------------------------------
            #              ^ <------------------ ^
            # SHOULD BE:   |                     |
            #           end=s-1              start=e-1
            #
            end_slice, start_slice = start_slice - 1, end_slice - 1

            # i == -1 triggers ``len(self) + i`` selection that points to the
            # last element, not before-the-first one, subtracting len(self)
            # compensates that.
            if end_slice == -1:
                end_slice -= len(self)
            if start_slice == -1:
                start_slice -= len(self)

        return start_slice, end_slice

    def delete(self, loc):
        """
        Make new Index with passed location(-s) deleted.

        Returns
        -------
        new_index : Index
        """
        return self._shallow_copy(np.delete(self._data, loc))

    def insert(self, loc, item):
        """
        Make new Index inserting new item at location.

        Follows Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        new_index : Index
        """
        _self = np.asarray(self)
        item = self._coerce_scalar_to_index(item)._ndarray_values
        idx = np.concatenate((_self[:loc], item, _self[loc:]))
        return self._shallow_copy_with_infer(idx)

    def drop(self, labels, errors='raise'):
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        dropped : Index

        Raises
        ------
        KeyError
            If not all of the labels are found in the selected axis
        """
        arr_dtype = 'object' if self.dtype == 'object' else None
        labels = com.index_labels_to_array(labels, dtype=arr_dtype)
        indexer = self.get_indexer(labels)
        mask = indexer == -1
        if mask.any():
            if errors != 'ignore':
                raise KeyError(
                    '{} not found in axis'.format(labels[mask]))
            indexer = indexer[~mask]
        return self.delete(indexer)

    # --------------------------------------------------------------------
    # Generated Arithmetic, Comparison, and Unary Methods

    def _evaluate_with_timedelta_like(self, other, op):
        # Timedelta knows how to operate with np.array, so dispatch to that
        # operation and then wrap the results
        if self._is_numeric_dtype and op.__name__ in ['add', 'sub',
                                                      'radd', 'rsub']:
            raise TypeError("Operation {opname} between {cls} and {other} "
                            "is invalid".format(opname=op.__name__,
                                                cls=self.dtype,
                                                other=type(other).__name__))

        other = Timedelta(other)
        values = self.values

        with np.errstate(all='ignore'):
            result = op(values, other)

        attrs = self._get_attributes_dict()
        attrs = self._maybe_update_attributes(attrs)
        if op == divmod:
            return Index(result[0], **attrs), Index(result[1], **attrs)
        return Index(result, **attrs)

    def _evaluate_with_datetime_like(self, other, op):
        raise TypeError("can only perform ops with datetime like values")

    @classmethod
    def _add_comparison_methods(cls):
        """
        Add in comparison methods.
        """
        cls.__eq__ = _make_comparison_op(operator.eq, cls)
        cls.__ne__ = _make_comparison_op(operator.ne, cls)
        cls.__lt__ = _make_comparison_op(operator.lt, cls)
        cls.__gt__ = _make_comparison_op(operator.gt, cls)
        cls.__le__ = _make_comparison_op(operator.le, cls)
        cls.__ge__ = _make_comparison_op(operator.ge, cls)

    @classmethod
    def _add_numeric_methods_add_sub_disabled(cls):
        """
        Add in the numeric add/sub methods to disable.
        """
        cls.__add__ = make_invalid_op('__add__')
        cls.__radd__ = make_invalid_op('__radd__')
        cls.__iadd__ = make_invalid_op('__iadd__')
        cls.__sub__ = make_invalid_op('__sub__')
        cls.__rsub__ = make_invalid_op('__rsub__')
        cls.__isub__ = make_invalid_op('__isub__')

    @classmethod
    def _add_numeric_methods_disabled(cls):
        """
        Add in numeric methods to disable other than add/sub.
        """
        cls.__pow__ = make_invalid_op('__pow__')
        cls.__rpow__ = make_invalid_op('__rpow__')
        cls.__mul__ = make_invalid_op('__mul__')
        cls.__rmul__ = make_invalid_op('__rmul__')
        cls.__floordiv__ = make_invalid_op('__floordiv__')
        cls.__rfloordiv__ = make_invalid_op('__rfloordiv__')
        cls.__truediv__ = make_invalid_op('__truediv__')
        cls.__rtruediv__ = make_invalid_op('__rtruediv__')
        if not compat.PY3:
            cls.__div__ = make_invalid_op('__div__')
            cls.__rdiv__ = make_invalid_op('__rdiv__')
        cls.__mod__ = make_invalid_op('__mod__')
        cls.__divmod__ = make_invalid_op('__divmod__')
        cls.__neg__ = make_invalid_op('__neg__')
        cls.__pos__ = make_invalid_op('__pos__')
        cls.__abs__ = make_invalid_op('__abs__')
        cls.__inv__ = make_invalid_op('__inv__')

    def _maybe_update_attributes(self, attrs):
        """
        Update Index attributes (e.g. freq) depending on op.
        """
        return attrs

    def _validate_for_numeric_unaryop(self, op, opstr):
        """
        Validate if we can perform a numeric unary operation.
        """
        if not self._is_numeric_dtype:
            raise TypeError("cannot evaluate a numeric op "
                            "{opstr} for type: {typ}"
                            .format(opstr=opstr, typ=type(self).__name__))

    def _validate_for_numeric_binop(self, other, op):
        """
        Return valid other; evaluate or raise TypeError if we are not of
        the appropriate type.

        Notes
        -----
        This is an internal method called by ops.
        """
        opstr = '__{opname}__'.format(opname=op.__name__)
        # if we are an inheritor of numeric,
        # but not actually numeric (e.g. DatetimeIndex/PeriodIndex)
        if not self._is_numeric_dtype:
            raise TypeError("cannot evaluate a numeric op {opstr} "
                            "for type: {typ}"
                            .format(opstr=opstr, typ=type(self).__name__))

        if isinstance(other, Index):
            if not other._is_numeric_dtype:
                raise TypeError("cannot evaluate a numeric op "
                                "{opstr} with type: {typ}"
                                .format(opstr=opstr, typ=type(other)))
        elif isinstance(other, np.ndarray) and not other.ndim:
            other = other.item()

        if isinstance(other, (Index, ABCSeries, np.ndarray)):
            if len(self) != len(other):
                raise ValueError("cannot evaluate a numeric op with "
                                 "unequal lengths")
            other = com.values_from_object(other)
            if other.dtype.kind not in ['f', 'i', 'u']:
                raise TypeError("cannot evaluate a numeric op "
                                "with a non-numeric dtype")
        elif isinstance(other, (ABCDateOffset, np.timedelta64, timedelta)):
            # higher up to handle
            pass
        elif isinstance(other, (datetime, np.datetime64)):
            # higher up to handle
            pass
        else:
            if not (is_float(other) or is_integer(other)):
                raise TypeError("can only perform ops with scalar values")

        return other

    @classmethod
    def _add_numeric_methods_binary(cls):
        """
        Add in numeric methods.
        """
        cls.__add__ = _make_arithmetic_op(operator.add, cls)
        cls.__radd__ = _make_arithmetic_op(ops.radd, cls)
        cls.__sub__ = _make_arithmetic_op(operator.sub, cls)
        cls.__rsub__ = _make_arithmetic_op(ops.rsub, cls)
        cls.__rpow__ = _make_arithmetic_op(ops.rpow, cls)
        cls.__pow__ = _make_arithmetic_op(operator.pow, cls)

        cls.__truediv__ = _make_arithmetic_op(operator.truediv, cls)
        cls.__rtruediv__ = _make_arithmetic_op(ops.rtruediv, cls)
        if not compat.PY3:
            cls.__div__ = _make_arithmetic_op(operator.div, cls)
            cls.__rdiv__ = _make_arithmetic_op(ops.rdiv, cls)

        # TODO: rmod? rdivmod?
        cls.__mod__ = _make_arithmetic_op(operator.mod, cls)
        cls.__floordiv__ = _make_arithmetic_op(operator.floordiv, cls)
        cls.__rfloordiv__ = _make_arithmetic_op(ops.rfloordiv, cls)
        cls.__divmod__ = _make_arithmetic_op(divmod, cls)
        cls.__mul__ = _make_arithmetic_op(operator.mul, cls)
        cls.__rmul__ = _make_arithmetic_op(ops.rmul, cls)

    @classmethod
    def _add_numeric_methods_unary(cls):
        """
        Add in numeric unary methods.
        """
        def _make_evaluate_unary(op, opstr):

            def _evaluate_numeric_unary(self):

                self._validate_for_numeric_unaryop(op, opstr)
                attrs = self._get_attributes_dict()
                attrs = self._maybe_update_attributes(attrs)
                return Index(op(self.values), **attrs)

            _evaluate_numeric_unary.__name__ = opstr
            return _evaluate_numeric_unary

        cls.__neg__ = _make_evaluate_unary(operator.neg, '__neg__')
        cls.__pos__ = _make_evaluate_unary(operator.pos, '__pos__')
        cls.__abs__ = _make_evaluate_unary(np.abs, '__abs__')
        cls.__inv__ = _make_evaluate_unary(lambda x: -x, '__inv__')

    @classmethod
    def _add_numeric_methods(cls):
        cls._add_numeric_methods_unary()
        cls._add_numeric_methods_binary()

    @classmethod
    def _add_logical_methods(cls):
        """
        Add in logical methods.
        """
        _doc = """
        %(desc)s

        Parameters
        ----------
        *args
            These parameters will be passed to numpy.%(outname)s.
        **kwargs
            These parameters will be passed to numpy.%(outname)s.

        Returns
        -------
        %(outname)s : bool or array_like (if axis is specified)
            A single element array_like may be converted to bool."""

        _index_shared_docs['index_all'] = dedent("""

        See Also
        --------
        pandas.Index.any : Return whether any element in an Index is True.
        pandas.Series.any : Return whether any element in a Series is True.
        pandas.Series.all : Return whether all elements in a Series are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        **all**

        True, because nonzero integers are considered True.

        >>> pd.Index([1, 2, 3]).all()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 1, 2]).all()
        False

        **any**

        True, because ``1`` is considered True.

        >>> pd.Index([0, 0, 1]).any()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 0, 0]).any()
        False
        """)

        _index_shared_docs['index_any'] = dedent("""

        See Also
        --------
        pandas.Index.all : Return whether all elements are True.
        pandas.Series.all : Return whether all elements are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        >>> index = pd.Index([0, 1, 2])
        >>> index.any()
        True

        >>> index = pd.Index([0, 0, 0])
        >>> index.any()
        False
        """)

        def _make_logical_function(name, desc, f):
            @Substitution(outname=name, desc=desc)
            @Appender(_index_shared_docs['index_' + name])
            @Appender(_doc)
            def logical_func(self, *args, **kwargs):
                result = f(self.values)
                if (isinstance(result, (np.ndarray, ABCSeries, Index)) and
                        result.ndim == 0):
                    # return NumPy type
                    return result.dtype.type(result.item())
                else:  # pragma: no cover
                    return result

            logical_func.__name__ = name
            return logical_func

        cls.all = _make_logical_function('all', 'Return whether all elements '
                                                'are True.',
                                         np.all)
        cls.any = _make_logical_function('any',
                                         'Return whether any element is True.',
                                         np.any)

    @classmethod
    def _add_logical_methods_disabled(cls):
        """
        Add in logical methods to disable.
        """
        cls.all = make_invalid_op('all')
        cls.any = make_invalid_op('any')


Index._add_numeric_methods_disabled()
Index._add_logical_methods()
Index._add_comparison_methods()


def ensure_index_from_sequences(sequences, names=None):
    """
    Construct an index from sequences of data.

    A single sequence returns an Index. Many sequences returns a
    MultiIndex.

    Parameters
    ----------
    sequences : sequence of sequences
    names : sequence of str

    Returns
    -------
    index : Index or MultiIndex

    Examples
    --------
    >>> ensure_index_from_sequences([[1, 2, 3]], names=['name'])
    Int64Index([1, 2, 3], dtype='int64', name='name')

    >>> ensure_index_from_sequences([['a', 'a'], ['a', 'b']],
                                    names=['L1', 'L2'])
    MultiIndex(levels=[['a'], ['a', 'b']],
               codes=[[0, 0], [0, 1]],
               names=['L1', 'L2'])

    See Also
    --------
    ensure_index
    """
    from .multi import MultiIndex

    if len(sequences) == 1:
        if names is not None:
            names = names[0]
        return Index(sequences[0], name=names)
    else:
        return MultiIndex.from_arrays(sequences, names=names)


def ensure_index(index_like, copy=False):
    """
    Ensure that we have an index from some index-like object.

    Parameters
    ----------
    index : sequence
        An Index or other sequence
    copy : bool

    Returns
    -------
    index : Index or MultiIndex

    Examples
    --------
    >>> ensure_index(['a', 'b'])
    Index(['a', 'b'], dtype='object')

    >>> ensure_index([('a', 'a'),  ('b', 'c')])
    Index([('a', 'a'), ('b', 'c')], dtype='object')

    >>> ensure_index([['a', 'a'], ['b', 'c']])
    MultiIndex(levels=[['a'], ['b', 'c']],
               codes=[[0, 0], [0, 1]])

    See Also
    --------
    ensure_index_from_sequences
    """
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like
    if hasattr(index_like, 'name'):
        return Index(index_like, name=index_like.name, copy=copy)

    if is_iterator(index_like):
        index_like = list(index_like)

    # must check for exactly list here because of strict type
    # check in clean_index_list
    if isinstance(index_like, list):
        if type(index_like) != list:
            index_like = list(index_like)

        converted, all_arrays = lib.clean_index_list(index_like)

        if len(converted) > 0 and all_arrays:
            from .multi import MultiIndex
            return MultiIndex.from_arrays(converted)
        else:
            index_like = converted
    else:
        # clean_index_list does the equivalent of copying
        # so only need to do this if not list instance
        if copy:
            from copy import copy
            index_like = copy(index_like)

    return Index(index_like)


def _ensure_has_len(seq):
    """
    If seq is an iterator, put its values into a list.
    """
    try:
        len(seq)
    except TypeError:
        return list(seq)
    else:
        return seq


def _trim_front(strings):
    """
    Trims zeros and decimal points.
    """
    trimmed = strings
    while len(strings) > 0 and all(x[0] == ' ' for x in trimmed):
        trimmed = [x[1:] for x in trimmed]
    return trimmed


def _validate_join_method(method):
    if method not in ['left', 'right', 'inner', 'outer']:
        raise ValueError('do not recognize join method %s' % method)


def default_index(n):
    from pandas.core.index import RangeIndex
    return RangeIndex(0, n, name=None)
