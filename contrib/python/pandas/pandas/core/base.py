"""
Base and utility classes for pandas objects.
"""
import textwrap
import warnings

import numpy as np

import pandas._libs.lib as lib
import pandas.compat as compat
from pandas.compat import PYPY, OrderedDict, builtins, map, range
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, cache_readonly
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.common import (
    is_datetime64_ns_dtype, is_datetime64tz_dtype, is_datetimelike,
    is_extension_array_dtype, is_extension_type, is_list_like, is_object_dtype,
    is_scalar, is_timedelta64_ns_dtype)
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries
from pandas.core.dtypes.missing import isna

from pandas.core import algorithms, common as com
from pandas.core.accessor import DirNamesMixin
import pandas.core.nanops as nanops

_shared_docs = dict()
_indexops_doc_kwargs = dict(klass='IndexOpsMixin', inplace='',
                            unique='IndexOpsMixin', duplicated='IndexOpsMixin')


class StringMixin(object):
    """implements string methods so long as object defines a `__unicode__`
    method.

    Handles Python2/3 compatibility transparently.
    """
    # side note - this could be made into a metaclass if more than one
    #             object needs

    # ----------------------------------------------------------------------
    # Formatting

    def __unicode__(self):
        raise AbstractMethodError(self)

    def __str__(self):
        """
        Return a string representation for a particular Object

        Invoked by str(df) in both py2/py3.
        Yields Bytestring in Py2, Unicode String in py3.
        """

        if compat.PY3:
            return self.__unicode__()
        return self.__bytes__()

    def __bytes__(self):
        """
        Return a string representation for a particular object.

        Invoked by bytes(obj) in py3 only.
        Yields a bytestring in both py2/py3.
        """
        from pandas.core.config import get_option

        encoding = get_option("display.encoding")
        return self.__unicode__().encode(encoding, 'replace')

    def __repr__(self):
        """
        Return a string representation for a particular object.

        Yields Bytestring in Py2, Unicode String in py3.
        """
        return str(self)


class PandasObject(StringMixin, DirNamesMixin):

    """baseclass for various pandas objects"""

    @property
    def _constructor(self):
        """class constructor (for this class it's just `__class__`"""
        return self.__class__

    def __unicode__(self):
        """
        Return a string representation for a particular object.

        Invoked by unicode(obj) in py2 only. Yields a Unicode String in both
        py2/py3.
        """
        # Should be overwritten by base classes
        return object.__repr__(self)

    def _reset_cache(self, key=None):
        """
        Reset cached properties. If ``key`` is passed, only clears that key.
        """
        if getattr(self, '_cache', None) is None:
            return
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __sizeof__(self):
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
        if hasattr(self, 'memory_usage'):
            mem = self.memory_usage(deep=True)
            if not is_scalar(mem):
                mem = mem.sum()
            return int(mem)

        # no memory_usage attribute, so fall back to
        # object's 'sizeof'
        return super(PandasObject, self).__sizeof__()


class NoNewAttributesMixin(object):
    """Mixin which prevents adding new attributes.

    Prevents additional attributes via xxx.attribute = "something" after a
    call to `self.__freeze()`. Mainly used to prevent the user from using
    wrong attributes on a accessor (`Series.cat/.str/.dt`).

    If you really want to add a new attribute at a later time, you need to use
    `object.__setattr__(self, key, value)`.
    """

    def _freeze(self):
        """Prevents setting additional attributes"""
        object.__setattr__(self, "__frozen", True)

    # prevent adding any attribute via s.xxx.new_attribute = ...
    def __setattr__(self, key, value):
        # _cache is used by a decorator
        # We need to check both 1.) cls.__dict__ and 2.) getattr(self, key)
        # because
        # 1.) getattr is false for attributes that raise errors
        # 2.) cls.__dict__ doesn't traverse into base classes
        if (getattr(self, "__frozen", False) and not
                (key == "_cache" or
                 key in type(self).__dict__ or
                 getattr(self, key, None) is not None)):
            raise AttributeError("You cannot add any new attribute '{key}'".
                                 format(key=key))
        object.__setattr__(self, key, value)


class GroupByError(Exception):
    pass


class DataError(GroupByError):
    pass


class SpecificationError(GroupByError):
    pass


class SelectionMixin(object):
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """
    _selection = None
    _internal_names = ['_cache', '__setstate__']
    _internal_names_set = set(_internal_names)

    _builtin_table = OrderedDict((
        (builtins.sum, np.sum),
        (builtins.max, np.max),
        (builtins.min, np.min),
    ))

    _cython_table = OrderedDict((
        (builtins.sum, 'sum'),
        (builtins.max, 'max'),
        (builtins.min, 'min'),
        (np.all, 'all'),
        (np.any, 'any'),
        (np.sum, 'sum'),
        (np.nansum, 'sum'),
        (np.mean, 'mean'),
        (np.nanmean, 'mean'),
        (np.prod, 'prod'),
        (np.nanprod, 'prod'),
        (np.std, 'std'),
        (np.nanstd, 'std'),
        (np.var, 'var'),
        (np.nanvar, 'var'),
        (np.median, 'median'),
        (np.nanmedian, 'median'),
        (np.max, 'max'),
        (np.nanmax, 'max'),
        (np.min, 'min'),
        (np.nanmin, 'min'),
        (np.cumprod, 'cumprod'),
        (np.nancumprod, 'cumprod'),
        (np.cumsum, 'cumsum'),
        (np.nancumsum, 'cumsum'),
    ))

    @property
    def _selection_name(self):
        """
        return a name for myself; this would ideally be called
        the 'name' property, but we cannot conflict with the
        Series.name property which can be set
        """
        if self._selection is None:
            return None  # 'result'
        else:
            return self._selection

    @property
    def _selection_list(self):
        if not isinstance(self._selection, (list, tuple, ABCSeries,
                                            ABCIndexClass, np.ndarray)):
            return [self._selection]
        return self._selection

    @cache_readonly
    def _selected_obj(self):

        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            return self.obj[self._selection]

    @cache_readonly
    def ndim(self):
        return self._selected_obj.ndim

    @cache_readonly
    def _obj_with_exclusions(self):
        if self._selection is not None and isinstance(self.obj,
                                                      ABCDataFrame):
            return self.obj.reindex(columns=self._selection_list)

        if len(self.exclusions) > 0:
            return self.obj.drop(self.exclusions, axis=1)
        else:
            return self.obj

    def __getitem__(self, key):
        if self._selection is not None:
            raise IndexError('Column(s) {selection} already selected'
                             .format(selection=self._selection))

        if isinstance(key, (list, tuple, ABCSeries, ABCIndexClass,
                            np.ndarray)):
            if len(self.obj.columns.intersection(key)) != len(key):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError("Columns not found: {missing}"
                               .format(missing=str(bad_keys)[1:-1]))
            return self._gotitem(list(key), ndim=2)

        elif not getattr(self, 'as_index', False):
            if key not in self.obj.columns:
                raise KeyError("Column not found: {key}".format(key=key))
            return self._gotitem(key, ndim=2)

        else:
            if key not in self.obj:
                raise KeyError("Column not found: {key}".format(key=key))
            return self._gotitem(key, ndim=1)

    def _gotitem(self, key, ndim, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : string / list of selections
        ndim : 1,2
            requested ndim of result
        subset : object, default None
            subset to act on

        """
        raise AbstractMethodError(self)

    def aggregate(self, func, *args, **kwargs):
        raise AbstractMethodError(self)

    agg = aggregate

    def _try_aggregate_string_function(self, arg, *args, **kwargs):
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on ourselves
        - try to find a numpy function
        - raise

        """
        assert isinstance(arg, compat.string_types)

        f = getattr(self, arg, None)
        if f is not None:
            if callable(f):
                return f(*args, **kwargs)

            # people may try to aggregate on a non-callable attribute
            # but don't let them think they can pass args to it
            assert len(args) == 0
            assert len([kwarg for kwarg in kwargs
                        if kwarg not in ['axis', '_level']]) == 0
            return f

        f = getattr(np, arg, None)
        if f is not None:
            return f(self, *args, **kwargs)

        raise ValueError("{arg} is an unknown string function".format(arg=arg))

    def _aggregate(self, arg, *args, **kwargs):
        """
        provide an implementation for the aggregators

        Parameters
        ----------
        arg : string, dict, function
        *args : args to pass on to the function
        **kwargs : kwargs to pass on to the function

        Returns
        -------
        tuple of result, how

        Notes
        -----
        how can be a string describe the required post-processing, or
        None if not required
        """
        is_aggregator = lambda x: isinstance(x, (list, tuple, dict))
        is_nested_renamer = False

        _axis = kwargs.pop('_axis', None)
        if _axis is None:
            _axis = getattr(self, 'axis', 0)
        _level = kwargs.pop('_level', None)

        if isinstance(arg, compat.string_types):
            return self._try_aggregate_string_function(arg, *args,
                                                       **kwargs), None

        if isinstance(arg, dict):

            # aggregate based on the passed dict
            if _axis != 0:  # pragma: no cover
                raise ValueError('Can only pass dict with axis=0')

            obj = self._selected_obj

            def nested_renaming_depr(level=4):
                # deprecation of nested renaming
                # GH 15931
                warnings.warn(
                    ("using a dict with renaming "
                     "is deprecated and will be removed in a future "
                     "version"),
                    FutureWarning, stacklevel=level)

            # if we have a dict of any non-scalars
            # eg. {'A' : ['mean']}, normalize all to
            # be list-likes
            if any(is_aggregator(x) for x in compat.itervalues(arg)):
                new_arg = compat.OrderedDict()
                for k, v in compat.iteritems(arg):
                    if not isinstance(v, (tuple, list, dict)):
                        new_arg[k] = [v]
                    else:
                        new_arg[k] = v

                    # the keys must be in the columns
                    # for ndim=2, or renamers for ndim=1

                    # ok for now, but deprecated
                    # {'A': { 'ra': 'mean' }}
                    # {'A': { 'ra': ['mean'] }}
                    # {'ra': ['mean']}

                    # not ok
                    # {'ra' : { 'A' : 'mean' }}
                    if isinstance(v, dict):
                        is_nested_renamer = True

                        if k not in obj.columns:
                            msg = ('cannot perform renaming for {key} with a '
                                   'nested dictionary').format(key=k)
                            raise SpecificationError(msg)
                        nested_renaming_depr(4 + (_level or 0))

                    elif isinstance(obj, ABCSeries):
                        nested_renaming_depr()
                    elif (isinstance(obj, ABCDataFrame) and
                          k not in obj.columns):
                        raise KeyError(
                            "Column '{col}' does not exist!".format(col=k))

                arg = new_arg

            else:
                # deprecation of renaming keys
                # GH 15931
                keys = list(compat.iterkeys(arg))
                if (isinstance(obj, ABCDataFrame) and
                        len(obj.columns.intersection(keys)) != len(keys)):
                    nested_renaming_depr()

            from pandas.core.reshape.concat import concat

            def _agg_1dim(name, how, subset=None):
                """
                aggregate a 1-dim with how
                """
                colg = self._gotitem(name, ndim=1, subset=subset)
                if colg.ndim != 1:
                    raise SpecificationError("nested dictionary is ambiguous "
                                             "in aggregation")
                return colg.aggregate(how, _level=(_level or 0) + 1)

            def _agg_2dim(name, how):
                """
                aggregate a 2-dim with how
                """
                colg = self._gotitem(self._selection, ndim=2,
                                     subset=obj)
                return colg.aggregate(how, _level=None)

            def _agg(arg, func):
                """
                run the aggregations over the arg with func
                return an OrderedDict
                """
                result = compat.OrderedDict()
                for fname, agg_how in compat.iteritems(arg):
                    result[fname] = func(fname, agg_how)
                return result

            # set the final keys
            keys = list(compat.iterkeys(arg))
            result = compat.OrderedDict()

            # nested renamer
            if is_nested_renamer:
                result = list(_agg(arg, _agg_1dim).values())

                if all(isinstance(r, dict) for r in result):

                    result, results = compat.OrderedDict(), result
                    for r in results:
                        result.update(r)
                    keys = list(compat.iterkeys(result))

                else:

                    if self._selection is not None:
                        keys = None

            # some selection on the object
            elif self._selection is not None:

                sl = set(self._selection_list)

                # we are a Series like object,
                # but may have multiple aggregations
                if len(sl) == 1:

                    result = _agg(arg, lambda fname,
                                  agg_how: _agg_1dim(self._selection, agg_how))

                # we are selecting the same set as we are aggregating
                elif not len(sl - set(keys)):

                    result = _agg(arg, _agg_1dim)

                # we are a DataFrame, with possibly multiple aggregations
                else:

                    result = _agg(arg, _agg_2dim)

            # no selection
            else:

                try:
                    result = _agg(arg, _agg_1dim)
                except SpecificationError:

                    # we are aggregating expecting all 1d-returns
                    # but we have 2d
                    result = _agg(arg, _agg_2dim)

            # combine results

            def is_any_series():
                # return a boolean if we have *any* nested series
                return any(isinstance(r, ABCSeries)
                           for r in compat.itervalues(result))

            def is_any_frame():
                # return a boolean if we have *any* nested series
                return any(isinstance(r, ABCDataFrame)
                           for r in compat.itervalues(result))

            if isinstance(result, list):
                return concat(result, keys=keys, axis=1, sort=True), True

            elif is_any_frame():
                # we have a dict of DataFrames
                # return a MI DataFrame

                return concat([result[k] for k in keys],
                              keys=keys, axis=1), True

            elif isinstance(self, ABCSeries) and is_any_series():

                # we have a dict of Series
                # return a MI Series
                try:
                    result = concat(result)
                except TypeError:
                    # we want to give a nice error here if
                    # we have non-same sized objects, so
                    # we don't automatically broadcast

                    raise ValueError("cannot perform both aggregation "
                                     "and transformation operations "
                                     "simultaneously")

                return result, True

            # fall thru
            from pandas import DataFrame, Series
            try:
                result = DataFrame(result)
            except ValueError:

                # we have a dict of scalars
                result = Series(result,
                                name=getattr(self, 'name', None))

            return result, True
        elif is_list_like(arg) and arg not in compat.string_types:
            # we require a list, but not an 'str'
            return self._aggregate_multiple_funcs(arg,
                                                  _level=_level,
                                                  _axis=_axis), None
        else:
            result = None

        f = self._is_cython_func(arg)
        if f and not args and not kwargs:
            return getattr(self, f)(), None

        # caller can react
        return result, True

    def _aggregate_multiple_funcs(self, arg, _level, _axis):
        from pandas.core.reshape.concat import concat

        if _axis != 0:
            raise NotImplementedError("axis other than 0 is not supported")

        if self._selected_obj.ndim == 1:
            obj = self._selected_obj
        else:
            obj = self._obj_with_exclusions

        results = []
        keys = []

        # degenerate case
        if obj.ndim == 1:
            for a in arg:
                try:
                    colg = self._gotitem(obj.name, ndim=1, subset=obj)
                    results.append(colg.aggregate(a))

                    # make sure we find a good name
                    name = com.get_callable_name(a) or a
                    keys.append(name)
                except (TypeError, DataError):
                    pass
                except SpecificationError:
                    raise

        # multiples
        else:
            for index, col in enumerate(obj):
                try:
                    colg = self._gotitem(col, ndim=1,
                                         subset=obj.iloc[:, index])
                    results.append(colg.aggregate(arg))
                    keys.append(col)
                except (TypeError, DataError):
                    pass
                except ValueError:
                    # cannot aggregate
                    continue
                except SpecificationError:
                    raise

        # if we are empty
        if not len(results):
            raise ValueError("no results")

        try:
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError:

            # we are concatting non-NDFrame objects,
            # e.g. a list of scalars

            from pandas.core.dtypes.cast import is_nested_object
            from pandas import Series
            result = Series(results, index=keys, name=self.name)
            if is_nested_object(result):
                raise ValueError("cannot combine transform and "
                                 "aggregation operations")
            return result

    def _shallow_copy(self, obj=None, obj_type=None, **kwargs):
        """
        return a new object with the replacement attributes
        """
        if obj is None:
            obj = self._selected_obj.copy()
        if obj_type is None:
            obj_type = self._constructor
        if isinstance(obj, obj_type):
            obj = obj.obj
        for attr in self._attributes:
            if attr not in kwargs:
                kwargs[attr] = getattr(self, attr)
        return obj_type(obj, **kwargs)

    def _is_cython_func(self, arg):
        """
        if we define an internal function for this argument, return it
        """
        return self._cython_table.get(arg)

    def _is_builtin_func(self, arg):
        """
        if we define an builtin function for this argument, return it,
        otherwise return the arg
        """
        return self._builtin_table.get(arg, arg)


class IndexOpsMixin(object):
    """ common ops mixin to support a unified interface / docs for Series /
    Index
    """

    # ndarray compatibility
    __array_priority__ = 1000

    def transpose(self, *args, **kwargs):
        """
        Return the transpose, which is by definition self.
        """
        nv.validate_transpose(args, kwargs)
        return self

    T = property(transpose, doc="Return the transpose, which is by "
                                "definition self.")

    @property
    def _is_homogeneous_type(self):
        """
        Whether the object has a single dtype.

        By definition, Series and Index are always considered homogeneous.
        A MultiIndex may or may not be homogeneous, depending on the
        dtypes of the levels.

        See Also
        --------
        DataFrame._is_homogeneous_type
        MultiIndex._is_homogeneous_type
        """
        return True

    @property
    def shape(self):
        """
        Return a tuple of the shape of the underlying data.
        """
        return self._values.shape

    @property
    def ndim(self):
        """
        Number of dimensions of the underlying data, by definition 1.
        """
        return 1

    def item(self):
        """
        Return the first element of the underlying data as a python scalar.
        """
        try:
            return self.values.item()
        except IndexError:
            # copy numpy's message here because Py26 raises an IndexError
            raise ValueError('can only convert an array of size 1 to a '
                             'Python scalar')

    @property
    def data(self):
        """
        Return the data pointer of the underlying data.
        """
        warnings.warn("{obj}.data is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return self.values.data

    @property
    def itemsize(self):
        """
        Return the size of the dtype of the item of the underlying data.
        """
        warnings.warn("{obj}.itemsize is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return self._ndarray_values.itemsize

    @property
    def nbytes(self):
        """
        Return the number of bytes in the underlying data.
        """
        return self._values.nbytes

    @property
    def strides(self):
        """
        Return the strides of the underlying data.
        """
        warnings.warn("{obj}.strides is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return self._ndarray_values.strides

    @property
    def size(self):
        """
        Return the number of elements in the underlying data.
        """
        return len(self._values)

    @property
    def flags(self):
        """
        Return the ndarray.flags for the underlying data.
        """
        warnings.warn("{obj}.flags is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return self.values.flags

    @property
    def base(self):
        """
        Return the base object if the memory of the underlying data is shared.
        """
        warnings.warn("{obj}.base is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return self.values.base

    @property
    def array(self):
        # type: () -> ExtensionArray
        """
        The ExtensionArray of the data backing this Series or Index.

        .. versionadded:: 0.24.0

        Returns
        -------
        array : ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs ``.values`` which may require converting the
            data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================

        For any 3rd-party extension types, the array type will be an
        ExtensionArray.

        For all remaining dtypes ``.array`` will be a
        :class:`arrays.NumpyExtensionArray` wrapping the actual ndarray
        stored within. If you absolutely need a NumPy array (possibly with
        copying / coercing data), then use :meth:`Series.to_numpy` instead.

        Examples
        --------

        For regular NumPy types like int, and float, a PandasArray
        is returned.

        >>> pd.Series([1, 2, 3]).array
        <PandasArray>
        [1, 2, 3]
        Length: 3, dtype: int64

        For extension types, like Categorical, the actual ExtensionArray
        is returned

        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.array
        [a, b, a]
        Categories (2, object): [a, b]
        """
        result = self._values

        if is_datetime64_ns_dtype(result.dtype):
            from pandas.arrays import DatetimeArray
            result = DatetimeArray(result)
        elif is_timedelta64_ns_dtype(result.dtype):
            from pandas.arrays import TimedeltaArray
            result = TimedeltaArray(result)

        elif not is_extension_array_dtype(result.dtype):
            from pandas.core.arrays.numpy_ import PandasArray
            result = PandasArray(result)

        return result

    def to_numpy(self, dtype=None, copy=False):
        """
        A NumPy ndarray representing the values in this Series or Index.

        .. versionadded:: 0.24.0


        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        Series.array : Get the actual data stored within.
        Index.array : Get the actual data stored within.
        DataFrame.to_numpy : Similar method for DataFrame.

        Notes
        -----
        The returned array will be the same up to equality (values equal
        in `self` will be equal in the returned array; likewise for values
        that are not equal). When `self` contains an ExtensionArray, the
        dtype may be different. For example, for a category-dtype Series,
        ``to_numpy()`` will return a NumPy array and the categorical dtype
        will be lost.

        For NumPy dtypes, this will be a reference to the actual data stored
        in this Series or Index (assuming ``copy=False``). Modifying the result
        in place will modify the data stored in the Series or Index (not that
        we recommend doing that).

        For extension types, ``to_numpy()`` *may* require copying data and
        coercing the result to a NumPy type (possibly object), which may be
        expensive. When you need a no-copy reference to the underlying data,
        :attr:`Series.array` should be used instead.

        This table lays out the different dtypes and default return types of
        ``to_numpy()`` for various dtypes within pandas.

        ================== ================================
        dtype              array type
        ================== ================================
        category[T]        ndarray[T] (same dtype as input)
        period             ndarray[object] (Periods)
        interval           ndarray[object] (Intervals)
        IntegerNA          ndarray[object]
        datetime64[ns]     datetime64[ns]
        datetime64[ns, tz] ndarray[object] (Timestamps)
        ================== ================================

        Examples
        --------
        >>> ser = pd.Series(pd.Categorical(['a', 'b', 'a']))
        >>> ser.to_numpy()
        array(['a', 'b', 'a'], dtype=object)

        Specify the `dtype` to control how datetime-aware data is represented.
        Use ``dtype=object`` to return an ndarray of pandas :class:`Timestamp`
        objects, each with the correct ``tz``.

        >>> ser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
        >>> ser.to_numpy(dtype=object)
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET', freq='D'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET', freq='D')],
              dtype=object)

        Or ``dtype='datetime64[ns]'`` to return an ndarray of native
        datetime64 values. The values are converted to UTC and the timezone
        info is dropped.

        >>> ser.to_numpy(dtype="datetime64[ns]")
        ... # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00...'],
              dtype='datetime64[ns]')
        """
        if is_datetime64tz_dtype(self.dtype) and dtype is None:
            # note: this is going to change very soon.
            # I have a WIP PR making this unnecessary, but it's
            # a bit out of scope for the DatetimeArray PR.
            dtype = "object"

        result = np.asarray(self._values, dtype=dtype)
        # TODO(GH-24345): Avoid potential double copy
        if copy:
            result = result.copy()
        return result

    @property
    def _ndarray_values(self):
        # type: () -> np.ndarray
        """
        The data as an ndarray, possibly losing information.

        The expectation is that this is cheap to compute, and is primarily
        used for interacting with our indexers.

        - categorical -> codes
        """
        if is_extension_array_dtype(self):
            return self.array._ndarray_values
        return self.values

    @property
    def empty(self):
        return not self.size

    def max(self, axis=None, skipna=True):
        """
        Return the maximum value of the Index.

        Parameters
        ----------
        axis : int, optional
            For compatibility with NumPy. Only 0 or None are allowed.
        skipna : bool, default True

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.max()
        ('b', 2)
        """
        nv.validate_minmax_axis(axis)
        return nanops.nanmax(self._values, skipna=skipna)

    def argmax(self, axis=None, skipna=True):
        """
        Return a ndarray of the maximum argument indexer.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series
        skipna : bool, default True

        See Also
        --------
        numpy.ndarray.argmax
        """
        nv.validate_minmax_axis(axis)
        return nanops.nanargmax(self._values, skipna=skipna)

    def min(self, axis=None, skipna=True):
        """
        Return the minimum value of the Index.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series
        skipna : bool, default True

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.min()
        'a'

        For a MultiIndex, the minimum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.min()
        ('a', 1)
        """
        nv.validate_minmax_axis(axis)
        return nanops.nanmin(self._values, skipna=skipna)

    def argmin(self, axis=None, skipna=True):
        """
        Return a ndarray of the minimum argument indexer.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series
        skipna : bool, default True

        See Also
        --------
        numpy.ndarray.argmin
        """
        nv.validate_minmax_axis(axis)
        return nanops.nanargmin(self._values, skipna=skipna)

    def tolist(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        See Also
        --------
        numpy.ndarray.tolist
        """
        if is_datetimelike(self._values):
            return [com.maybe_box_datetimelike(x) for x in self._values]
        elif is_extension_array_dtype(self._values):
            return list(self._values)
        else:
            return self._values.tolist()

    to_list = tolist

    def __iter__(self):
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)
        """
        # We are explicity making element iterators.
        if is_datetimelike(self._values):
            return map(com.maybe_box_datetimelike, self._values)
        elif is_extension_array_dtype(self._values):
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self):
        """
        Return if I have any nans; enables various perf speedups.
        """
        return bool(isna(self).any())

    def _reduce(self, op, name, axis=0, skipna=True, numeric_only=None,
                filter_type=None, **kwds):
        """ perform the reduction type operation if we can """
        func = getattr(self, name, None)
        if func is None:
            raise TypeError("{klass} cannot perform the operation {op}".format(
                            klass=self.__class__.__name__, op=name))
        return func(skipna=skipna, **kwds)

    def _map_values(self, mapper, na_action=None):
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Parameters
        ----------
        mapper : function, dict, or Series
            The input correspondence object
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping function

        Returns
        -------
        applied : Union[Index, MultiIndex], inferred
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.

        """

        # we can fastpath dict/Series to an efficient map
        # as we know that we are not going to have to yield
        # python types
        if isinstance(mapper, dict):
            if hasattr(mapper, '__missing__'):
                # If a dictionary subclass defines a default value method,
                # convert mapper to a lookup function (GH #15999).
                dict_with_default = mapper
                mapper = lambda x: dict_with_default[x]
            else:
                # Dictionary does not have a default. Thus it's safe to
                # convert to an Series for efficiency.
                # we specify the keys here to handle the
                # possibility that they are tuples
                from pandas import Series
                mapper = Series(mapper)

        if isinstance(mapper, ABCSeries):
            # Since values were input this means we came from either
            # a dict or a series and mapper should be an index
            if is_extension_type(self.dtype):
                values = self._values
            else:
                values = self.values

            indexer = mapper.index.get_indexer(values)
            new_values = algorithms.take_1d(mapper._values, indexer)

            return new_values

        # we must convert to python types
        if is_extension_type(self.dtype):
            values = self._values
            if na_action is not None:
                raise NotImplementedError
            map_f = lambda values, f: values.map(f)
        else:
            values = self.astype(object)
            values = getattr(values, 'values', values)
            if na_action == 'ignore':
                def map_f(values, f):
                    return lib.map_infer_mask(values, f,
                                              isna(values).view(np.uint8))
            else:
                map_f = lib.map_infer

        # mapper is a function
        new_values = map_f(values, mapper)

        return new_values

    def value_counts(self, normalize=False, sort=True, ascending=False,
                     bins=None, dropna=True):
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : boolean, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : boolean, default True
            Sort by values.
        ascending : boolean, default False
            Sort in ascending order.
        bins : integer, optional
            Rather than count values, group them into half-open bins,
            a convenience for ``pd.cut``, only works with numeric data.
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.count: Number of non-NA elements in a DataFrame.

        Examples
        --------
        >>> index = pd.Index([3, 1, 2, 3, 4, np.nan])
        >>> index.value_counts()
        3.0    2
        4.0    1
        2.0    1
        1.0    1
        dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> s = pd.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(normalize=True)
        3.0    0.4
        4.0    0.2
        2.0    0.2
        1.0    0.2
        dtype: float64

        **bins**

        Bins can be useful for going from a continuous variable to a
        categorical variable; instead of counting unique
        apparitions of values, divide the index in the specified
        number of half-open bins.

        >>> s.value_counts(bins=3)
        (2.0, 3.0]      2
        (0.996, 2.0]    2
        (3.0, 4.0]      1
        dtype: int64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> s.value_counts(dropna=False)
        3.0    2
        NaN    1
        4.0    1
        2.0    1
        1.0    1
        dtype: int64
        """
        from pandas.core.algorithms import value_counts
        result = value_counts(self, sort=sort, ascending=ascending,
                              normalize=normalize, bins=bins, dropna=dropna)
        return result

    def unique(self):
        values = self._values

        if hasattr(values, 'unique'):

            result = values.unique()
        else:
            from pandas.core.algorithms import unique1d
            result = unique1d(values)

        return result

    def nunique(self, dropna=True):
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        dropna : boolean, default True
            Don't include NaN in the count.

        Returns
        -------
        nunique : int
        """
        uniqs = self.unique()
        n = len(uniqs)
        if dropna and isna(uniqs).any():
            n -= 1
        return n

    @property
    def is_unique(self):
        """
        Return boolean if values in the object are unique.

        Returns
        -------
        is_unique : boolean
        """
        return self.nunique(dropna=False) == len(self)

    @property
    def is_monotonic(self):
        """
        Return boolean if values in the object are
        monotonic_increasing.

        .. versionadded:: 0.19.0

        Returns
        -------
        is_monotonic : boolean
        """
        from pandas import Index
        return Index(self).is_monotonic

    is_monotonic_increasing = is_monotonic

    @property
    def is_monotonic_decreasing(self):
        """
        Return boolean if values in the object are
        monotonic_decreasing.

        .. versionadded:: 0.19.0

        Returns
        -------
        is_monotonic_decreasing : boolean
        """
        from pandas import Index
        return Index(self).is_monotonic_decreasing

    def memory_usage(self, deep=False):
        """
        Memory usage of the values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        See Also
        --------
        numpy.ndarray.nbytes

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy
        """
        if hasattr(self.array, 'memory_usage'):
            return self.array.memory_usage(deep=deep)

        v = self.array.nbytes
        if deep and is_object_dtype(self) and not PYPY:
            v += lib.memory_usage_of_objects(self.array)
        return v

    @Substitution(
        values='', order='', size_hint='',
        sort=textwrap.dedent("""\
            sort : boolean, default False
                Sort `uniques` and shuffle `labels` to maintain the
                relationship.
            """))
    @Appender(algorithms._shared_docs['factorize'])
    def factorize(self, sort=False, na_sentinel=-1):
        return algorithms.factorize(self, sort=sort, na_sentinel=na_sentinel)

    _shared_docs['searchsorted'] = (
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted %(klass)s `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Parameters
        ----------
        value : array_like
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.

        Returns
        -------
        int or array of int
            A scalar or array of insertion points with the
            same shape as `value`.

            .. versionchanged :: 0.24.0
                If `value` is a scalar, an int is now always returned.
                Previously, scalar inputs returned an 1-item array for
                :class:`Series` and :class:`Categorical`.

        See Also
        --------
        numpy.searchsorted

        Notes
        -----
        Binary search is used to find the required insertion points.

        Examples
        --------

        >>> x = pd.Series([1, 2, 3])
        >>> x
        0    1
        1    2
        2    3
        dtype: int64

        >>> x.searchsorted(4)
        3

        >>> x.searchsorted([0, 4])
        array([0, 3])

        >>> x.searchsorted([1, 3], side='left')
        array([0, 2])

        >>> x.searchsorted([1, 3], side='right')
        array([1, 3])

        >>> x = pd.Categorical(['apple', 'bread', 'bread',
                                'cheese', 'milk'], ordered=True)
        [apple, bread, bread, cheese, milk]
        Categories (4, object): [apple < bread < cheese < milk]

        >>> x.searchsorted('bread')
        1

        >>> x.searchsorted(['bread'], side='right')
        array([3])
        """)

    @Substitution(klass='IndexOpsMixin')
    @Appender(_shared_docs['searchsorted'])
    def searchsorted(self, value, side='left', sorter=None):
        # needs coercion on the key (DatetimeIndex does already)
        return self._values.searchsorted(value, side=side, sorter=sorter)

    def drop_duplicates(self, keep='first', inplace=False):
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if isinstance(self, ABCIndexClass):
            if self.is_unique:
                return self._shallow_copy()

        duplicated = self.duplicated(keep=keep)
        result = self[np.logical_not(duplicated)]
        if inplace:
            return self._update_inplace(result)
        else:
            return result

    def duplicated(self, keep='first'):
        from pandas.core.algorithms import duplicated
        if isinstance(self, ABCIndexClass):
            if self.is_unique:
                return np.zeros(len(self), dtype=np.bool)
            return duplicated(self, keep=keep)
        else:
            return self._constructor(duplicated(self, keep=keep),
                                     index=self.index).__finalize__(self)

    # ----------------------------------------------------------------------
    # abstracts

    def _update_inplace(self, result, **kwargs):
        raise AbstractMethodError(self)
