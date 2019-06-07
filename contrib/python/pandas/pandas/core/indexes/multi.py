# pylint: disable=E1101,E1103,W0232
from collections import OrderedDict
import datetime
from sys import getsizeof
import warnings

import numpy as np

from pandas._libs import (
    Timestamp, algos as libalgos, index as libindex, lib, tslibs)
import pandas.compat as compat
from pandas.compat import lrange, lzip, map, range, zip
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning, UnsortedIndexError
from pandas.util._decorators import Appender, cache_readonly, deprecate_kwarg

from pandas.core.dtypes.common import (
    ensure_int64, ensure_platform_int, is_categorical_dtype, is_hashable,
    is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar,
    pandas_dtype)
from pandas.core.dtypes.dtypes import ExtensionDtype, PandasExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame
from pandas.core.dtypes.missing import array_equivalent, isna

import pandas.core.algorithms as algos
import pandas.core.common as com
from pandas.core.config import get_option
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index, InvalidIndexError, _index_shared_docs, ensure_index)
from pandas.core.indexes.frozen import FrozenList, _ensure_frozen
import pandas.core.missing as missing

from pandas.io.formats.printing import pprint_thing

_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update(
    dict(klass='MultiIndex',
         target_klass='MultiIndex or list of tuples'))


class MultiIndexUIntEngine(libindex.BaseMultiIndexCodesEngine,
                           libindex.UInt64Engine):
    """
    This class manages a MultiIndex by mapping label combinations to positive
    integers.
    """
    _base = libindex.UInt64Engine

    def _codes_to_ints(self, codes):
        """
        Transform combination(s) of uint64 in one uint64 (each), in a strictly
        monotonic way (i.e. respecting the lexicographic order of integer
        combinations): see BaseMultiIndexCodesEngine documentation.

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint64
            Combinations of integers (one per row)

        Returns
        ------
        int_keys : scalar or 1-dimensional array, of dtype uint64
            Integer(s) representing one combination (each)
        """
        # Shift the representation of each level by the pre-calculated number
        # of bits:
        codes <<= self.offsets

        # Now sum and OR are in fact interchangeable. This is a simple
        # composition of the (disjunct) significant bits of each level (i.e.
        # each column in "codes") in a single positive integer:
        if codes.ndim == 1:
            # Single key
            return np.bitwise_or.reduce(codes)

        # Multiple keys
        return np.bitwise_or.reduce(codes, axis=1)


class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine,
                            libindex.ObjectEngine):
    """
    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """
    _base = libindex.ObjectEngine

    def _codes_to_ints(self, codes):
        """
        Transform combination(s) of uint64 in one Python integer (each), in a
        strictly monotonic way (i.e. respecting the lexicographic order of
        integer combinations): see BaseMultiIndexCodesEngine documentation.

        Parameters
        ----------
        codes : 1- or 2-dimensional array of dtype uint64
            Combinations of integers (one per row)

        Returns
        ------
        int_keys : int, or 1-dimensional array of dtype object
            Integer(s) representing one combination (each)
        """

        # Shift the representation of each level by the pre-calculated number
        # of bits. Since this can overflow uint64, first make sure we are
        # working with Python integers:
        codes = codes.astype('object') << self.offsets

        # Now sum and OR are in fact interchangeable. This is a simple
        # composition of the (disjunct) significant bits of each level (i.e.
        # each column in "codes") in a single positive integer (per row):
        if codes.ndim == 1:
            # Single key
            return np.bitwise_or.reduce(codes)

        # Multiple keys
        return np.bitwise_or.reduce(codes, axis=1)


class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.

        .. versionadded:: 0.24.0
    labels : sequence of arrays
        Integers for each level designating which label at each location.

        .. deprecated:: 0.24.0
            Use ``codes`` instead
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    Attributes
    ----------
    names
    levels
    codes
    nlevels
    levshape

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    is_lexsorted
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : The base pandas Index type.

    Examples
    ---------
    A new ``MultiIndex`` is typically constructed using one of the helper
    methods :meth:`MultiIndex.from_arrays`, :meth:`MultiIndex.from_product`
    and :meth:`MultiIndex.from_tuples`. For example (using ``.from_arrays``):

    >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
    >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
    MultiIndex(levels=[[1, 2], ['blue', 'red']],
               codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
               names=['number', 'color'])

    See further examples for how to construct a MultiIndex in the doc strings
    of the mentioned helper methods.

    Notes
    -----
    See the `user guide
    <http://pandas.pydata.org/pandas-docs/stable/advanced.html>`_ for more.
    """

    # initialize to zero-length tuples to make everything work
    _typ = 'multiindex'
    _names = FrozenList()
    _levels = FrozenList()
    _codes = FrozenList()
    _comparables = ['names']
    rename = Index.set_names

    # --------------------------------------------------------------------
    # Constructors

    @deprecate_kwarg(old_arg_name='labels', new_arg_name='codes')
    def __new__(cls, levels=None, codes=None, sortorder=None, names=None,
                dtype=None, copy=False, name=None,
                verify_integrity=True, _set_identity=True):

        # compat with Index
        if name is not None:
            names = name
        if levels is None or codes is None:
            raise TypeError("Must pass both levels and codes")
        if len(levels) != len(codes):
            raise ValueError('Length of levels and codes must be the same.')
        if len(levels) == 0:
            raise ValueError('Must pass non-zero number of levels/codes')

        result = object.__new__(MultiIndex)

        # we've already validated levels and codes, so shortcut here
        result._set_levels(levels, copy=copy, validate=False)
        result._set_codes(codes, copy=copy, validate=False)

        if names is not None:
            # handles name validation
            result._set_names(names)

        if sortorder is not None:
            result.sortorder = int(sortorder)
        else:
            result.sortorder = sortorder

        if verify_integrity:
            result._verify_integrity()
        if _set_identity:
            result._reset_identity()
        return result

    def _verify_integrity(self, codes=None, levels=None):
        """

        Parameters
        ----------
        codes : optional list
            Codes to check for validity. Defaults to current codes.
        levels : optional list
            Levels to check for validity. Defaults to current levels.

        Raises
        ------
        ValueError
            If length of levels and codes don't match, if the codes for any
            level would exceed level bounds, or there are any duplicate levels.
        """
        # NOTE: Currently does not check, among other things, that cached
        # nlevels matches nor that sortorder matches actually sortorder.
        codes = codes or self.codes
        levels = levels or self.levels

        if len(levels) != len(codes):
            raise ValueError("Length of levels and codes must match. NOTE:"
                             " this index is in an inconsistent state.")
        codes_length = len(self.codes[0])
        for i, (level, level_codes) in enumerate(zip(levels, codes)):
            if len(level_codes) != codes_length:
                raise ValueError("Unequal code lengths: %s" %
                                 ([len(code_) for code_ in codes]))
            if len(level_codes) and level_codes.max() >= len(level):
                raise ValueError("On level %d, code max (%d) >= length of"
                                 " level  (%d). NOTE: this index is in an"
                                 " inconsistent state" % (i, level_codes.max(),
                                                          len(level)))
            if not level.is_unique:
                raise ValueError("Level values must be unique: {values} on "
                                 "level {level}".format(
                                     values=[value for value in level],
                                     level=i))

    @classmethod
    def from_arrays(cls, arrays, sortorder=None, names=None):
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex(levels=[[1, 2], ['blue', 'red']],
                   codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
                   names=['number', 'color'])
        """
        if not is_list_like(arrays):
            raise TypeError("Input must be a list / sequence of array-likes.")
        elif is_iterator(arrays):
            arrays = list(arrays)

        # Check if lengths of all arrays are equal or not,
        # raise ValueError, if not
        for i in range(1, len(arrays)):
            if len(arrays[i]) != len(arrays[i - 1]):
                raise ValueError('all arrays must be same length')

        from pandas.core.arrays.categorical import _factorize_from_iterables

        codes, levels = _factorize_from_iterables(arrays)
        if names is None:
            names = [getattr(arr, "name", None) for arr in arrays]

        return MultiIndex(levels=levels, codes=codes, sortorder=sortorder,
                          names=names, verify_integrity=False)

    @classmethod
    def from_tuples(cls, tuples, sortorder=None, names=None):
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, u'red'), (1, u'blue'),
        ...           (2, u'red'), (2, u'blue')]
        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex(levels=[[1, 2], ['blue', 'red']],
                   codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
                   names=['number', 'color'])
        """
        if not is_list_like(tuples):
            raise TypeError('Input must be a list / sequence of tuple-likes.')
        elif is_iterator(tuples):
            tuples = list(tuples)

        if len(tuples) == 0:
            if names is None:
                msg = 'Cannot infer number of levels from empty list'
                raise TypeError(msg)
            arrays = [[]] * len(names)
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = tuples._values

            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            arrays = lzip(*tuples)

        return MultiIndex.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(cls, iterables, sortorder=None, names=None):
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> pd.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex(levels=[[0, 1, 2], ['green', 'purple']],
                   codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
                   names=['number', 'color'])
        """
        from pandas.core.arrays.categorical import _factorize_from_iterables
        from pandas.core.reshape.util import cartesian_product

        if not is_list_like(iterables):
            raise TypeError("Input must be a list / sequence of iterables.")
        elif is_iterator(iterables):
            iterables = list(iterables)

        codes, levels = _factorize_from_iterables(iterables)
        codes = cartesian_product(codes)
        return MultiIndex(levels, codes, sortorder=sortorder, names=names)

    @classmethod
    def from_frame(cls, df, sortorder=None, names=None):
        """
        Make a MultiIndex from a DataFrame.

        .. versionadded:: 0.24.0

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        sortorder : int, optional
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = pd.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> pd.MultiIndex.from_frame(df)
        MultiIndex(levels=[['HI', 'NJ'], ['Precip', 'Temp']],
                   codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> pd.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex(levels=[['HI', 'NJ'], ['Precip', 'Temp']],
                   codes=[[0, 0, 1, 1], [1, 0, 1, 0]],
                   names=['state', 'observation'])
        """
        if not isinstance(df, ABCDataFrame):
            raise TypeError("Input must be a DataFrame")

        column_names, columns = lzip(*df.iteritems())
        names = column_names if names is None else names
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    # --------------------------------------------------------------------

    @property
    def levels(self):
        return self._levels

    @property
    def _values(self):
        # We override here, since our parent uses _data, which we dont' use.
        return self.values

    @property
    def array(self):
        """
        Raises a ValueError for `MultiIndex` because there's no single
        array backing a MultiIndex.

        Raises
        ------
        ValueError
        """
        msg = ("MultiIndex has no single backing array. Use "
               "'MultiIndex.to_numpy()' to get a NumPy array of tuples.")
        raise ValueError(msg)

    @property
    def _is_homogeneous_type(self):
        """Whether the levels of a MultiIndex all have the same dtype.

        This looks at the dtypes of the levels.

        See Also
        --------
        Index._is_homogeneous_type
        DataFrame._is_homogeneous_type

        Examples
        --------
        >>> MultiIndex.from_tuples([
        ...     ('a', 'b'), ('a', 'c')])._is_homogeneous_type
        True
        >>> MultiIndex.from_tuples([
        ...     ('a', 1), ('a', 2)])._is_homogeneous_type
        False
        """
        return len({x.dtype for x in self.levels}) <= 1

    def _set_levels(self, levels, level=None, copy=False, validate=True,
                    verify_integrity=False):
        # This is NOT part of the levels property because it should be
        # externally not allowed to set levels. User beware if you change
        # _levels directly
        if validate and len(levels) == 0:
            raise ValueError('Must set non-zero number of levels.')
        if validate and level is None and len(levels) != self.nlevels:
            raise ValueError('Length of levels must match number of levels.')
        if validate and level is not None and len(levels) != len(level):
            raise ValueError('Length of levels must match length of level.')

        if level is None:
            new_levels = FrozenList(
                ensure_index(lev, copy=copy)._shallow_copy()
                for lev in levels)
        else:
            level = [self._get_level_number(l) for l in level]
            new_levels = list(self._levels)
            for l, v in zip(level, levels):
                new_levels[l] = ensure_index(v, copy=copy)._shallow_copy()
            new_levels = FrozenList(new_levels)

        if verify_integrity:
            self._verify_integrity(levels=new_levels)

        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)

        self._tuples = None
        self._reset_cache()

    def set_levels(self, levels, level=None, inplace=False,
                   verify_integrity=True):
        """
        Set new levels on MultiIndex. Defaults to returning
        new index.

        Parameters
        ----------
        levels : sequence or list of sequence
            new level(s) to apply
        level : int, level name, or sequence of int/level names (default None)
            level(s) to set (None for all levels)
        inplace : bool
            if True, mutates in place
        verify_integrity : bool (default True)
            if True, checks that levels and codes are compatible

        Returns
        -------
        new index (of same type and class...etc)

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples([(1, u'one'), (1, u'two'),
                                            (2, u'one'), (2, u'two')],
                                            names=['foo', 'bar'])
        >>> idx.set_levels([['a','b'], [1,2]])
        MultiIndex(levels=[[u'a', u'b'], [1, 2]],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                   names=[u'foo', u'bar'])
        >>> idx.set_levels(['a','b'], level=0)
        MultiIndex(levels=[[u'a', u'b'], [u'one', u'two']],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                   names=[u'foo', u'bar'])
        >>> idx.set_levels(['a','b'], level='bar')
        MultiIndex(levels=[[1, 2], [u'a', u'b']],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                   names=[u'foo', u'bar'])
        >>> idx.set_levels([['a','b'], [1,2]], level=[0,1])
        MultiIndex(levels=[[u'a', u'b'], [1, 2]],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                   names=[u'foo', u'bar'])
        """
        if is_list_like(levels) and not isinstance(levels, Index):
            levels = list(levels)

        if level is not None and not is_list_like(level):
            if not is_list_like(levels):
                raise TypeError("Levels must be list-like")
            if is_list_like(levels[0]):
                raise TypeError("Levels must be list-like")
            level = [level]
            levels = [levels]
        elif level is None or is_list_like(level):
            if not is_list_like(levels) or not is_list_like(levels[0]):
                raise TypeError("Levels must be list of lists-like")

        if inplace:
            idx = self
        else:
            idx = self._shallow_copy()
        idx._reset_identity()
        idx._set_levels(levels, level=level, validate=True,
                        verify_integrity=verify_integrity)
        if not inplace:
            return idx

    @property
    def codes(self):
        return self._codes

    @property
    def labels(self):
        warnings.warn((".labels was deprecated in version 0.24.0. "
                       "Use .codes instead."),
                      FutureWarning, stacklevel=2)
        return self.codes

    def _set_codes(self, codes, level=None, copy=False, validate=True,
                   verify_integrity=False):

        if validate and level is None and len(codes) != self.nlevels:
            raise ValueError("Length of codes must match number of levels")
        if validate and level is not None and len(codes) != len(level):
            raise ValueError('Length of codes must match length of levels.')

        if level is None:
            new_codes = FrozenList(
                _ensure_frozen(level_codes, lev, copy=copy)._shallow_copy()
                for lev, level_codes in zip(self.levels, codes))
        else:
            level = [self._get_level_number(l) for l in level]
            new_codes = list(self._codes)
            for lev_idx, level_codes in zip(level, codes):
                lev = self.levels[lev_idx]
                new_codes[lev_idx] = _ensure_frozen(
                    level_codes, lev, copy=copy)._shallow_copy()
            new_codes = FrozenList(new_codes)

        if verify_integrity:
            self._verify_integrity(codes=new_codes)

        self._codes = new_codes
        self._tuples = None
        self._reset_cache()

    def set_labels(self, labels, level=None, inplace=False,
                   verify_integrity=True):
        warnings.warn((".set_labels was deprecated in version 0.24.0. "
                       "Use .set_codes instead."),
                      FutureWarning, stacklevel=2)
        return self.set_codes(codes=labels, level=level, inplace=inplace,
                              verify_integrity=verify_integrity)

    @deprecate_kwarg(old_arg_name='labels', new_arg_name='codes')
    def set_codes(self, codes, level=None, inplace=False,
                  verify_integrity=True):
        """
        Set new codes on MultiIndex. Defaults to returning
        new index.

        .. versionadded:: 0.24.0

           New name for deprecated method `set_labels`.

        Parameters
        ----------
        codes : sequence or list of sequence
            new codes to apply
        level : int, level name, or sequence of int/level names (default None)
            level(s) to set (None for all levels)
        inplace : bool
            if True, mutates in place
        verify_integrity : bool (default True)
            if True, checks that levels and codes are compatible

        Returns
        -------
        new index (of same type and class...etc)

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples([(1, u'one'), (1, u'two'),
                                            (2, u'one'), (2, u'two')],
                                            names=['foo', 'bar'])
        >>> idx.set_codes([[1,0,1,0], [0,0,1,1]])
        MultiIndex(levels=[[1, 2], [u'one', u'two']],
                   codes=[[1, 0, 1, 0], [0, 0, 1, 1]],
                   names=[u'foo', u'bar'])
        >>> idx.set_codes([1,0,1,0], level=0)
        MultiIndex(levels=[[1, 2], [u'one', u'two']],
                   codes=[[1, 0, 1, 0], [0, 1, 0, 1]],
                   names=[u'foo', u'bar'])
        >>> idx.set_codes([0,0,1,1], level='bar')
        MultiIndex(levels=[[1, 2], [u'one', u'two']],
                   codes=[[0, 0, 1, 1], [0, 0, 1, 1]],
                   names=[u'foo', u'bar'])
        >>> idx.set_codes([[1,0,1,0], [0,0,1,1]], level=[0,1])
        MultiIndex(levels=[[1, 2], [u'one', u'two']],
                   codes=[[1, 0, 1, 0], [0, 0, 1, 1]],
                   names=[u'foo', u'bar'])
        """
        if level is not None and not is_list_like(level):
            if not is_list_like(codes):
                raise TypeError("Codes must be list-like")
            if is_list_like(codes[0]):
                raise TypeError("Codes must be list-like")
            level = [level]
            codes = [codes]
        elif level is None or is_list_like(level):
            if not is_list_like(codes) or not is_list_like(codes[0]):
                raise TypeError("Codes must be list of lists-like")

        if inplace:
            idx = self
        else:
            idx = self._shallow_copy()
        idx._reset_identity()
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        if not inplace:
            return idx

    @deprecate_kwarg(old_arg_name='labels', new_arg_name='codes')
    def copy(self, names=None, dtype=None, levels=None, codes=None,
             deep=False, _set_identity=False, **kwargs):
        """
        Make a copy of this object. Names, dtype, levels and codes can be
        passed and will be set on new copy.

        Parameters
        ----------
        names : sequence, optional
        dtype : numpy dtype or pandas type, optional
        levels : sequence, optional
        codes : sequence, optional

        Returns
        -------
        copy : MultiIndex

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.
        This could be potentially expensive on large MultiIndex objects.
        """
        name = kwargs.get('name')
        names = self._validate_names(name=name, names=names, deep=deep)

        if deep:
            from copy import deepcopy
            if levels is None:
                levels = deepcopy(self.levels)
            if codes is None:
                codes = deepcopy(self.codes)
        else:
            if levels is None:
                levels = self.levels
            if codes is None:
                codes = self.codes
        return MultiIndex(levels=levels, codes=codes, names=names,
                          sortorder=self.sortorder, verify_integrity=False,
                          _set_identity=_set_identity)

    def __array__(self, dtype=None):
        """ the array interface, return my values """
        return self.values

    def view(self, cls=None):
        """ this is defined as a copy with the same identity """
        result = self.copy()
        result._id = self._id
        return result

    def _shallow_copy_with_infer(self, values, **kwargs):
        # On equal MultiIndexes the difference is empty.
        # Therefore, an empty MultiIndex is returned GH13490
        if len(values) == 0:
            return MultiIndex(levels=[[] for _ in range(self.nlevels)],
                              codes=[[] for _ in range(self.nlevels)],
                              **kwargs)
        return self._shallow_copy(values, **kwargs)

    @Appender(_index_shared_docs['contains'] % _index_doc_kwargs)
    def __contains__(self, key):
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError):
            return False

    contains = __contains__

    @Appender(_index_shared_docs['_shallow_copy'])
    def _shallow_copy(self, values=None, **kwargs):
        if values is not None:
            names = kwargs.pop('names', kwargs.pop('name', self.names))
            # discards freq
            kwargs.pop('freq', None)
            return MultiIndex.from_tuples(values, names=names, **kwargs)
        return self.view()

    @cache_readonly
    def dtype(self):
        return np.dtype('O')

    def _is_memory_usage_qualified(self):
        """ return a boolean if we need a qualified .info display """
        def f(l):
            return 'mixed' in l or 'string' in l or 'unicode' in l
        return any(f(l) for l in self._inferred_type_levels)

    @Appender(Index.memory_usage.__doc__)
    def memory_usage(self, deep=False):
        # we are overwriting our base class to avoid
        # computing .values here which could materialize
        # a tuple representation uncessarily
        return self._nbytes(deep)

    @cache_readonly
    def nbytes(self):
        """ return the number of bytes in the underlying data """
        return self._nbytes(False)

    def _nbytes(self, deep=False):
        """
        return the number of bytes in the underlying data
        deeply introspect the level data if deep=True

        include the engine hashtable

        *this is in internal routine*

        """

        # for implementations with no useful getsizeof (PyPy)
        objsize = 24

        level_nbytes = sum(i.memory_usage(deep=deep) for i in self.levels)
        label_nbytes = sum(i.nbytes for i in self.codes)
        names_nbytes = sum(getsizeof(i, objsize) for i in self.names)
        result = level_nbytes + label_nbytes + names_nbytes

        # include our engine hashtable
        result += self._engine.sizeof(deep=deep)
        return result

    # --------------------------------------------------------------------
    # Rendering Methods

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        attrs = [
            ('levels', ibase.default_pprint(self._levels,
                                            max_seq_items=False)),
            ('codes', ibase.default_pprint(self._codes,
                                           max_seq_items=False))]
        if com._any_not_none(*self.names):
            attrs.append(('names', ibase.default_pprint(self.names)))
        if self.sortorder is not None:
            attrs.append(('sortorder', ibase.default_pprint(self.sortorder)))
        return attrs

    def _format_space(self):
        return "\n%s" % (' ' * (len(self.__class__.__name__) + 1))

    def _format_data(self, name=None):
        # we are formatting thru the attributes
        return None

    def _format_native_types(self, na_rep='nan', **kwargs):
        new_levels = []
        new_codes = []

        # go through the levels and format them
        for level, level_codes in zip(self.levels, self.codes):
            level = level._format_native_types(na_rep=na_rep, **kwargs)
            # add nan values, if there are any
            mask = (level_codes == -1)
            if mask.any():
                nan_index = len(level)
                level = np.append(level, na_rep)
                level_codes = level_codes.values()
                level_codes[mask] = nan_index
            new_levels.append(level)
            new_codes.append(level_codes)

        if len(new_levels) == 1:
            return Index(new_levels[0])._format_native_types()
        else:
            # reconstruct the multi-index
            mi = MultiIndex(levels=new_levels, codes=new_codes,
                            names=self.names, sortorder=self.sortorder,
                            verify_integrity=False)
            return mi.values

    def format(self, space=2, sparsify=None, adjoin=True, names=False,
               na_rep=None, formatter=None):
        if len(self) == 0:
            return []

        stringified_levels = []
        for lev, level_codes in zip(self.levels, self.codes):
            na = na_rep if na_rep is not None else _get_na_rep(lev.dtype.type)

            if len(lev) > 0:

                formatted = lev.take(level_codes).format(formatter=formatter)

                # we have some NA
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()

            else:
                # weird all NA case
                formatted = [pprint_thing(na if isna(x) else x,
                                          escape_chars=('\t', '\r', '\n'))
                             for x in algos.take_1d(lev._values, level_codes)]
            stringified_levels.append(formatted)

        result_levels = []
        for lev, name in zip(stringified_levels, self.names):
            level = []

            if names:
                level.append(pprint_thing(name,
                                          escape_chars=('\t', '\r', '\n'))
                             if name is not None else '')

            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)

        if sparsify is None:
            sparsify = get_option("display.multi_sparse")

        if sparsify:
            sentinel = ''
            # GH3547
            # use value of sparsify as sentinel,  unless it's an obvious
            # "Truthey" value
            if sparsify not in [True, 1]:
                sentinel = sparsify
            # little bit of a kludge job for #1217
            result_levels = _sparsify(result_levels, start=int(names),
                                      sentinel=sentinel)

        if adjoin:
            from pandas.io.formats.format import _get_adjustment
            adj = _get_adjustment()
            return adj.adjoin(space, *result_levels).split('\n')
        else:
            return result_levels

    # --------------------------------------------------------------------

    def __len__(self):
        return len(self.codes[0])

    def _get_names(self):
        return FrozenList(level.name for level in self.levels)

    def _set_names(self, names, level=None, validate=True):
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None
        validate : boolean, default True
            validate that the names match level lengths

        Raises
        ------
        TypeError if each name is not hashable.

        Notes
        -----
        sets names on levels. WARNING: mutates!

        Note that you generally want to set this *after* changing levels, so
        that it only acts on copies
        """
        # GH 15110
        # Don't allow a single string for names in a MultiIndex
        if names is not None and not is_list_like(names):
            raise ValueError('Names should be list-like for a MultiIndex')
        names = list(names)

        if validate and level is not None and len(names) != len(level):
            raise ValueError('Length of names must match length of level.')
        if validate and level is None and len(names) != self.nlevels:
            raise ValueError('Length of names must match number of levels in '
                             'MultiIndex.')

        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._get_level_number(l) for l in level]

        # set the name
        for l, name in zip(level, names):
            if name is not None:
                # GH 20527
                # All items in 'names' need to be hashable:
                if not is_hashable(name):
                    raise TypeError('{}.name must be a hashable type'
                                    .format(self.__class__.__name__))
            self.levels[l].rename(name, inplace=True)

    names = property(fset=_set_names, fget=_get_names,
                     doc="Names of levels in MultiIndex")

    @Appender(_index_shared_docs['_get_grouper_for_level'])
    def _get_grouper_for_level(self, mapper, level):
        indexer = self.codes[level]
        level_index = self.levels[level]

        if mapper is not None:
            # Handle group mapping function and return
            level_values = self.levels[level].take(indexer)
            grouper = level_values.map(mapper)
            return grouper, None, None

        codes, uniques = algos.factorize(indexer, sort=True)

        if len(uniques) > 0 and uniques[0] == -1:
            # Handle NAs
            mask = indexer != -1
            ok_codes, uniques = algos.factorize(indexer[mask], sort=True)

            codes = np.empty(len(indexer), dtype=indexer.dtype)
            codes[mask] = ok_codes
            codes[~mask] = -1

        if len(uniques) < len(level_index):
            # Remove unobserved levels from level_index
            level_index = level_index.take(uniques)

        grouper = level_index.take(codes)

        return grouper, codes, level_index

    @property
    def _constructor(self):
        return MultiIndex.from_tuples

    @cache_readonly
    def inferred_type(self):
        return 'mixed'

    def _get_level_number(self, level):
        count = self.names.count(level)
        if (count > 1) and not is_integer(level):
            raise ValueError('The name %s occurs multiple times, use a '
                             'level number' % level)
        try:
            level = self.names.index(level)
        except ValueError:
            if not is_integer(level):
                raise KeyError('Level %s not found' % str(level))
            elif level < 0:
                level += self.nlevels
                if level < 0:
                    orig_level = level - self.nlevels
                    raise IndexError('Too many levels: Index has only %d '
                                     'levels, %d is not a valid level number' %
                                     (self.nlevels, orig_level))
            # Note: levels are zero-based
            elif level >= self.nlevels:
                raise IndexError('Too many levels: Index has only %d levels, '
                                 'not %d' % (self.nlevels, level + 1))
        return level

    _tuples = None

    @cache_readonly
    def _engine(self):
        # Calculate the number of bits needed to represent labels in each
        # level, as log2 of their sizes (including -1 for NaN):
        sizes = np.ceil(np.log2([len(l) + 1 for l in self.levels]))

        # Sum bit counts, starting from the _right_....
        lev_bits = np.cumsum(sizes[::-1])[::-1]

        # ... in order to obtain offsets such that sorting the combination of
        # shifted codes (one for each level, resulting in a unique integer) is
        # equivalent to sorting lexicographically the codes themselves. Notice
        # that each level needs to be shifted by the number of bits needed to
        # represent the _previous_ ones:
        offsets = np.concatenate([lev_bits[1:], [0]]).astype('uint64')

        # Check the total number of bits needed for our representation:
        if lev_bits[0] > 64:
            # The levels would overflow a 64 bit uint - use Python integers:
            return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
        return MultiIndexUIntEngine(self.levels, self.codes, offsets)

    @property
    def values(self):
        if self._tuples is not None:
            return self._tuples

        values = []

        for i in range(self.nlevels):
            vals = self._get_level_values(i)
            if is_categorical_dtype(vals):
                vals = vals.get_values()
            if (isinstance(vals.dtype, (PandasExtensionDtype, ExtensionDtype))
                    or hasattr(vals, '_box_values')):
                vals = vals.astype(object)
            vals = np.array(vals, copy=False)
            values.append(vals)

        self._tuples = lib.fast_zip(values)
        return self._tuples

    @property
    def _has_complex_internals(self):
        # to disable groupby tricks
        return True

    @cache_readonly
    def is_monotonic_increasing(self):
        """
        return if the index is monotonic increasing (only equal or
        increasing) values.
        """

        # reversed() because lexsort() wants the most significant key last.
        values = [self._get_level_values(i).values
                  for i in reversed(range(len(self.levels)))]
        try:
            sort_order = np.lexsort(values)
            return Index(sort_order).is_monotonic
        except TypeError:

            # we have mixed types and np.lexsort is not happy
            return Index(self.values).is_monotonic

    @cache_readonly
    def is_monotonic_decreasing(self):
        """
        return if the index is monotonic decreasing (only equal or
        decreasing) values.
        """
        # monotonic decreasing if and only if reverse is monotonic increasing
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def _have_mixed_levels(self):
        """ return a boolean list indicated if we have mixed levels """
        return ['mixed' in l for l in self._inferred_type_levels]

    @cache_readonly
    def _inferred_type_levels(self):
        """ return a list of the inferred types, one for each level """
        return [i.inferred_type for i in self.levels]

    @cache_readonly
    def _hashed_values(self):
        """ return a uint64 ndarray of my hashed values """
        from pandas.core.util.hashing import hash_tuples
        return hash_tuples(self)

    def _hashed_indexing_key(self, key):
        """
        validate and return the hash for the provided key

        *this is internal for use for the cython routines*

        Parameters
        ----------
        key : string or tuple

        Returns
        -------
        np.uint64

        Notes
        -----
        we need to stringify if we have mixed levels

        """
        from pandas.core.util.hashing import hash_tuples, hash_tuple

        if not isinstance(key, tuple):
            return hash_tuples(key)

        if not len(key) == self.nlevels:
            raise KeyError

        def f(k, stringify):
            if stringify and not isinstance(k, compat.string_types):
                k = str(k)
            return k
        key = tuple(f(k, stringify)
                    for k, stringify in zip(key, self._have_mixed_levels))
        return hash_tuple(key)

    @Appender(Index.duplicated.__doc__)
    def duplicated(self, keep='first'):
        from pandas.core.sorting import get_group_index
        from pandas._libs.hashtable import duplicated_int64

        shape = map(len, self.levels)
        ids = get_group_index(self.codes, shape, sort=False, xnull=False)

        return duplicated_int64(ids, keep)

    def fillna(self, value=None, downcast=None):
        """
        fillna is not implemented for MultiIndex
        """
        raise NotImplementedError('isna is not defined for MultiIndex')

    @Appender(_index_shared_docs['dropna'])
    def dropna(self, how='any'):
        nans = [level_codes == -1 for level_codes in self.codes]
        if how == 'any':
            indexer = np.any(nans, axis=0)
        elif how == 'all':
            indexer = np.all(nans, axis=0)
        else:
            raise ValueError("invalid how option: {0}".format(how))

        new_codes = [level_codes[~indexer] for level_codes in self.codes]
        return self.copy(codes=new_codes, deep=True)

    def get_value(self, series, key):
        # somewhat broken encapsulation
        from pandas.core.indexing import maybe_droplevels

        # Label-based
        s = com.values_from_object(series)
        k = com.values_from_object(key)

        def _try_mi(k):
            # TODO: what if a level contains tuples??
            loc = self.get_loc(k)
            new_values = series._values[loc]
            new_index = self[loc]
            new_index = maybe_droplevels(new_index, k)
            return series._constructor(new_values, index=new_index,
                                       name=series.name).__finalize__(self)

        try:
            return self._engine.get_value(s, k)
        except KeyError as e1:
            try:
                return _try_mi(key)
            except KeyError:
                pass

            try:
                return libindex.get_value_at(s, k)
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

            # a Timestamp will raise a TypeError in a multi-index
            # rather than a KeyError, try it here
            # note that a string that 'looks' like a Timestamp will raise
            # a KeyError! (GH5725)
            if (isinstance(key, (datetime.datetime, np.datetime64)) or
                    (compat.PY3 and isinstance(key, compat.string_types))):
                try:
                    return _try_mi(key)
                except KeyError:
                    raise
                except (IndexError, ValueError, TypeError):
                    pass

                try:
                    return _try_mi(Timestamp(key))
                except (KeyError, TypeError,
                        IndexError, ValueError, tslibs.OutOfBoundsDatetime):
                    pass

            raise InvalidIndexError(key)

    def _get_level_values(self, level, unique=False):
        """
        Return vector of label values for requested level,
        equal to the length of the index

        **this is an internal method**

        Parameters
        ----------
        level : int level
        unique : bool, default False
            if True, drop duplicated values

        Returns
        -------
        values : ndarray
        """

        values = self.levels[level]
        level_codes = self.codes[level]
        if unique:
            level_codes = algos.unique(level_codes)
        filled = algos.take_1d(values._values, level_codes,
                               fill_value=values._na_value)
        values = values._shallow_copy(filled)
        return values

    def get_level_values(self, level):
        """
        Return vector of label values for requested level,
        equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        values : Index
            ``values`` is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Examples
        ---------

        Create a MultiIndex:

        >>> mi = pd.MultiIndex.from_arrays((list('abc'), list('def')))
        >>> mi.names = ['level_1', 'level_2']

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='level_1')
        >>> mi.get_level_values('level_2')
        Index(['d', 'e', 'f'], dtype='object', name='level_2')
        """
        level = self._get_level_number(level)
        values = self._get_level_values(level)
        return values

    @Appender(_index_shared_docs['index_unique'] % _index_doc_kwargs)
    def unique(self, level=None):

        if level is None:
            return super(MultiIndex, self).unique()
        else:
            level = self._get_level_number(level)
            return self._get_level_values(level=level, unique=True)

    def _to_safe_for_reshape(self):
        """ convert to object if we are a categorical """
        return self.set_levels([i._to_safe_for_reshape() for i in self.levels])

    def to_frame(self, index=True, name=None):
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        .. versionadded:: 0.24.0

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original MultiIndex.

        name : list / sequence of strings, optional
            The passed names should substitute index level names.

        Returns
        -------
        DataFrame : a DataFrame containing the original MultiIndex data.

        See Also
        --------
        DataFrame
        """

        from pandas import DataFrame
        if name is not None:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence "
                                "of column names.")

            if len(name) != len(self.levels):
                raise ValueError("'name' should have same length as "
                                 "number of levels on index.")
            idx_names = name
        else:
            idx_names = self.names

        # Guarantee resulting column order
        result = DataFrame(
            OrderedDict([
                ((level if lvlname is None else lvlname),
                 self._get_level_values(level))
                for lvlname, level in zip(idx_names, range(len(self.levels)))
            ]),
            copy=False
        )

        if index:
            result.index = self
        return result

    def to_hierarchical(self, n_repeat, n_shuffle=1):
        """
        Return a MultiIndex reshaped to conform to the
        shapes given by n_repeat and n_shuffle.

        .. deprecated:: 0.24.0

        Useful to replicate and rearrange a MultiIndex for combination
        with another Index with n_repeat items.

        Parameters
        ----------
        n_repeat : int
            Number of times to repeat the labels on self
        n_shuffle : int
            Controls the reordering of the labels. If the result is going
            to be an inner level in a MultiIndex, n_shuffle will need to be
            greater than one. The size of each label must divisible by
            n_shuffle.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_tuples([(1, u'one'), (1, u'two'),
                                            (2, u'one'), (2, u'two')])
        >>> idx.to_hierarchical(3)
        MultiIndex(levels=[[1, 2], [u'one', u'two']],
                   codes=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]])
        """
        levels = self.levels
        codes = [np.repeat(level_codes, n_repeat) for
                 level_codes in self.codes]
        # Assumes that each level_codes is divisible by n_shuffle
        codes = [x.reshape(n_shuffle, -1).ravel(order='F') for x in codes]
        names = self.names
        warnings.warn("Method .to_hierarchical is deprecated and will "
                      "be removed in a future version",
                      FutureWarning, stacklevel=2)
        return MultiIndex(levels=levels, codes=codes, names=names)

    def to_flat_index(self):
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        .. versionadded:: 0.24.0

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [['foo', 'bar'], ['baz', 'qux']],
        ...     names=['a', 'b'])
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
        return Index(self.values, tupleize_cols=False)

    @property
    def is_all_dates(self):
        return False

    def is_lexsorted(self):
        """
        Return True if the codes are lexicographically sorted
        """
        return self.lexsort_depth == self.nlevels

    @cache_readonly
    def lexsort_depth(self):
        if self.sortorder is not None:
            if self.sortorder == 0:
                return self.nlevels
            else:
                return 0

        int64_codes = [ensure_int64(level_codes) for level_codes in self.codes]
        for k in range(self.nlevels, 0, -1):
            if libalgos.is_lexsorted(int64_codes[:k]):
                return k

        return 0

    def _sort_levels_monotonic(self):
        """
        .. versionadded:: 0.20.0

        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------

        >>> i = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
                              codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> i
        MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]])

        >>> i.sort_monotonic()
        MultiIndex(levels=[['a', 'b'], ['aa', 'bb']],
                   codes=[[0, 0, 1, 1], [1, 0, 1, 0]])

        """

        if self.is_lexsorted() and self.is_monotonic:
            return self

        new_levels = []
        new_codes = []

        for lev, level_codes in zip(self.levels, self.codes):

            if not lev.is_monotonic:
                try:
                    # indexer to reorder the levels
                    indexer = lev.argsort()
                except TypeError:
                    pass
                else:
                    lev = lev.take(indexer)

                    # indexer to reorder the level codes
                    indexer = ensure_int64(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    level_codes = algos.take_1d(ri, level_codes)

            new_levels.append(lev)
            new_codes.append(level_codes)

        return MultiIndex(new_levels, new_codes,
                          names=self.names, sortorder=self.sortorder,
                          verify_integrity=False)

    def remove_unused_levels(self):
        """
        Create a new MultiIndex from the current that removes
        unused levels, meaning that they are not expressed in the labels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        .. versionadded:: 0.20.0

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> i = pd.MultiIndex.from_product([range(2), list('ab')])
        MultiIndex(levels=[[0, 1], ['a', 'b']],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]])

        >>> i[2:]
        MultiIndex(levels=[[0, 1], ['a', 'b']],
                   codes=[[1, 1], [0, 1]])

        The 0 from the first level is not represented
        and can be removed

        >>> i[2:].remove_unused_levels()
        MultiIndex(levels=[[1], ['a', 'b']],
                   codes=[[0, 0], [0, 1]])
        """

        new_levels = []
        new_codes = []

        changed = False
        for lev, level_codes in zip(self.levels, self.codes):

            # Since few levels are typically unused, bincount() is more
            # efficient than unique() - however it only accepts positive values
            # (and drops order):
            uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
            has_na = int(len(uniques) and (uniques[0] == -1))

            if len(uniques) != len(lev) + has_na:
                # We have unused levels
                changed = True

                # Recalculate uniques, now preserving order.
                # Can easily be cythonized by exploiting the already existing
                # "uniques" and stop parsing "level_codes" when all items
                # are found:
                uniques = algos.unique(level_codes)
                if has_na:
                    na_idx = np.where(uniques == -1)[0]
                    # Just ensure that -1 is in first position:
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]

                # codes get mapped from uniques to 0:len(uniques)
                # -1 (if present) is mapped to last position
                code_mapping = np.zeros(len(lev) + has_na)
                # ... and reassigned value -1:
                code_mapping[uniques] = np.arange(len(uniques)) - has_na

                level_codes = code_mapping[level_codes]

                # new levels are simple
                lev = lev.take(uniques[has_na:])

            new_levels.append(lev)
            new_codes.append(level_codes)

        result = self._shallow_copy()

        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)

        return result

    @property
    def nlevels(self):
        """Integer number of levels in this MultiIndex."""
        return len(self.levels)

    @property
    def levshape(self):
        """A tuple with the length of each level."""
        return tuple(len(x) for x in self.levels)

    def __reduce__(self):
        """Necessary for making this object picklable"""
        d = dict(levels=[lev for lev in self.levels],
                 codes=[level_codes for level_codes in self.codes],
                 sortorder=self.sortorder, names=list(self.names))
        return ibase._new_Index, (self.__class__, d), None

    def __setstate__(self, state):
        """Necessary for making this object picklable"""

        if isinstance(state, dict):
            levels = state.get('levels')
            codes = state.get('codes')
            sortorder = state.get('sortorder')
            names = state.get('names')

        elif isinstance(state, tuple):

            nd_state, own_state = state
            levels, codes, sortorder, names = own_state

        self._set_levels([Index(x) for x in levels], validate=False)
        self._set_codes(codes)
        self._set_names(names)
        self.sortorder = sortorder
        self._verify_integrity()
        self._reset_identity()

    def __getitem__(self, key):
        if is_scalar(key):
            key = com.cast_scalar_indexer(key)

            retval = []
            for lev, level_codes in zip(self.levels, self.codes):
                if level_codes[key] == -1:
                    retval.append(np.nan)
                else:
                    retval.append(lev[level_codes[key]])

            return tuple(retval)
        else:
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder
            else:
                # cannot be sure whether the result will be sorted
                sortorder = None

                if isinstance(key, Index):
                    key = np.asarray(key)

            new_codes = [level_codes[key] for level_codes in self.codes]

            return MultiIndex(levels=self.levels, codes=new_codes,
                              names=self.names, sortorder=sortorder,
                              verify_integrity=False)

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self, indices, axis=0, allow_fill=True,
             fill_value=None, **kwargs):
        nv.validate_take(tuple(), kwargs)
        indices = ensure_platform_int(indices)
        taken = self._assert_take_fillable(self.codes, indices,
                                           allow_fill=allow_fill,
                                           fill_value=fill_value,
                                           na_value=-1)
        return MultiIndex(levels=self.levels, codes=taken,
                          names=self.names, verify_integrity=False)

    def _assert_take_fillable(self, values, indices, allow_fill=True,
                              fill_value=None, na_value=None):
        """ Internal method to handle NA filling of take """
        # only fill if we are passing a non-None fill_value
        if allow_fill and fill_value is not None:
            if (indices < -1).any():
                msg = ('When allow_fill=True and fill_value is not None, '
                       'all indices must be >= -1')
                raise ValueError(msg)
            taken = [lab.take(indices) for lab in self.codes]
            mask = indices == -1
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label.values()
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked
        else:
            taken = [lab.take(indices) for lab in self.codes]
        return taken

    def append(self, other):
        """
        Append a collection of Index options together

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        appended : Index
        """
        if not isinstance(other, (list, tuple)):
            other = [other]

        if all((isinstance(o, MultiIndex) and o.nlevels >= self.nlevels)
               for o in other):
            arrays = []
            for i in range(self.nlevels):
                label = self._get_level_values(i)
                appended = [o._get_level_values(i) for o in other]
                arrays.append(label.append(appended))
            return MultiIndex.from_arrays(arrays, names=self.names)

        to_concat = (self.values, ) + tuple(k._values for k in other)
        new_tuples = np.concatenate(to_concat)

        # if all(isinstance(x, MultiIndex) for x in other):
        try:
            return MultiIndex.from_tuples(new_tuples, names=self.names)
        except (TypeError, IndexError):
            return Index(new_tuples)

    def argsort(self, *args, **kwargs):
        return self.values.argsort(*args, **kwargs)

    @Appender(_index_shared_docs['repeat'] % _index_doc_kwargs)
    def repeat(self, repeats, axis=None):
        nv.validate_repeat(tuple(), dict(axis=axis))
        return MultiIndex(levels=self.levels,
                          codes=[level_codes.view(np.ndarray).repeat(repeats)
                                 for level_codes in self.codes],
                          names=self.names, sortorder=self.sortorder,
                          verify_integrity=False)

    def where(self, cond, other=None):
        raise NotImplementedError(".where is not supported for "
                                  "MultiIndex operations")

    @deprecate_kwarg(old_arg_name='labels', new_arg_name='codes')
    def drop(self, codes, level=None, errors='raise'):
        """
        Make new MultiIndex with passed list of codes deleted

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples
        level : int or level name, default None

        Returns
        -------
        dropped : MultiIndex
        """
        if level is not None:
            return self._drop_from_level(codes, level)

        try:
            if not isinstance(codes, (np.ndarray, Index)):
                codes = com.index_labels_to_array(codes)
            indexer = self.get_indexer(codes)
            mask = indexer == -1
            if mask.any():
                if errors != 'ignore':
                    raise ValueError('codes %s not contained in axis' %
                                     codes[mask])
        except Exception:
            pass

        inds = []
        for level_codes in codes:
            try:
                loc = self.get_loc(level_codes)
                # get_loc returns either an integer, a slice, or a boolean
                # mask
                if isinstance(loc, int):
                    inds.append(loc)
                elif isinstance(loc, slice):
                    inds.extend(lrange(loc.start, loc.stop))
                elif com.is_bool_indexer(loc):
                    if self.lexsort_depth == 0:
                        warnings.warn('dropping on a non-lexsorted multi-index'
                                      ' without a level parameter may impact '
                                      'performance.',
                                      PerformanceWarning,
                                      stacklevel=3)
                    loc = loc.nonzero()[0]
                    inds.extend(loc)
                else:
                    msg = 'unsupported indexer of type {}'.format(type(loc))
                    raise AssertionError(msg)
            except KeyError:
                if errors != 'ignore':
                    raise

        return self.delete(inds)

    def _drop_from_level(self, codes, level):
        codes = com.index_labels_to_array(codes)
        i = self._get_level_number(level)
        index = self.levels[i]
        values = index.get_indexer(codes)

        mask = ~algos.isin(self.codes[i], values)

        return self[mask]

    def swaplevel(self, i=-2, j=-1):
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex

        .. versionchanged:: 0.18.1

           The indexes ``i`` and ``j`` are now optional, and default to
           the two innermost levels of the index.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        Dataframe.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi.swaplevel(0, 1)
        MultiIndex(levels=[['bb', 'aa'], ['a', 'b']],
                   codes=[[0, 1, 0, 1], [0, 0, 1, 1]])
        """
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)

        i = self._get_level_number(i)
        j = self._get_level_number(j)

        new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
        new_codes[i], new_codes[j] = new_codes[j], new_codes[i]
        new_names[i], new_names[j] = new_names[j], new_names[i]

        return MultiIndex(levels=new_levels, codes=new_codes,
                          names=new_names, verify_integrity=False)

    def reorder_levels(self, order):
        """
        Rearrange levels using input order. May not drop or duplicate levels

        Parameters
        ----------
        """
        order = [self._get_level_number(i) for i in order]
        if len(order) != self.nlevels:
            raise AssertionError('Length of order must be same as '
                                 'number of levels (%d), got %d' %
                                 (self.nlevels, len(order)))
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]

        return MultiIndex(levels=new_levels, codes=new_codes,
                          names=new_names, verify_integrity=False)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def _get_codes_for_sorting(self):
        """
        we categorizing our codes by using the
        available categories (all, not just observed)
        excluding any missing ones (-1); this is in preparation
        for sorting, where we need to disambiguate that -1 is not
        a valid valid
        """
        from pandas.core.arrays import Categorical

        def cats(level_codes):
            return np.arange(np.array(level_codes).max() + 1 if
                             len(level_codes) else 0,
                             dtype=level_codes.dtype)

        return [Categorical.from_codes(level_codes, cats(level_codes),
                                       ordered=True)
                for level_codes in self.codes]

    def sortlevel(self, level=0, ascending=True, sort_remaining=True):
        """
        Sort MultiIndex at the requested level. The result will respect the
        original ordering of the associated factor at that level.

        Parameters
        ----------
        level : list-like, int or str, default 0
            If a string is given, must be a name of the level
            If list-like must be names or ints of levels.
        ascending : boolean, default True
            False to sort in descending order
            Can also be a list to specify a directed ordering
        sort_remaining : sort by the remaining levels after level

        Returns
        -------
        sorted_index : pd.MultiIndex
            Resulting index
        indexer : np.ndarray
            Indices of output values in original index
        """
        from pandas.core.sorting import indexer_from_factorized

        if isinstance(level, (compat.string_types, int)):
            level = [level]
        level = [self._get_level_number(lev) for lev in level]
        sortorder = None

        # we have a directed ordering via ascending
        if isinstance(ascending, list):
            if not len(level) == len(ascending):
                raise ValueError("level must have same length as ascending")

            from pandas.core.sorting import lexsort_indexer
            indexer = lexsort_indexer([self.codes[lev] for lev in level],
                                      orders=ascending)

        # level ordering
        else:

            codes = list(self.codes)
            shape = list(self.levshape)

            # partition codes and shape
            primary = tuple(codes.pop(lev - i) for i, lev in enumerate(level))
            primshp = tuple(shape.pop(lev - i) for i, lev in enumerate(level))

            if sort_remaining:
                primary += primary + tuple(codes)
                primshp += primshp + tuple(shape)
            else:
                sortorder = level[0]

            indexer = indexer_from_factorized(primary, primshp,
                                              compress=False)

            if not ascending:
                indexer = indexer[::-1]

        indexer = ensure_platform_int(indexer)
        new_codes = [level_codes.take(indexer) for level_codes in self.codes]

        new_index = MultiIndex(codes=new_codes, levels=self.levels,
                               names=self.names, sortorder=sortorder,
                               verify_integrity=False)

        return new_index, indexer

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
        indexer, keyarr = super(MultiIndex, self)._convert_listlike_indexer(
            keyarr, kind=kind)

        # are we indexing a specific level
        if indexer is None and len(keyarr) and not isinstance(keyarr[0],
                                                              tuple):
            level = 0
            _, indexer = self.reindex(keyarr, level=level)

            # take all
            if indexer is None:
                indexer = np.arange(len(self))

            check = self.levels[0].get_indexer(keyarr)
            mask = check == -1
            if mask.any():
                raise KeyError('%s not in index' % keyarr[mask])

        return indexer, keyarr

    @Appender(_index_shared_docs['get_indexer'] % _index_doc_kwargs)
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        method = missing.clean_reindex_fill_method(method)
        target = ensure_index(target)

        # empty indexer
        if is_list_like(target) and not len(target):
            return ensure_platform_int(np.array([]))

        if not isinstance(target, MultiIndex):
            try:
                target = MultiIndex.from_tuples(target)
            except (TypeError, ValueError):

                # let's instead try with a straight Index
                if method is None:
                    return Index(self.values).get_indexer(target,
                                                          method=method,
                                                          limit=limit,
                                                          tolerance=tolerance)

        if not self.is_unique:
            raise ValueError('Reindexing only valid with uniquely valued '
                             'Index objects')

        if method == 'pad' or method == 'backfill':
            if tolerance is not None:
                raise NotImplementedError("tolerance not implemented yet "
                                          'for MultiIndex')
            indexer = self._engine.get_indexer(target, method, limit)
        elif method == 'nearest':
            raise NotImplementedError("method='nearest' not implemented yet "
                                      'for MultiIndex; see GitHub issue 9365')
        else:
            indexer = self._engine.get_indexer(target)

        return ensure_platform_int(indexer)

    @Appender(_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target):
        return super(MultiIndex, self).get_indexer_non_unique(target)

    def reindex(self, target, method=None, level=None, limit=None,
                tolerance=None):
        """
        Create index with target's values (move/add/delete values as necessary)

        Returns
        -------
        new_index : pd.MultiIndex
            Resulting index
        indexer : np.ndarray or None
            Indices of output values in original index

        """
        # GH6552: preserve names when reindexing to non-named target
        # (i.e. neither Index nor Series).
        preserve_names = not hasattr(target, 'names')

        if level is not None:
            if method is not None:
                raise TypeError('Fill method not supported if level passed')

            # GH7774: preserve dtype/tz if target is empty and not an Index.
            # target may be an iterator
            target = ibase._ensure_has_len(target)
            if len(target) == 0 and not isinstance(target, Index):
                idx = self.levels[level]
                attrs = idx._get_attributes_dict()
                attrs.pop('freq', None)  # don't preserve freq
                target = type(idx)._simple_new(np.empty(0, dtype=idx.dtype),
                                               **attrs)
            else:
                target = ensure_index(target)
            target, indexer, _ = self._join_level(target, level, how='right',
                                                  return_indexers=True,
                                                  keep_order=False)
        else:
            target = ensure_index(target)
            if self.equals(target):
                indexer = None
            else:
                if self.is_unique:
                    indexer = self.get_indexer(target, method=method,
                                               limit=limit,
                                               tolerance=tolerance)
                else:
                    raise ValueError("cannot handle a non-unique multi-index!")

        if not isinstance(target, MultiIndex):
            if indexer is None:
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                # hopefully?
                target = MultiIndex.from_tuples(target)

        if (preserve_names and target.nlevels == self.nlevels and
                target.names != self.names):
            target = target.copy(deep=False)
            target.names = self.names

        return target, indexer

    def get_slice_bound(self, label, side, kind):

        if not isinstance(label, tuple):
            label = label,
        return self._partial_tup_index(label, side=side)

    def slice_locs(self, start=None, end=None, step=None, kind=None):
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step
        kind : string, optional, defaults None

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbd'), list('deff')],
        ...                                names=['A', 'B'])

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start='b')
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start='b', end=('b', 'f'))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        # This function adds nothing to its parent implementation (the magic
        # happens in get_slice_bound method), but it adds meaningful doc.
        return super(MultiIndex, self).slice_locs(start, end, step, kind=kind)

    def _partial_tup_index(self, tup, side='left'):
        if len(tup) > self.lexsort_depth:
            raise UnsortedIndexError(
                'Key length (%d) was greater than MultiIndex'
                ' lexsort depth (%d)' %
                (len(tup), self.lexsort_depth))

        n = len(tup)
        start, end = 0, len(self)
        zipped = zip(tup, self.levels, self.codes)
        for k, (lab, lev, labs) in enumerate(zipped):
            section = labs[start:end]

            if lab not in lev:
                if not lev.is_type_compatible(lib.infer_dtype([lab],
                                                              skipna=False)):
                    raise TypeError('Level type mismatch: %s' % lab)

                # short circuit
                loc = lev.searchsorted(lab, side=side)
                if side == 'right' and loc >= 0:
                    loc -= 1
                return start + section.searchsorted(loc, side=side)

            idx = lev.get_loc(lab)
            if k < n - 1:
                end = start + section.searchsorted(idx, side='right')
                start = start + section.searchsorted(idx, side='left')
            else:
                return start + section.searchsorted(idx, side=side)

    def get_loc(self, key, method=None):
        """
        Get location for a label or a tuple of labels as an integer, slice or
        boolean mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)
        method : None

        Returns
        -------
        loc : int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        Examples
        ---------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_loc('b')
        slice(1, 3, None)

        >>> mi.get_loc(('b', 'e'))
        1

        Notes
        ------
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        if method is not None:
            raise NotImplementedError('only the default get_loc method is '
                                      'currently supported for MultiIndex')

        def _maybe_to_slice(loc):
            """convert integer indexer to boolean mask or slice if possible"""
            if not isinstance(loc, np.ndarray) or loc.dtype != 'int64':
                return loc

            loc = lib.maybe_indices_to_slice(loc, len(self))
            if isinstance(loc, slice):
                return loc

            mask = np.empty(len(self), dtype='bool')
            mask.fill(False)
            mask[loc] = True
            return mask

        if not isinstance(key, tuple):
            loc = self._get_level_indexer(key, level=0)
            return _maybe_to_slice(loc)

        keylen = len(key)
        if self.nlevels < keylen:
            raise KeyError('Key length ({0}) exceeds index depth ({1})'
                           ''.format(keylen, self.nlevels))

        if keylen == self.nlevels and self.is_unique:
            return self._engine.get_loc(key)

        # -- partial selection or non-unique index
        # break the key into 2 parts based on the lexsort_depth of the index;
        # the first part returns a continuous slice of the index; the 2nd part
        # needs linear search within the slice
        i = self.lexsort_depth
        lead_key, follow_key = key[:i], key[i:]
        start, stop = (self.slice_locs(lead_key, lead_key)
                       if lead_key else (0, len(self)))

        if start == stop:
            raise KeyError(key)

        if not follow_key:
            return slice(start, stop)

        warnings.warn('indexing past lexsort depth may impact performance.',
                      PerformanceWarning, stacklevel=10)

        loc = np.arange(start, stop, dtype='int64')

        for i, k in enumerate(follow_key, len(lead_key)):
            mask = self.codes[i][loc] == self.levels[i].get_loc(k)
            if not mask.all():
                loc = loc[mask]
            if not len(loc):
                raise KeyError(key)

        return (_maybe_to_slice(loc) if len(loc) != stop - start else
                slice(start, stop))

    def get_loc_level(self, key, level=0, drop_level=True):
        """
        Get both the location for the requested label(s) and the
        resulting sliced index.

        Parameters
        ----------
        key : label or sequence of labels
        level : int/level name or list thereof, optional
        drop_level : bool, default True
            if ``False``, the resulting index will not drop any level.

        Returns
        -------
        loc : A 2-tuple where the elements are:
              Element 0: int, slice object or boolean array
              Element 1: The resulting sliced multiindex/index. If the key
              contains all levels, this will be ``None``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')],
        ...                                names=['A', 'B'])

        >>> mi.get_loc_level('b')
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level('e', level='B')
        (array([False,  True, False], dtype=bool),
        Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(['b', 'e'])
        (1, None)

        See Also
        ---------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """

        def maybe_droplevels(indexer, levels, drop_level):
            if not drop_level:
                return self[indexer]
            # kludgearound
            orig_index = new_index = self[indexer]
            levels = [self._get_level_number(i) for i in levels]
            for i in sorted(levels, reverse=True):
                try:
                    new_index = new_index.droplevel(i)
                except ValueError:

                    # no dropping here
                    return orig_index
            return new_index

        if isinstance(level, (tuple, list)):
            if len(key) != len(level):
                raise AssertionError('Key for location must have same '
                                     'length as number of levels')
            result = None
            for lev, k in zip(level, key):
                loc, new_index = self.get_loc_level(k, level=lev)
                if isinstance(loc, slice):
                    mask = np.zeros(len(self), dtype=bool)
                    mask[loc] = True
                    loc = mask

                result = loc if result is None else result & loc

            return result, maybe_droplevels(result, level, drop_level)

        level = self._get_level_number(level)

        # kludge for #1796
        if isinstance(key, list):
            key = tuple(key)

        if isinstance(key, tuple) and level == 0:

            try:
                if key in self.levels[0]:
                    indexer = self._get_level_indexer(key, level=level)
                    new_index = maybe_droplevels(indexer, [0], drop_level)
                    return indexer, new_index
            except TypeError:
                pass

            if not any(isinstance(k, slice) for k in key):

                # partial selection
                # optionally get indexer to avoid re-calculation
                def partial_selection(key, indexer=None):
                    if indexer is None:
                        indexer = self.get_loc(key)
                    ilevels = [i for i in range(len(key))
                               if key[i] != slice(None, None)]
                    return indexer, maybe_droplevels(indexer, ilevels,
                                                     drop_level)

                if len(key) == self.nlevels and self.is_unique:
                    # Complete key in unique index -> standard get_loc
                    return (self._engine.get_loc(key), None)
                else:
                    return partial_selection(key)
            else:
                indexer = None
                for i, k in enumerate(key):
                    if not isinstance(k, slice):
                        k = self._get_level_indexer(k, level=i)
                        if isinstance(k, slice):
                            # everything
                            if k.start == 0 and k.stop == len(self):
                                k = slice(None, None)
                        else:
                            k_index = k

                    if isinstance(k, slice):
                        if k == slice(None, None):
                            continue
                        else:
                            raise TypeError(key)

                    if indexer is None:
                        indexer = k_index
                    else:  # pragma: no cover
                        indexer &= k_index
                if indexer is None:
                    indexer = slice(None, None)
                ilevels = [i for i in range(len(key))
                           if key[i] != slice(None, None)]
                return indexer, maybe_droplevels(indexer, ilevels, drop_level)
        else:
            indexer = self._get_level_indexer(key, level=level)
            return indexer, maybe_droplevels(indexer, [level], drop_level)

    def _get_level_indexer(self, key, level=0, indexer=None):
        # return an indexer, boolean array or a slice showing where the key is
        # in the totality of values
        # if the indexer is provided, then use this

        level_index = self.levels[level]
        level_codes = self.codes[level]

        def convert_indexer(start, stop, step, indexer=indexer,
                            codes=level_codes):
            # given the inputs and the codes/indexer, compute an indexer set
            # if we have a provided indexer, then this need not consider
            # the entire labels set

            r = np.arange(start, stop, step)
            if indexer is not None and len(indexer) != len(codes):

                # we have an indexer which maps the locations in the labels
                # that we have already selected (and is not an indexer for the
                # entire set) otherwise this is wasteful so we only need to
                # examine locations that are in this set the only magic here is
                # that the result are the mappings to the set that we have
                # selected
                from pandas import Series
                mapper = Series(indexer)
                indexer = codes.take(ensure_platform_int(indexer))
                result = Series(Index(indexer).isin(r).nonzero()[0])
                m = result.map(mapper)._ndarray_values

            else:
                m = np.zeros(len(codes), dtype=bool)
                m[np.in1d(codes, r,
                          assume_unique=Index(codes).is_unique)] = True

            return m

        if isinstance(key, slice):
            # handle a slice, returnig a slice if we can
            # otherwise a boolean indexer

            try:
                if key.start is not None:
                    start = level_index.get_loc(key.start)
                else:
                    start = 0
                if key.stop is not None:
                    stop = level_index.get_loc(key.stop)
                else:
                    stop = len(level_index) - 1
                step = key.step
            except KeyError:

                # we have a partial slice (like looking up a partial date
                # string)
                start = stop = level_index.slice_indexer(key.start, key.stop,
                                                         key.step, kind='loc')
                step = start.step

            if isinstance(start, slice) or isinstance(stop, slice):
                # we have a slice for start and/or stop
                # a partial date slicer on a DatetimeIndex generates a slice
                # note that the stop ALREADY includes the stopped point (if
                # it was a string sliced)
                return convert_indexer(start.start, stop.stop, step)

            elif level > 0 or self.lexsort_depth == 0 or step is not None:
                # need to have like semantics here to right
                # searching as when we are using a slice
                # so include the stop+1 (so we include stop)
                return convert_indexer(start, stop + 1, step)
            else:
                # sorted, so can return slice object -> view
                i = level_codes.searchsorted(start, side='left')
                j = level_codes.searchsorted(stop, side='right')
                return slice(i, j, step)

        else:

            code = level_index.get_loc(key)

            if level > 0 or self.lexsort_depth == 0:
                # Desired level is not sorted
                locs = np.array(level_codes == code, dtype=bool, copy=False)
                if not locs.any():
                    # The label is present in self.levels[level] but unused:
                    raise KeyError(key)
                return locs

            i = level_codes.searchsorted(code, side='left')
            j = level_codes.searchsorted(code, side='right')
            if i == j:
                # The label is present in self.levels[level] but unused:
                raise KeyError(key)
            return slice(i, j)

    def get_locs(self, seq):
        """
        Get location for a given label/slice/list/mask or a sequence of such as
        an array of integers.

        Parameters
        ----------
        seq : label/slice/list/mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        locs : array of integers suitable for passing to iloc

        Examples
        ---------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_locs('b')
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ['e', 'f']])
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice('e', 'f')])
        array([2], dtype=int64)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        """
        from .numeric import Int64Index

        # must be lexsorted to at least as many levels
        true_slices = [i for (i, s) in enumerate(com.is_true_slices(seq)) if s]
        if true_slices and true_slices[-1] >= self.lexsort_depth:
            raise UnsortedIndexError('MultiIndex slicing requires the index '
                                     'to be lexsorted: slicing on levels {0}, '
                                     'lexsort depth {1}'
                                     .format(true_slices, self.lexsort_depth))
        # indexer
        # this is the list of all values that we want to select
        n = len(self)
        indexer = None

        def _convert_to_indexer(r):
            # return an indexer
            if isinstance(r, slice):
                m = np.zeros(n, dtype=bool)
                m[r] = True
                r = m.nonzero()[0]
            elif com.is_bool_indexer(r):
                if len(r) != n:
                    raise ValueError("cannot index with a boolean indexer "
                                     "that is not the same length as the "
                                     "index")
                r = r.nonzero()[0]
            return Int64Index(r)

        def _update_indexer(idxr, indexer=indexer):
            if indexer is None:
                indexer = Index(np.arange(n))
            if idxr is None:
                return indexer
            return indexer & idxr

        for i, k in enumerate(seq):

            if com.is_bool_indexer(k):
                # a boolean indexer, must be the same length!
                k = np.asarray(k)
                indexer = _update_indexer(_convert_to_indexer(k),
                                          indexer=indexer)

            elif is_list_like(k):
                # a collection of labels to include from this level (these
                # are or'd)
                indexers = None
                for x in k:
                    try:
                        idxrs = _convert_to_indexer(
                            self._get_level_indexer(x, level=i,
                                                    indexer=indexer))
                        indexers = (idxrs if indexers is None
                                    else indexers | idxrs)
                    except KeyError:

                        # ignore not founds
                        continue

                if indexers is not None:
                    indexer = _update_indexer(indexers, indexer=indexer)
                else:
                    # no matches we are done
                    return Int64Index([])._ndarray_values

            elif com.is_null_slice(k):
                # empty slice
                indexer = _update_indexer(None, indexer=indexer)

            elif isinstance(k, slice):

                # a slice, include BOTH of the labels
                indexer = _update_indexer(_convert_to_indexer(
                    self._get_level_indexer(k, level=i, indexer=indexer)),
                    indexer=indexer)
            else:
                # a single label
                indexer = _update_indexer(_convert_to_indexer(
                    self.get_loc_level(k, level=i, drop_level=False)[0]),
                    indexer=indexer)

        # empty indexer
        if indexer is None:
            return Int64Index([])._ndarray_values
        return indexer._ndarray_values

    def truncate(self, before=None, after=None):
        """
        Slice index between two labels / tuples, return new MultiIndex

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start
        after : label or tuple, can be partial. Default None
            None defaults to end

        Returns
        -------
        truncated : MultiIndex
        """
        if after and before and after < before:
            raise ValueError('after < before')

        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)

        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]

        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = new_codes[0] - i

        return MultiIndex(levels=new_levels, codes=new_codes,
                          verify_integrity=False)

    def equals(self, other):
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        if not isinstance(other, MultiIndex):
            other_vals = com.values_from_object(ensure_index(other))
            return array_equivalent(self._ndarray_values, other_vals)

        if self.nlevels != other.nlevels:
            return False

        if len(self) != len(other):
            return False

        for i in range(self.nlevels):
            self_codes = self.codes[i]
            self_codes = self_codes[self_codes != -1]
            self_values = algos.take_nd(np.asarray(self.levels[i]._values),
                                        self_codes, allow_fill=False)

            other_codes = other.codes[i]
            other_codes = other_codes[other_codes != -1]
            other_values = algos.take_nd(
                np.asarray(other.levels[i]._values),
                other_codes, allow_fill=False)

            # since we use NaT both datetime64 and timedelta64
            # we can have a situation where a level is typed say
            # timedelta64 in self (IOW it has other values than NaT)
            # but types datetime64 in other (where its all NaT)
            # but these are equivalent
            if len(self_values) == 0 and len(other_values) == 0:
                continue

            if not array_equivalent(self_values, other_values):
                return False

        return True

    def equal_levels(self, other):
        """
        Return True if the levels of both MultiIndex objects are the same

        """
        if self.nlevels != other.nlevels:
            return False

        for i in range(self.nlevels):
            if not self.levels[i].equals(other.levels[i]):
                return False
        return True

    def union(self, other, sort=None):
        """
        Form the union of two MultiIndex objects

        Parameters
        ----------
        other : MultiIndex or array / Index of tuples
        sort : False or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` has length 0.
              3. Some values in `self` or `other` cannot be compared.
                 A RuntimeWarning is issued in this case.

            * False : do not sort the result.

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default value from ``True`` to ``None``
               (without change in behaviour).

        Returns
        -------
        Index

        >>> index.union(index2)
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_names = self._convert_can_do_setop(other)

        if len(other) == 0 or self.equals(other):
            return self

        # TODO: Index.union returns other when `len(self)` is 0.

        uniq_tuples = lib.fast_unique_multiple([self._ndarray_values,
                                                other._ndarray_values],
                                               sort=sort)

        return MultiIndex.from_arrays(lzip(*uniq_tuples), sortorder=0,
                                      names=result_names)

    def intersection(self, other, sort=False):
        """
        Form the intersection of two MultiIndex objects.

        Parameters
        ----------
        other : MultiIndex or array / Index of tuples
        sort : False or None, default False
            Sort the resulting MultiIndex if possible

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default from ``True`` to ``False``, to match
               behaviour from before 0.24.0

        Returns
        -------
        Index
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_names = self._convert_can_do_setop(other)

        if self.equals(other):
            return self

        self_tuples = self._ndarray_values
        other_tuples = other._ndarray_values
        uniq_tuples = set(self_tuples) & set(other_tuples)

        if sort is None:
            uniq_tuples = sorted(uniq_tuples)

        if len(uniq_tuples) == 0:
            return MultiIndex(levels=self.levels,
                              codes=[[]] * self.nlevels,
                              names=result_names, verify_integrity=False)
        else:
            return MultiIndex.from_arrays(lzip(*uniq_tuples), sortorder=0,
                                          names=result_names)

    def difference(self, other, sort=None):
        """
        Compute set difference of two MultiIndex objects

        Parameters
        ----------
        other : MultiIndex
        sort : False or None, default None
            Sort the resulting MultiIndex if possible

            .. versionadded:: 0.24.0

            .. versionchanged:: 0.24.1

               Changed the default value from ``True`` to ``None``
               (without change in behaviour).

        Returns
        -------
        diff : MultiIndex
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_names = self._convert_can_do_setop(other)

        if len(other) == 0:
            return self

        if self.equals(other):
            return MultiIndex(levels=self.levels,
                              codes=[[]] * self.nlevels,
                              names=result_names, verify_integrity=False)

        this = self._get_unique_index()

        indexer = this.get_indexer(other)
        indexer = indexer.take((indexer != -1).nonzero()[0])

        label_diff = np.setdiff1d(np.arange(this.size), indexer,
                                  assume_unique=True)
        difference = this.values.take(label_diff)
        if sort is None:
            difference = sorted(difference)

        if len(difference) == 0:
            return MultiIndex(levels=[[]] * self.nlevels,
                              codes=[[]] * self.nlevels,
                              names=result_names, verify_integrity=False)
        else:
            return MultiIndex.from_tuples(difference, sortorder=0,
                                          names=result_names)

    @Appender(_index_shared_docs['astype'])
    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_categorical_dtype(dtype):
            msg = '> 1 ndim Categorical are not supported at this time'
            raise NotImplementedError(msg)
        elif not is_object_dtype(dtype):
            msg = ('Setting {cls} dtype to anything other than object '
                   'is not supported').format(cls=self.__class__)
            raise TypeError(msg)
        elif copy is True:
            return self._shallow_copy()
        return self

    def _convert_can_do_setop(self, other):
        result_names = self.names

        if not hasattr(other, 'names'):
            if len(other) == 0:
                other = MultiIndex(levels=[[]] * self.nlevels,
                                   codes=[[]] * self.nlevels,
                                   verify_integrity=False)
            else:
                msg = 'other must be a MultiIndex or a list of tuples'
                try:
                    other = MultiIndex.from_tuples(other)
                except TypeError:
                    raise TypeError(msg)
        else:
            result_names = self.names if self.names == other.names else None
        return other, result_names

    def insert(self, loc, item):
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
        item : tuple
            Must be same length as number of levels in the MultiIndex

        Returns
        -------
        new_index : Index
        """
        # Pad the key with empty strings if lower levels of the key
        # aren't specified:
        if not isinstance(item, tuple):
            item = (item, ) + ('', ) * (self.nlevels - 1)
        elif len(item) != self.nlevels:
            raise ValueError('Item must have length equal to number of '
                             'levels.')

        new_levels = []
        new_codes = []
        for k, level, level_codes in zip(item, self.levels, self.codes):
            if k not in level:
                # have to insert into level
                # must insert at end otherwise you have to recompute all the
                # other codes
                lev_loc = len(level)
                level = level.insert(lev_loc, k)
            else:
                lev_loc = level.get_loc(k)

            new_levels.append(level)
            new_codes.append(np.insert(
                ensure_int64(level_codes), loc, lev_loc))

        return MultiIndex(levels=new_levels, codes=new_codes,
                          names=self.names, verify_integrity=False)

    def delete(self, loc):
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        return MultiIndex(levels=self.levels, codes=new_codes,
                          names=self.names, verify_integrity=False)

    def _wrap_joined_index(self, joined, other):
        names = self.names if self.names == other.names else None
        return MultiIndex.from_tuples(joined, names=names)

    @Appender(Index.isin.__doc__)
    def isin(self, values, level=None):
        if level is None:
            values = MultiIndex.from_tuples(values,
                                            names=self.names).values
            return algos.isin(self.values, values)
        else:
            num = self._get_level_number(level)
            levs = self.levels[num]
            level_codes = self.codes[num]

            sought_labels = levs.isin(values).nonzero()[0]
            if levs.size == 0:
                return np.zeros(len(level_codes), dtype=np.bool_)
            else:
                return np.lib.arraysetops.in1d(level_codes, sought_labels)


MultiIndex._add_numeric_methods_disabled()
MultiIndex._add_numeric_methods_add_sub_disabled()
MultiIndex._add_logical_methods_disabled()


def _sparsify(label_list, start=0, sentinel=''):
    pivoted = lzip(*label_list)
    k = len(label_list)

    result = pivoted[:start + 1]
    prev = pivoted[start]

    for cur in pivoted[start + 1:]:
        sparse_cur = []

        for i, (p, t) in enumerate(zip(prev, cur)):
            if i == k - 1:
                sparse_cur.append(t)
                result.append(sparse_cur)
                break

            if p == t:
                sparse_cur.append(sentinel)
            else:
                sparse_cur.extend(cur[i:])
                result.append(sparse_cur)
                break

        prev = cur

    return lzip(*result)


def _get_na_rep(dtype):
    return {np.datetime64: 'NaT', np.timedelta64: 'NaT'}.get(dtype, 'NaN')
