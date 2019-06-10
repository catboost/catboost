"""
concat routines
"""

import numpy as np

import pandas.core.dtypes.concat as _concat

from pandas import DataFrame, Index, MultiIndex, Series, compat
from pandas.core import common as com
from pandas.core.arrays.categorical import (
    _factorize_from_iterable, _factorize_from_iterables)
from pandas.core.generic import NDFrame
from pandas.core.index import (
    _all_indexes_same, _get_consensus_names, _get_objs_combined_axis,
    ensure_index)
import pandas.core.indexes.base as ibase
from pandas.core.internals import concatenate_block_managers

# ---------------------------------------------------------------------
# Concatenate DataFrame objects


def concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
           keys=None, levels=None, names=None, verify_integrity=False,
           sort=None, copy=True):
    """
    Concatenate pandas objects along a particular axis with optional set logic
    along the other axes.

    Can also add a layer of hierarchical indexing on the concatenation axis,
    which may be useful if the labels are the same (or overlapping) on
    the passed axis number.

    Parameters
    ----------
    objs : a sequence or mapping of Series, DataFrame, or Panel objects
        If a dict is passed, the sorted keys will be used as the `keys`
        argument, unless it is passed, in which case the values will be
        selected (see below). Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis(es)
    join_axes : list of Index objects
        Specific indexes to use for the other n - 1 axes instead of performing
        inner/outer set logic
    ignore_index : boolean, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys
    names : list, default None
        Names for the levels in the resulting hierarchical index
    verify_integrity : boolean, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation
    sort : boolean, default None
        Sort non-concatenation axis if it is not already aligned when `join`
        is 'outer'. The current default of sorting is deprecated and will
        change to not-sorting in a future version of pandas.

        Explicitly pass ``sort=True`` to silence the warning and sort.
        Explicitly pass ``sort=False`` to silence the warning and not sort.

        This has no effect when ``join='inner'``, which already preserves
        the order of the non-concatenation axis.

        .. versionadded:: 0.23.0

    copy : boolean, default True
        If False, do not copy data unnecessarily

    Returns
    -------
    concatenated : object, type of objs
        When concatenating all ``Series`` along the index (axis=0), a
        ``Series`` is returned. When ``objs`` contains at least one
        ``DataFrame``, a ``DataFrame`` is returned. When concatenating along
        the columns (axis=1), a ``DataFrame`` is returned.

    See Also
    --------
    Series.append
    DataFrame.append
    DataFrame.join
    DataFrame.merge

    Notes
    -----
    The keys, levels, and names arguments are all optional.

    A walkthrough of how this method fits in with other tools for combining
    pandas objects can be found `here
    <http://pandas.pydata.org/pandas-docs/stable/merging.html>`__.

    Examples
    --------
    Combine two ``Series``.

    >>> s1 = pd.Series(['a', 'b'])
    >>> s2 = pd.Series(['c', 'd'])
    >>> pd.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: object

    Clear the existing index and reset it in the result
    by setting the ``ignore_index`` option to ``True``.

    >>> pd.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: object

    Add a hierarchical index at the outermost level of
    the data with the ``keys`` option.

    >>> pd.concat([s1, s2], keys=['s1', 's2',])
    s1  0    a
        1    b
    s2  0    c
        1    d
    dtype: object

    Label the index keys you create with the ``names`` option.

    >>> pd.concat([s1, s2], keys=['s1', 's2'],
    ...           names=['Series name', 'Row ID'])
    Series name  Row ID
    s1           0         a
                 1         b
    s2           0         c
                 1         d
    dtype: object

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = pd.DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = pd.DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    >>> pd.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``NaN`` values.

    >>> df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> pd.concat([df1, df3], sort=False)
      letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> pd.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` objects horizontally along the x axis by
    passing in ``axis=1``.

    >>> df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])
    >>> pd.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    Prevent the result from including duplicate index values with the
    ``verify_integrity`` option.

    >>> df5 = pd.DataFrame([1], index=['a'])
    >>> df5
       0
    a  1
    >>> df6 = pd.DataFrame([2], index=['a'])
    >>> df6
       0
    a  2
    >>> pd.concat([df5, df6], verify_integrity=True)
    Traceback (most recent call last):
        ...
    ValueError: Indexes have overlapping values: ['a']
    """
    op = _Concatenator(objs, axis=axis, join_axes=join_axes,
                       ignore_index=ignore_index, join=join,
                       keys=keys, levels=levels, names=names,
                       verify_integrity=verify_integrity,
                       copy=copy, sort=sort)
    return op.get_result()


class _Concatenator(object):
    """
    Orchestrates a concatenation operation for BlockManagers
    """

    def __init__(self, objs, axis=0, join='outer', join_axes=None,
                 keys=None, levels=None, names=None,
                 ignore_index=False, verify_integrity=False, copy=True,
                 sort=False):
        if isinstance(objs, (NDFrame, compat.string_types)):
            raise TypeError('first argument must be an iterable of pandas '
                            'objects, you passed an object of type '
                            '"{name}"'.format(name=type(objs).__name__))

        if join == 'outer':
            self.intersect = False
        elif join == 'inner':
            self.intersect = True
        else:  # pragma: no cover
            raise ValueError('Only can inner (intersect) or outer (union) '
                             'join the other axis')

        if isinstance(objs, dict):
            if keys is None:
                keys = sorted(objs)
            objs = [objs[k] for k in keys]
        else:
            objs = list(objs)

        if len(objs) == 0:
            raise ValueError('No objects to concatenate')

        if keys is None:
            objs = list(com._not_none(*objs))
        else:
            # #1649
            clean_keys = []
            clean_objs = []
            for k, v in zip(keys, objs):
                if v is None:
                    continue
                clean_keys.append(k)
                clean_objs.append(v)
            objs = clean_objs
            name = getattr(keys, 'name', None)
            keys = Index(clean_keys, name=name)

        if len(objs) == 0:
            raise ValueError('All objects passed were None')

        # consolidate data & figure out what our result ndim is going to be
        ndims = set()
        for obj in objs:
            if not isinstance(obj, NDFrame):
                msg = ('cannot concatenate object of type "{0}";'
                       ' only pd.Series, pd.DataFrame, and pd.Panel'
                       ' (deprecated) objs are valid'.format(type(obj)))
                raise TypeError(msg)

            # consolidate
            obj._consolidate(inplace=True)
            ndims.add(obj.ndim)

        # get the sample
        # want the highest ndim that we have, and must be non-empty
        # unless all objs are empty
        sample = None
        if len(ndims) > 1:
            max_ndim = max(ndims)
            for obj in objs:
                if obj.ndim == max_ndim and np.sum(obj.shape):
                    sample = obj
                    break

        else:
            # filter out the empties if we have not multi-index possibilities
            # note to keep empty Series as it affect to result columns / name
            non_empties = [obj for obj in objs
                           if sum(obj.shape) > 0 or isinstance(obj, Series)]

            if (len(non_empties) and (keys is None and names is None and
                                      levels is None and
                                      join_axes is None and
                                      not self.intersect)):
                objs = non_empties
                sample = objs[0]

        if sample is None:
            sample = objs[0]
        self.objs = objs

        # Standardize axis parameter to int
        if isinstance(sample, Series):
            axis = DataFrame._get_axis_number(axis)
        else:
            axis = sample._get_axis_number(axis)

        # Need to flip BlockManager axis in the DataFrame special case
        self._is_frame = isinstance(sample, DataFrame)
        if self._is_frame:
            axis = 1 if axis == 0 else 0

        self._is_series = isinstance(sample, Series)
        if not 0 <= axis <= sample.ndim:
            raise AssertionError("axis must be between 0 and {ndim}, input was"
                                 " {axis}".format(ndim=sample.ndim, axis=axis))

        # if we have mixed ndims, then convert to highest ndim
        # creating column numbers as needed
        if len(ndims) > 1:
            current_column = 0
            max_ndim = sample.ndim
            self.objs, objs = [], self.objs
            for obj in objs:

                ndim = obj.ndim
                if ndim == max_ndim:
                    pass

                elif ndim != max_ndim - 1:
                    raise ValueError("cannot concatenate unaligned mixed "
                                     "dimensional NDFrame objects")

                else:
                    name = getattr(obj, 'name', None)
                    if ignore_index or name is None:
                        name = current_column
                        current_column += 1

                    # doing a row-wise concatenation so need everything
                    # to line up
                    if self._is_frame and axis == 1:
                        name = 0
                    obj = sample._constructor({name: obj})

                self.objs.append(obj)

        # note: this is the BlockManager axis (since DataFrame is transposed)
        self.axis = axis
        self.join_axes = join_axes
        self.keys = keys
        self.names = names or getattr(keys, 'names', None)
        self.levels = levels
        self.sort = sort

        self.ignore_index = ignore_index
        self.verify_integrity = verify_integrity
        self.copy = copy

        self.new_axes = self._get_new_axes()

    def get_result(self):

        # series only
        if self._is_series:

            # stack blocks
            if self.axis == 0:
                name = com.consensus_name_attr(self.objs)

                mgr = self.objs[0]._data.concat([x._data for x in self.objs],
                                                self.new_axes)
                cons = _concat._get_series_result_type(mgr, self.objs)
                return cons(mgr, name=name).__finalize__(self, method='concat')

            # combine as columns in a frame
            else:
                data = dict(zip(range(len(self.objs)), self.objs))
                cons = _concat._get_series_result_type(data)

                index, columns = self.new_axes
                df = cons(data, index=index)
                df.columns = columns
                return df.__finalize__(self, method='concat')

        # combine block managers
        else:
            mgrs_indexers = []
            for obj in self.objs:
                mgr = obj._data
                indexers = {}
                for ax, new_labels in enumerate(self.new_axes):
                    if ax == self.axis:
                        # Suppress reindexing on concat axis
                        continue

                    obj_labels = mgr.axes[ax]
                    if not new_labels.equals(obj_labels):
                        indexers[ax] = obj_labels.reindex(new_labels)[1]

                mgrs_indexers.append((obj._data, indexers))

            new_data = concatenate_block_managers(
                mgrs_indexers, self.new_axes, concat_axis=self.axis,
                copy=self.copy)
            if not self.copy:
                new_data._consolidate_inplace()

            cons = _concat._get_frame_result_type(new_data, self.objs)
            return (cons._from_axes(new_data, self.new_axes)
                    .__finalize__(self, method='concat'))

    def _get_result_dim(self):
        if self._is_series and self.axis == 1:
            return 2
        else:
            return self.objs[0].ndim

    def _get_new_axes(self):
        ndim = self._get_result_dim()
        new_axes = [None] * ndim

        if self.join_axes is None:
            for i in range(ndim):
                if i == self.axis:
                    continue
                new_axes[i] = self._get_comb_axis(i)
        else:
            if len(self.join_axes) != ndim - 1:
                raise AssertionError("length of join_axes must be equal "
                                     "to {length}".format(length=ndim - 1))

            # ufff...
            indices = compat.lrange(ndim)
            indices.remove(self.axis)

            for i, ax in zip(indices, self.join_axes):
                new_axes[i] = ax

        new_axes[self.axis] = self._get_concat_axis()
        return new_axes

    def _get_comb_axis(self, i):
        data_axis = self.objs[0]._get_block_manager_axis(i)
        try:
            return _get_objs_combined_axis(self.objs, axis=data_axis,
                                           intersect=self.intersect,
                                           sort=self.sort)
        except IndexError:
            types = [type(x).__name__ for x in self.objs]
            raise TypeError("Cannot concatenate list of {types}"
                            .format(types=types))

    def _get_concat_axis(self):
        """
        Return index to be used along concatenation axis.
        """
        if self._is_series:
            if self.axis == 0:
                indexes = [x.index for x in self.objs]
            elif self.ignore_index:
                idx = ibase.default_index(len(self.objs))
                return idx
            elif self.keys is None:
                names = [None] * len(self.objs)
                num = 0
                has_names = False
                for i, x in enumerate(self.objs):
                    if not isinstance(x, Series):
                        raise TypeError("Cannot concatenate type 'Series' "
                                        "with object of type {type!r}"
                                        .format(type=type(x).__name__))
                    if x.name is not None:
                        names[i] = x.name
                        has_names = True
                    else:
                        names[i] = num
                        num += 1
                if has_names:
                    return Index(names)
                else:
                    return ibase.default_index(len(self.objs))
            else:
                return ensure_index(self.keys).set_names(self.names)
        else:
            indexes = [x._data.axes[self.axis] for x in self.objs]

        if self.ignore_index:
            idx = ibase.default_index(sum(len(i) for i in indexes))
            return idx

        if self.keys is None:
            concat_axis = _concat_indexes(indexes)
        else:
            concat_axis = _make_concat_multiindex(indexes, self.keys,
                                                  self.levels, self.names)

        self._maybe_check_integrity(concat_axis)

        return concat_axis

    def _maybe_check_integrity(self, concat_index):
        if self.verify_integrity:
            if not concat_index.is_unique:
                overlap = concat_index[concat_index.duplicated()].unique()
                raise ValueError('Indexes have overlapping values: '
                                 '{overlap!s}'.format(overlap=overlap))


def _concat_indexes(indexes):
    return indexes[0].append(indexes[1:])


def _make_concat_multiindex(indexes, keys, levels=None, names=None):

    if ((levels is None and isinstance(keys[0], tuple)) or
            (levels is not None and len(levels) > 1)):
        zipped = compat.lzip(*keys)
        if names is None:
            names = [None] * len(zipped)

        if levels is None:
            _, levels = _factorize_from_iterables(zipped)
        else:
            levels = [ensure_index(x) for x in levels]
    else:
        zipped = [keys]
        if names is None:
            names = [None]

        if levels is None:
            levels = [ensure_index(keys)]
        else:
            levels = [ensure_index(x) for x in levels]

    if not _all_indexes_same(indexes):
        codes_list = []

        # things are potentially different sizes, so compute the exact codes
        # for each level and pass those to MultiIndex.from_arrays

        for hlevel, level in zip(zipped, levels):
            to_concat = []
            for key, index in zip(hlevel, indexes):
                try:
                    i = level.get_loc(key)
                except KeyError:
                    raise ValueError('Key {key!s} not in level {level!s}'
                                     .format(key=key, level=level))

                to_concat.append(np.repeat(i, len(index)))
            codes_list.append(np.concatenate(to_concat))

        concat_index = _concat_indexes(indexes)

        # these go at the end
        if isinstance(concat_index, MultiIndex):
            levels.extend(concat_index.levels)
            codes_list.extend(concat_index.codes)
        else:
            codes, categories = _factorize_from_iterable(concat_index)
            levels.append(categories)
            codes_list.append(codes)

        if len(names) == len(levels):
            names = list(names)
        else:
            # make sure that all of the passed indices have the same nlevels
            if not len({idx.nlevels for idx in indexes}) == 1:
                raise AssertionError("Cannot concat indices that do"
                                     " not have the same number of levels")

            # also copies
            names = names + _get_consensus_names(indexes)

        return MultiIndex(levels=levels, codes=codes_list, names=names,
                          verify_integrity=False)

    new_index = indexes[0]
    n = len(new_index)
    kpieces = len(indexes)

    # also copies
    new_names = list(names)
    new_levels = list(levels)

    # construct codes
    new_codes = []

    # do something a bit more speedy

    for hlevel, level in zip(zipped, levels):
        hlevel = ensure_index(hlevel)
        mapped = level.get_indexer(hlevel)

        mask = mapped == -1
        if mask.any():
            raise ValueError('Values not found in passed level: {hlevel!s}'
                             .format(hlevel=hlevel[mask]))

        new_codes.append(np.repeat(mapped, n))

    if isinstance(new_index, MultiIndex):
        new_levels.extend(new_index.levels)
        new_codes.extend([np.tile(lab, kpieces) for lab in new_index.codes])
    else:
        new_levels.append(new_index)
        new_codes.append(np.tile(np.arange(n), kpieces))

    if len(new_names) < len(new_levels):
        new_names.extend(new_index.names)

    return MultiIndex(levels=new_levels, codes=new_codes, names=new_names,
                      verify_integrity=False)
