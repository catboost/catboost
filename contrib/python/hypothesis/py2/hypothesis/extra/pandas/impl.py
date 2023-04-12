# coding=utf-8
#
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2019 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# END HEADER

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from copy import copy

import attr
import numpy as np
import pandas

import hypothesis.extra.numpy as npst
import hypothesis.internal.conjecture.utils as cu
import hypothesis.strategies._internal.core as st
from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import abc, hrange
from hypothesis.internal.coverage import check, check_function
from hypothesis.internal.validation import (
    check_type,
    check_valid_interval,
    check_valid_size,
    try_convert,
)

try:
    from pandas.api.types import is_categorical_dtype
except ImportError:  # pragma: no cover

    def is_categorical_dtype(dt):
        if isinstance(dt, np.dtype):
            return False
        return dt == "category"


if False:
    from typing import Any, List, Union, Sequence, Set  # noqa
    from hypothesis.strategies._internal.strategies import Ex  # noqa


def dtype_for_elements_strategy(s):
    return st.shared(
        s.map(lambda x: pandas.Series([x]).dtype),
        key=("hypothesis.extra.pandas.dtype_for_elements_strategy", s),
    )


def infer_dtype_if_necessary(dtype, values, elements, draw):
    if dtype is None and not values:
        return draw(dtype_for_elements_strategy(elements))
    return dtype


@check_function
def elements_and_dtype(elements, dtype, source=None):

    if source is None:
        prefix = ""
    else:
        prefix = "%s." % (source,)

    if elements is not None:
        st.check_strategy(elements, "%selements" % (prefix,))
    else:
        with check("dtype is not None"):
            if dtype is None:
                raise InvalidArgument(
                    (
                        "At least one of %(prefix)selements or %(prefix)sdtype "
                        "must be provided."
                    )
                    % {"prefix": prefix}
                )

    with check("is_categorical_dtype"):
        if is_categorical_dtype(dtype):
            raise InvalidArgument(
                "%sdtype is categorical, which is currently unsupported" % (prefix,)
            )

    dtype = try_convert(np.dtype, dtype, "dtype")

    if elements is None:
        elements = npst.from_dtype(dtype)
    elif dtype is not None:

        def convert_element(value):
            name = "draw(%selements)" % (prefix,)
            try:
                return np.array([value], dtype=dtype)[0]
            except TypeError:
                raise InvalidArgument(
                    "Cannot convert %s=%r of type %s to dtype %s"
                    % (name, value, type(value).__name__, dtype.str)
                )
            except ValueError:
                raise InvalidArgument(
                    "Cannot convert %s=%r to type %s" % (name, value, dtype.str)
                )

        elements = elements.map(convert_element)
    assert elements is not None

    return elements, dtype


class ValueIndexStrategy(st.SearchStrategy):
    def __init__(self, elements, dtype, min_size, max_size, unique):
        super(ValueIndexStrategy, self).__init__()
        self.elements = elements
        self.dtype = dtype
        self.min_size = min_size
        self.max_size = max_size
        self.unique = unique

    def do_draw(self, data):
        result = []
        seen = set()

        iterator = cu.many(
            data,
            min_size=self.min_size,
            max_size=self.max_size,
            average_size=(self.min_size + self.max_size) / 2,
        )

        while iterator.more():
            elt = data.draw(self.elements)

            if self.unique:
                if elt in seen:
                    iterator.reject()
                    continue
                seen.add(elt)
            result.append(elt)

        dtype = infer_dtype_if_necessary(
            dtype=self.dtype, values=result, elements=self.elements, draw=data.draw
        )
        return pandas.Index(result, dtype=dtype, tupleize_cols=False)


DEFAULT_MAX_SIZE = 10


@st.cacheable
@st.defines_strategy
def range_indexes(min_size=0, max_size=None):
    # type: (int, int) -> st.SearchStrategy[pandas.RangeIndex]
    """Provides a strategy which generates an :class:`~pandas.Index` whose
    values are 0, 1, ..., n for some n.

    Arguments:

    * min_size is the smallest number of elements the index can have.
    * max_size is the largest number of elements the index can have. If None
      it will default to some suitable value based on min_size.
    """
    check_valid_size(min_size, "min_size")
    check_valid_size(max_size, "max_size")
    if max_size is None:
        max_size = min([min_size + DEFAULT_MAX_SIZE, 2 ** 63 - 1])
    check_valid_interval(min_size, max_size, "min_size", "max_size")
    return st.integers(min_size, max_size).map(pandas.RangeIndex)


@st.cacheable
@st.defines_strategy
def indexes(
    elements=None,  # type: st.SearchStrategy[Ex]
    dtype=None,  # type: Any
    min_size=0,  # type: int
    max_size=None,  # type: int
    unique=True,  # type: bool
):
    """Provides a strategy for producing a :class:`pandas.Index`.

    Arguments:

    * elements is a strategy which will be used to generate the individual
      values of the index. If None, it will be inferred from the dtype. Note:
      even if the elements strategy produces tuples, the generated value
      will not be a MultiIndex, but instead be a normal index whose elements
      are tuples.
    * dtype is the dtype of the resulting index. If None, it will be inferred
      from the elements strategy. At least one of dtype or elements must be
      provided.
    * min_size is the minimum number of elements in the index.
    * max_size is the maximum number of elements in the index. If None then it
      will default to a suitable small size. If you want larger indexes you
      should pass a max_size explicitly.
    * unique specifies whether all of the elements in the resulting index
      should be distinct.
    """
    check_valid_size(min_size, "min_size")
    check_valid_size(max_size, "max_size")
    check_valid_interval(min_size, max_size, "min_size", "max_size")
    check_type(bool, unique, "unique")

    elements, dtype = elements_and_dtype(elements, dtype)

    if max_size is None:
        max_size = min_size + DEFAULT_MAX_SIZE
    return ValueIndexStrategy(elements, dtype, min_size, max_size, unique)


@st.defines_strategy
def series(
    elements=None,  # type: st.SearchStrategy[Ex]
    dtype=None,  # type: Any
    index=None,  # type: st.SearchStrategy[Union[Sequence, pandas.Index]]
    fill=None,  # type: st.SearchStrategy[Ex]
    unique=False,  # type: bool
):
    # type: (...) -> st.SearchStrategy[pandas.Series]
    """Provides a strategy for producing a :class:`pandas.Series`.

    Arguments:

    * elements: a strategy that will be used to generate the individual
      values in the series. If None, we will attempt to infer a suitable
      default from the dtype.

    * dtype: the dtype of the resulting series and may be any value
      that can be passed to :class:`numpy.dtype`. If None, will use
      pandas's standard behaviour to infer it from the type of the elements
      values. Note that if the type of values that comes out of your
      elements strategy varies, then so will the resulting dtype of the
      series.

    * index: If not None, a strategy for generating indexes for the
      resulting Series. This can generate either :class:`pandas.Index`
      objects or any sequence of values (which will be passed to the
      Index constructor).

      You will probably find it most convenient to use the
      :func:`~hypothesis.extra.pandas.indexes` or
      :func:`~hypothesis.extra.pandas.range_indexes` function to produce
      values for this argument.

    Usage:

    .. code-block:: pycon

        >>> series(dtype=int).example()
        0   -2001747478
        1    1153062837
    """
    if index is None:
        index = range_indexes()
    else:
        st.check_strategy(index)

    elements, dtype = elements_and_dtype(elements, dtype)
    index_strategy = index

    @st.composite
    def result(draw):
        index = draw(index_strategy)

        if len(index) > 0:
            if dtype is not None:
                result_data = draw(
                    npst.arrays(
                        dtype=dtype,
                        elements=elements,
                        shape=len(index),
                        fill=fill,
                        unique=unique,
                    )
                )
            else:
                result_data = list(
                    draw(
                        npst.arrays(
                            dtype=object,
                            elements=elements,
                            shape=len(index),
                            fill=fill,
                            unique=unique,
                        )
                    )
                )

            return pandas.Series(result_data, index=index, dtype=dtype)
        else:
            return pandas.Series(
                (),
                index=index,
                dtype=dtype
                if dtype is not None
                else draw(dtype_for_elements_strategy(elements)),
            )

    return result()


@attr.s(slots=True)
class column(object):
    """Data object for describing a column in a DataFrame.

    Arguments:

    * name: the column name, or None to default to the column position. Must
      be hashable, but can otherwise be any value supported as a pandas column
      name.
    * elements: the strategy for generating values in this column, or None
      to infer it from the dtype.
    * dtype: the dtype of the column, or None to infer it from the element
      strategy. At least one of dtype or elements must be provided.
    * fill: A default value for elements of the column. See
      :func:`~hypothesis.extra.numpy.arrays` for a full explanation.
    * unique: If all values in this column should be distinct.
    """

    name = attr.ib(default=None)
    elements = attr.ib(default=None)
    dtype = attr.ib(default=None)
    fill = attr.ib(default=None)
    unique = attr.ib(default=False)


def columns(
    names_or_number,  # type: Union[int, Sequence[str]]
    dtype=None,  # type: Any
    elements=None,  # type: st.SearchStrategy[Ex]
    fill=None,  # type: st.SearchStrategy[Ex]
    unique=False,  # type: bool
):
    # type: (...) -> List[column]
    """A convenience function for producing a list of :class:`column` objects
    of the same general shape.

    The names_or_number argument is either a sequence of values, the
    elements of which will be used as the name for individual column
    objects, or a number, in which case that many unnamed columns will
    be created. All other arguments are passed through verbatim to
    create the columns.
    """
    if isinstance(names_or_number, (int, float)):
        names = [None] * names_or_number  # type: list
    else:
        names = list(names_or_number)
    return [
        column(name=n, dtype=dtype, elements=elements, fill=fill, unique=unique)
        for n in names
    ]


@st.defines_strategy
def data_frames(
    columns=None,  # type: Sequence[column]
    rows=None,  # type: st.SearchStrategy[Union[dict, Sequence[Any]]]
    index=None,  # type: st.SearchStrategy[Ex]
):
    # type: (...) -> st.SearchStrategy[pandas.DataFrame]
    """Provides a strategy for producing a :class:`pandas.DataFrame`.

    Arguments:

    * columns: An iterable of :class:`column` objects describing the shape
      of the generated DataFrame.

    * rows: A strategy for generating a row object. Should generate
      either dicts mapping column names to values or a sequence mapping
      column position to the value in that position (note that unlike the
      :class:`pandas.DataFrame` constructor, single values are not allowed
      here. Passing e.g. an integer is an error, even if there is only one
      column).

      At least one of rows and columns must be provided. If both are
      provided then the generated rows will be validated against the
      columns and an error will be raised if they don't match.

      Caveats on using rows:

      * In general you should prefer using columns to rows, and only use
        rows if the columns interface is insufficiently flexible to
        describe what you need - you will get better performance and
        example quality that way.
      * If you provide rows and not columns, then the shape and dtype of
        the resulting DataFrame may vary. e.g. if you have a mix of int
        and float in the values for one column in your row entries, the
        column will sometimes have an integral dtype and sometimes a float.

    * index: If not None, a strategy for generating indexes for the
      resulting DataFrame. This can generate either :class:`pandas.Index`
      objects or any sequence of values (which will be passed to the
      Index constructor).

      You will probably find it most convenient to use the
      :func:`~hypothesis.extra.pandas.indexes` or
      :func:`~hypothesis.extra.pandas.range_indexes` function to produce
      values for this argument.

    Usage:

    The expected usage pattern is that you use :class:`column` and
    :func:`columns` to specify a fixed shape of the DataFrame you want as
    follows. For example the following gives a two column data frame:

    .. code-block:: pycon

        >>> from hypothesis.extra.pandas import column, data_frames
        >>> data_frames([
        ... column('A', dtype=int), column('B', dtype=float)]).example()
                    A              B
        0  2021915903  1.793898e+232
        1  1146643993            inf
        2 -2096165693   1.000000e+07

    If you want the values in different columns to interact in some way you
    can use the rows argument. For example the following gives a two column
    DataFrame where the value in the first column is always at most the value
    in the second:

    .. code-block:: pycon

        >>> from hypothesis.extra.pandas import column, data_frames
        >>> import hypothesis.strategies as st
        >>> data_frames(
        ...     rows=st.tuples(st.floats(allow_nan=False),
        ...                    st.floats(allow_nan=False)).map(sorted)
        ... ).example()
                       0             1
        0  -3.402823e+38  9.007199e+15
        1 -1.562796e-298  5.000000e-01

    You can also combine the two:

    .. code-block:: pycon

        >>> from hypothesis.extra.pandas import columns, data_frames
        >>> import hypothesis.strategies as st
        >>> data_frames(
        ...     columns=columns(["lo", "hi"], dtype=float),
        ...     rows=st.tuples(st.floats(allow_nan=False),
        ...                    st.floats(allow_nan=False)).map(sorted)
        ... ).example()
                 lo            hi
        0   9.314723e-49  4.353037e+45
        1  -9.999900e-01  1.000000e+07
        2 -2.152861e+134 -1.069317e-73

    (Note that the column dtype must still be specified and will not be
    inferred from the rows. This restriction may be lifted in future).

    Combining rows and columns has the following behaviour:

    * The column names and dtypes will be used.
    * If the column is required to be unique, this will be enforced.
    * Any values missing from the generated rows will be provided using the
      column's fill.
    * Any values in the row not present in the column specification (if
      dicts are passed, if there are keys with no corresponding column name,
      if sequences are passed if there are too many items) will result in
      InvalidArgument being raised.
    """
    if index is None:
        index = range_indexes()
    else:
        st.check_strategy(index)

    index_strategy = index

    if columns is None:
        if rows is None:
            raise InvalidArgument("At least one of rows and columns must be provided")
        else:

            @st.composite
            def rows_only(draw):
                index = draw(index_strategy)

                @check_function
                def row():
                    result = draw(rows)
                    check_type(abc.Iterable, result, "draw(row)")
                    return result

                if len(index) > 0:
                    return pandas.DataFrame([row() for _ in index], index=index)
                else:
                    # If we haven't drawn any rows we need to draw one row and
                    # then discard it so that we get a consistent shape for the
                    # DataFrame.
                    base = pandas.DataFrame([row()])
                    return base.drop(0)

            return rows_only()

    assert columns is not None
    cols = try_convert(tuple, columns, "columns")  # type: Sequence[column]

    rewritten_columns = []
    column_names = set()  # type: Set[str]

    for i, c in enumerate(cols):
        check_type(column, c, "columns[%d]" % (i,))

        c = copy(c)
        if c.name is None:
            label = "columns[%d]" % (i,)
            c.name = i
        else:
            label = c.name
            try:
                hash(c.name)
            except TypeError:
                raise InvalidArgument(
                    "Column names must be hashable, but columns[%d].name was "
                    "%r of type %s, which cannot be hashed."
                    % (i, c.name, type(c.name).__name__)
                )

        if c.name in column_names:
            raise InvalidArgument("duplicate definition of column name %r" % (c.name,))

        column_names.add(c.name)

        c.elements, c.dtype = elements_and_dtype(c.elements, c.dtype, label)

        if c.dtype is None and rows is not None:
            raise InvalidArgument(
                "Must specify a dtype for all columns when combining rows with"
                " columns."
            )

        c.fill = npst.fill_for(
            fill=c.fill, elements=c.elements, unique=c.unique, name=label
        )

        rewritten_columns.append(c)

    if rows is None:

        @st.composite
        def just_draw_columns(draw):
            index = draw(index_strategy)
            local_index_strategy = st.just(index)

            data = OrderedDict((c.name, None) for c in rewritten_columns)

            # Depending on how the columns are going to be generated we group
            # them differently to get better shrinking. For columns with fill
            # enabled, the elements can be shrunk independently of the size,
            # so we can just shrink by shrinking the index then shrinking the
            # length and are generally much more free to move data around.

            # For columns with no filling the problem is harder, and drawing
            # them like that would result in rows being very far apart from
            # each other in the underlying data stream, which gets in the way
            # of shrinking. So what we do is reorder and draw those columns
            # row wise, so that the values of each row are next to each other.
            # This makes life easier for the shrinker when deleting blocks of
            # data.
            columns_without_fill = [c for c in rewritten_columns if c.fill.is_empty]

            if columns_without_fill:
                for c in columns_without_fill:
                    data[c.name] = pandas.Series(
                        np.zeros(shape=len(index), dtype=c.dtype), index=index
                    )
                seen = {c.name: set() for c in columns_without_fill if c.unique}

                for i in hrange(len(index)):
                    for c in columns_without_fill:
                        if c.unique:
                            for _ in range(5):
                                value = draw(c.elements)
                                if value not in seen[c.name]:
                                    seen[c.name].add(value)
                                    break
                            else:
                                reject()
                        else:
                            value = draw(c.elements)
                        data[c.name][i] = value

            for c in rewritten_columns:
                if not c.fill.is_empty:
                    data[c.name] = draw(
                        series(
                            index=local_index_strategy,
                            dtype=c.dtype,
                            elements=c.elements,
                            fill=c.fill,
                            unique=c.unique,
                        )
                    )

            return pandas.DataFrame(data, index=index)

        return just_draw_columns()
    else:

        @st.composite
        def assign_rows(draw):
            index = draw(index_strategy)

            result = pandas.DataFrame(
                OrderedDict(
                    (
                        c.name,
                        pandas.Series(
                            np.zeros(dtype=c.dtype, shape=len(index)), dtype=c.dtype
                        ),
                    )
                    for c in rewritten_columns
                ),
                index=index,
            )

            fills = {}

            any_unique = any(c.unique for c in rewritten_columns)

            if any_unique:
                all_seen = [set() if c.unique else None for c in rewritten_columns]
                while all_seen[-1] is None:
                    all_seen.pop()

            for row_index in hrange(len(index)):
                for _ in hrange(5):
                    original_row = draw(rows)
                    row = original_row
                    if isinstance(row, dict):
                        as_list = [None] * len(rewritten_columns)
                        for i, c in enumerate(rewritten_columns):
                            try:
                                as_list[i] = row[c.name]
                            except KeyError:
                                try:
                                    as_list[i] = fills[i]
                                except KeyError:
                                    fills[i] = draw(c.fill)
                                    as_list[i] = fills[i]
                        for k in row:
                            if k not in column_names:
                                raise InvalidArgument(
                                    "Row %r contains column %r not in columns %r)"
                                    % (row, k, [c.name for c in rewritten_columns])
                                )
                        row = as_list
                    if any_unique:
                        has_duplicate = False
                        for seen, value in zip(all_seen, row):
                            if seen is None:
                                continue
                            if value in seen:
                                has_duplicate = True
                                break
                            seen.add(value)
                        if has_duplicate:
                            continue
                    row = list(try_convert(tuple, row, "draw(rows)"))

                    if len(row) > len(rewritten_columns):
                        raise InvalidArgument(
                            (
                                "Row %r contains too many entries. Has %d but "
                                "expected at most %d"
                            )
                            % (original_row, len(row), len(rewritten_columns))
                        )
                    while len(row) < len(rewritten_columns):
                        row.append(draw(rewritten_columns[len(row)].fill))
                    result.iloc[row_index] = row
                    break
                else:
                    reject()
            return result

        return assign_rows()
