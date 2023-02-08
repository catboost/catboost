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

import math
import re
from collections import namedtuple

import numpy as np

import hypothesis.internal.conjecture.utils as cu
import hypothesis.strategies._internal.core as st
from hypothesis import Verbosity, assume
from hypothesis._settings import note_deprecation
from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import (
    PY2,
    hrange,
    integer_types,
    quiet_raise,
    string_types,
)
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import proxies, reserved_means_kwonly_star
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.reporting import current_verbosity
from hypothesis.strategies._internal import SearchStrategy
from hypothesis.utils.conventions import UniqueIdentifier, not_set

if PY2:
    BroadcastableShapes = namedtuple(
        "BroadcastableShapes", ["input_shapes", "result_shape"]
    )
else:
    from typing import NamedTuple, Tuple

    Shape = Tuple[int, ...]  # noqa
    BroadcastableShapes = NamedTuple(
        "BroadcastableShapes",
        [("input_shapes", Tuple[Shape, ...]), ("result_shape", Shape)],
    )

if False:
    from typing import Any, Union, Sequence, Tuple  # noqa
    from hypothesis.strategies._internal.strategies import T  # noqa

    BasicIndex = Tuple[Union[int, slice, ellipsis, np.newaxis], ...]  # noqa

TIME_RESOLUTIONS = tuple("Y  M  D  h  m  s  ms  us  ns  ps  fs  as".split())


@st.defines_strategy_with_reusable_values
def from_dtype(dtype):
    # type: (np.dtype) -> st.SearchStrategy[Any]
    """Creates a strategy which can generate any value of the given dtype."""
    check_type(np.dtype, dtype, "dtype")
    # Compound datatypes, eg 'f4,f4,f4'
    if dtype.names is not None:
        # mapping np.void.type over a strategy is nonsense, so return now.
        return st.tuples(*[from_dtype(dtype.fields[name][0]) for name in dtype.names])

    # Subarray datatypes, eg '(2, 3)i4'
    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape)

    # Scalar datatypes
    if dtype.kind == u"b":
        result = st.booleans()  # type: SearchStrategy[Any]
    elif dtype.kind == u"f":
        if dtype.itemsize == 2:
            result = st.floats(width=16)
        elif dtype.itemsize == 4:
            result = st.floats(width=32)
        else:
            result = st.floats()
    elif dtype.kind == u"c":
        if dtype.itemsize == 8:
            float32 = st.floats(width=32)
            result = st.builds(complex, float32, float32)
        else:
            result = st.complex_numbers()
    elif dtype.kind in (u"S", u"a"):
        # Numpy strings are null-terminated; only allow round-trippable values.
        # `itemsize == 0` means 'fixed length determined at array creation'
        result = st.binary(max_size=dtype.itemsize or None).filter(
            lambda b: b[-1:] != b"\0"
        )
    elif dtype.kind == u"u":
        result = st.integers(min_value=0, max_value=2 ** (8 * dtype.itemsize) - 1)
    elif dtype.kind == u"i":
        overflow = 2 ** (8 * dtype.itemsize - 1)
        result = st.integers(min_value=-overflow, max_value=overflow - 1)
    elif dtype.kind == u"U":
        # Encoded in UTF-32 (four bytes/codepoint) and null-terminated
        result = st.text(max_size=(dtype.itemsize or 0) // 4 or None).filter(
            lambda b: b[-1:] != u"\0"
        )
    elif dtype.kind in (u"m", u"M"):
        if "[" in dtype.str:
            res = st.just(dtype.str.split("[")[-1][:-1])
        else:
            res = st.sampled_from(TIME_RESOLUTIONS)
        result = st.builds(dtype.type, st.integers(-(2 ** 63), 2 ** 63 - 1), res)
    else:
        raise InvalidArgument(u"No strategy inference for {}".format(dtype))
    return result.map(dtype.type)


@check_function
def check_argument(condition, fail_message, *f_args, **f_kwargs):
    if not condition:
        raise InvalidArgument(fail_message.format(*f_args, **f_kwargs))


@check_function
def order_check(name, floor, small, large):
    check_argument(
        floor <= small,
        u"min_{name} must be at least {} but was {}",
        floor,
        small,
        name=name,
    )
    check_argument(
        small <= large,
        u"min_{name}={} is larger than max_{name}={}",
        small,
        large,
        name=name,
    )


class ArrayStrategy(SearchStrategy):
    def __init__(self, element_strategy, shape, dtype, fill, unique):
        self.shape = tuple(shape)
        self.fill = fill
        self.array_size = int(np.prod(shape))
        self.dtype = dtype
        self.element_strategy = element_strategy
        self.unique = unique

    def set_element(self, data, result, idx, strategy=None):
        strategy = strategy or self.element_strategy
        val = data.draw(strategy)
        result[idx] = val
        if self._report_overflow and val != result[idx] and val == val:
            note_deprecation(
                "Generated array element %r from %r cannot be represented as "
                "dtype %r - instead it becomes %r (type %r).  Consider using a more "
                "precise strategy, for example passing the `width` argument to "
                "`floats()`, as this will be an error in a future version."
                % (val, strategy, self.dtype, result[idx], type(result[idx])),
                since="2019-07-28",
            )
            # Because the message includes the value of the generated element,
            # it would be easy to spam users with thousands of warnings.
            # We therefore only warn once per draw, unless in verbose mode.
            self._report_overflow = current_verbosity() >= Verbosity.verbose

    def do_draw(self, data):
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)

        # Because Numpy allocates memory for strings at array creation, if we have
        # an unsized string dtype we'll fill an object array and then cast it back.
        unsized_string_dtype = (
            self.dtype.kind in (u"S", u"a", u"U") and self.dtype.itemsize == 0
        )

        # Reset this flag for each test case to emit warnings from set_element
        # Skip the check for object or void (multi-element) dtypes
        self._report_overflow = self.dtype.kind not in "OV" and not unsized_string_dtype

        # This could legitimately be a np.empty, but the performance gains for
        # that would be so marginal that there's really not much point risking
        # undefined behaviour shenanigans.
        result = np.zeros(
            shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype
        )

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly
            # disabled it or because the default behaviour was used and our
            # elements strategy does not produce reusable values), so we must
            # generate a fully dense array with a freshly drawn value for each
            # entry.
            if self.unique:
                seen = set()
                elements = cu.many(
                    data,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    average_size=self.array_size,
                )
                i = 0
                while elements.more():
                    # We assign first because this means we check for
                    # uniqueness after numpy has converted it to the relevant
                    # type for us. Because we don't increment the counter on
                    # a duplicate we will overwrite it on the next draw.
                    self.set_element(data, result, i)
                    if result[i] not in seen:
                        seen.add(result[i])
                        i += 1
                    else:
                        elements.reject()
            else:
                for i in hrange(len(result)):
                    self.set_element(data, result, i)
        else:
            # We draw numpy arrays as "sparse with an offset". We draw a
            # collection of index assignments within the array and assign
            # fresh values from our elements strategy to those indices. If at
            # the end we have not assigned every element then we draw a single
            # value from our fill strategy and use that to populate the
            # remaining positions with that strategy.

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                # sqrt isn't chosen for any particularly principled reason. It
                # just grows reasonably quickly but sublinearly, and for small
                # arrays it represents a decent fraction of the array size.
                average_size=math.sqrt(self.array_size),
            )

            needs_fill = np.full(self.array_size, True)
            seen = set()

            while elements.more():
                i = cu.integer_range(data, 0, self.array_size - 1)
                if not needs_fill[i]:
                    elements.reject()
                    continue
                self.set_element(data, result, i)
                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(result[i])
                needs_fill[i] = False
            if needs_fill.any():
                # We didn't fill all of the indices in the early loop, so we
                # put a fill value into the rest.

                # We have to do this hilarious little song and dance to work
                # around numpy's special handling of iterable values. If the
                # value here were e.g. a tuple then neither array creation
                # nor putmask would do the right thing. But by creating an
                # array of size one and then assigning the fill value as a
                # single element, we both get an array with the right value in
                # it and putmask will do the right thing by repeating the
                # values of the array across the mask.
                one_element = np.zeros(
                    shape=1, dtype=object if unsized_string_dtype else self.dtype
                )
                self.set_element(data, one_element, 0, self.fill)
                if unsized_string_dtype:
                    one_element = one_element.astype(self.dtype)
                fill_value = one_element[0]
                if self.unique:
                    try:
                        is_nan = np.isnan(fill_value)
                    except TypeError:
                        is_nan = False

                    if not is_nan:
                        raise InvalidArgument(
                            "Cannot fill unique array with non-NaN "
                            "value %r" % (fill_value,)
                        )

                np.putmask(result, needs_fill, one_element)

        if unsized_string_dtype:
            out = result.astype(self.dtype)
            mismatch = out != result
            if mismatch.any():
                note_deprecation(
                    "Array elements %r cannot be represented as dtype %r - instead "
                    "they becomes %r.  Use a more precise strategy, e.g. without "
                    "trailing null bytes, as this will be an error future versions."
                    % (result[mismatch], self.dtype, out[mismatch]),
                    since="2019-07-28",
                )
            result = out

        return result.reshape(self.shape)


@check_function
def fill_for(elements, unique, fill, name=""):
    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    else:
        st.check_strategy(fill, "%s.fill" % (name,) if name else "fill")
    return fill


@st.defines_strategy
def arrays(
    dtype,  # type: Any
    shape,  # type: Union[int, Shape, st.SearchStrategy[Shape]]
    elements=None,  # type: st.SearchStrategy[Any]
    fill=None,  # type: st.SearchStrategy[Any]
    unique=False,  # type: bool
):
    # type: (...) -> st.SearchStrategy[np.ndarray]
    r"""Returns a strategy for generating :class:`numpy:numpy.ndarray`\ s.

    * ``dtype`` may be any valid input to :class:`~numpy:numpy.dtype`
      (this includes :class:`~numpy:numpy.dtype` objects), or a strategy that
      generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a
      strategy that generates such values.
    * ``elements`` is a strategy for generating values to put in the array.
      If it is None a suitable value will be inferred based on the dtype,
      which may give any legal value (including eg ``NaN`` for floats).
      If you have more specific requirements, you should supply your own
      elements strategy.
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If None, a suitable default will be inferred
      based on the other arguments. If set to
      :func:`~hypothesis.strategies.nothing` then filling
      behaviour will be disabled entirely and every element will be generated
      independently.
    * ``unique`` specifies if the elements of the array should all be
      distinct from one another. Note that in this case multiple NaN values
      may still be allowed. If fill is also set, the only valid values for
      it to return are NaN values (anything for which :obj:`numpy:numpy.isnan`
      returns True. So e.g. for complex numbers (nan+1j) is also a valid fill).
      Note that if unique is set to True the generated values must be hashable.

    Arrays of specified ``dtype`` and ``shape`` are generated for example
    like this:

    .. code-block:: pycon

      >>> import numpy as np
      >>> arrays(np.int8, (2, 3)).example()
      array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)

    - See :doc:`What you can generate and how <data>`.

    .. code-block:: pycon

      >>> import numpy as np
      >>> from hypothesis.strategies import floats
      >>> arrays(np.float, 3, elements=floats(0, 1)).example()
      array([ 0.88974794,  0.77387938,  0.1977879 ])

    Array values are generated in two parts:

    1. Some subset of the coordinates of the array are populated with a value
       drawn from the elements strategy (or its inferred form).
    2. If any coordinates were not assigned in the previous step, a single
       value is drawn from the fill strategy and is assigned to all remaining
       places.

    You can set fill to :func:`~hypothesis.strategies.nothing` if you want to
    disable this behaviour and draw a value for every element.

    If fill is set to None then it will attempt to infer the correct behaviour
    automatically: If unique is True, no filling will occur by default.
    Otherwise, if it looks safe to reuse the values of elements across
    multiple coordinates (this will be the case for any inferred strategy, and
    for most of the builtins, but is not the case for mutable values or
    strategies built with flatmap, map, composite, etc) then it will use the
    elements strategy as the fill, else it will default to having no fill.

    Having a fill helps Hypothesis craft high quality examples, but its
    main importance is when the array generated is large: Hypothesis is
    primarily designed around testing small examples. If you have arrays with
    hundreds or more elements, having a fill value is essential if you want
    your tests to run in reasonable time.
    """
    # We support passing strategies as arguments for convenience, or at least
    # for legacy reasons, but don't want to pay the perf cost of a composite
    # strategy (i.e. repeated argument handling and validation) when it's not
    # needed.  So we get the best of both worlds by recursing with flatmap,
    # but only when it's actually needed.
    if isinstance(dtype, SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(dtype, s, elements=elements, fill=fill, unique=unique)
        )
    # From here on, we're only dealing with values and it's relatively simple.
    dtype = np.dtype(dtype)
    if elements is None:
        elements = from_dtype(dtype)
    if isinstance(shape, integer_types):
        shape = (shape,)
    shape = tuple(shape)
    check_argument(
        all(isinstance(s, integer_types) for s in shape),
        "Array shape must be integer in each dimension, provided shape was {}",
        shape,
    )
    fill = fill_for(elements=elements, unique=unique, fill=fill)
    return ArrayStrategy(elements, shape, dtype, fill, unique)


@st.defines_strategy
def array_shapes(min_dims=1, max_dims=None, min_side=1, max_side=None):
    # type: (int, int, int, int) -> st.SearchStrategy[Shape]
    """Return a strategy for array shapes (tuples of int >= 1)."""
    check_type(integer_types, min_dims, "min_dims")
    check_type(integer_types, min_side, "min_side")
    if min_dims > 32:
        raise InvalidArgument(
            "Got min_dims=%r, but numpy does not support arrays greater than 32 dimensions"
            % min_dims
        )
    if max_dims is None:
        max_dims = min(min_dims + 2, 32)
    check_type(integer_types, max_dims, "max_dims")
    if max_dims > 32:
        raise InvalidArgument(
            "Got max_dims=%r, but numpy does not support arrays greater than 32 dimensions"
            % max_dims
        )
    if max_side is None:
        max_side = min_side + 5
    check_type(integer_types, max_side, "max_side")
    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    return st.lists(
        st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims
    ).map(tuple)


@st.defines_strategy
def scalar_dtypes():
    # type: () -> st.SearchStrategy[np.dtype]
    """Return a strategy that can return any non-flexible scalar dtype."""
    return st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating_dtypes(),
        complex_number_dtypes(),
        datetime64_dtypes(),
        timedelta64_dtypes(),
    )


def defines_dtype_strategy(strat):
    # type: (T) -> T
    @st.defines_strategy
    @proxies(strat)
    def inner(*args, **kwargs):
        return strat(*args, **kwargs).map(np.dtype)

    return inner


@defines_dtype_strategy
def boolean_dtypes():
    # type: () -> st.SearchStrategy[np.dtype]
    return st.just("?")


def dtype_factory(kind, sizes, valid_sizes, endianness):
    # Utility function, shared logic for most integer and string types
    valid_endian = ("?", "<", "=", ">")
    check_argument(
        endianness in valid_endian,
        u"Unknown endianness: was {}, must be in {}",
        endianness,
        valid_endian,
    )
    if valid_sizes is not None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        check_argument(sizes, "Dtype must have at least one possible size.")
        check_argument(
            all(s in valid_sizes for s in sizes),
            u"Invalid sizes: was {} must be an item or sequence " u"in {}",
            sizes,
            valid_sizes,
        )
        if all(isinstance(s, int) for s in sizes):
            sizes = sorted({s // 8 for s in sizes})
    strat = st.sampled_from(sizes)
    if "{}" not in kind:
        kind += "{}"
    if endianness == "?":
        return strat.map(("<" + kind).format) | strat.map((">" + kind).format)
    return strat.map((endianness + kind).format)


@defines_dtype_strategy
def unsigned_integer_dtypes(endianness="?", sizes=(8, 16, 32, 64)):
    # type: (str, Sequence[int]) -> st.SearchStrategy[np.dtype]
    """Return a strategy for unsigned integer dtypes.

    endianness may be ``<`` for little-endian, ``>`` for big-endian,
    ``=`` for native byte order, or ``?`` to allow either byte order.
    This argument only applies to dtypes of more than one byte.

    sizes must be a collection of integer sizes in bits.  The default
    (8, 16, 32, 64) covers the full range of sizes.
    """
    return dtype_factory("u", sizes, (8, 16, 32, 64), endianness)


@defines_dtype_strategy
def integer_dtypes(endianness="?", sizes=(8, 16, 32, 64)):
    # type: (str, Sequence[int]) -> st.SearchStrategy[np.dtype]
    """Return a strategy for signed integer dtypes.

    endianness and sizes are treated as for
    :func:`unsigned_integer_dtypes`.
    """
    return dtype_factory("i", sizes, (8, 16, 32, 64), endianness)


@defines_dtype_strategy
def floating_dtypes(endianness="?", sizes=(16, 32, 64)):
    # type: (str, Sequence[int]) -> st.SearchStrategy[np.dtype]
    """Return a strategy for floating-point dtypes.

    sizes is the size in bits of floating-point number.  Some machines support
    96- or 128-bit floats, but these are not generated by default.

    Larger floats (96 and 128 bit real parts) are not supported on all
    platforms and therefore disabled by default.  To generate these dtypes,
    include these values in the sizes argument.
    """
    return dtype_factory("f", sizes, (16, 32, 64, 96, 128), endianness)


@defines_dtype_strategy
def complex_number_dtypes(endianness="?", sizes=(64, 128)):
    # type: (str, Sequence[int]) -> st.SearchStrategy[np.dtype]
    """Return a strategy for complex-number dtypes.

    sizes is the total size in bits of a complex number, which consists
    of two floats.  Complex halfs (a 16-bit real part) are not supported
    by numpy and will not be generated by this strategy.
    """
    return dtype_factory("c", sizes, (64, 128, 192, 256), endianness)


@check_function
def validate_time_slice(max_period, min_period):
    check_argument(
        max_period in TIME_RESOLUTIONS,
        u"max_period {} must be a valid resolution in {}",
        max_period,
        TIME_RESOLUTIONS,
    )
    check_argument(
        min_period in TIME_RESOLUTIONS,
        u"min_period {} must be a valid resolution in {}",
        min_period,
        TIME_RESOLUTIONS,
    )
    start = TIME_RESOLUTIONS.index(max_period)
    end = TIME_RESOLUTIONS.index(min_period) + 1
    check_argument(
        start < end,
        u"max_period {} must be earlier in sequence {} than " u"min_period {}",
        max_period,
        TIME_RESOLUTIONS,
        min_period,
    )
    return TIME_RESOLUTIONS[start:end]


@defines_dtype_strategy
def datetime64_dtypes(max_period="Y", min_period="ns", endianness="?"):
    # type: (str, str, str) -> st.SearchStrategy[np.dtype]
    """Return a strategy for datetime64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "datetime64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def timedelta64_dtypes(max_period="Y", min_period="ns", endianness="?"):
    # type: (str, str, str) -> st.SearchStrategy[np.dtype]
    """Return a strategy for timedelta64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "timedelta64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def byte_string_dtypes(endianness="?", min_len=1, max_len=16):
    # type: (str, int, int) -> st.SearchStrategy[np.dtype]
    """Return a strategy for generating bytestring dtypes, of various lengths
    and byteorder.

    While Hypothesis' string strategies can generate empty strings, string
    dtypes with length 0 indicate that size is still to be determined, so
    the minimum length for string dtypes is 1.
    """
    if min_len == 0:
        note_deprecation(
            "generating byte string dtypes for unspecified length ('S0') "
            "is deprecated. min_len will be 1 instead.",
            since="2019-09-09",
        )
        min_len = 1
    if max_len == 0:
        note_deprecation(
            "generating byte string dtypes for unspecified length ('S0') "
            "is deprecated. max_len will be 1 instead.",
            since="2019-09-09",
        )
        max_len = 1

    order_check("len", 1, min_len, max_len)
    return dtype_factory("S", list(range(min_len, max_len + 1)), None, endianness)


@defines_dtype_strategy
def unicode_string_dtypes(endianness="?", min_len=1, max_len=16):
    # type: (str, int, int) -> st.SearchStrategy[np.dtype]
    """Return a strategy for generating unicode string dtypes, of various
    lengths and byteorder.

    While Hypothesis' string strategies can generate empty strings, string
    dtypes with length 0 indicate that size is still to be determined, so
    the minimum length for string dtypes is 1.
    """
    if min_len == 0:
        note_deprecation(
            "generating unicode string dtypes for unspecified length ('U0') "
            "is deprecated. min_len will be 1 instead.",
            since="2019-09-09",
        )
        min_len = 1
    if max_len == 0:
        note_deprecation(
            "generating unicode string dtypes for unspecified length ('U0') "
            "is deprecated. max_len will be 1 instead.",
            since="2019-09-09",
        )
        max_len = 1

    order_check("len", 1, min_len, max_len)
    return dtype_factory("U", list(range(min_len, max_len + 1)), None, endianness)


@defines_dtype_strategy
def array_dtypes(
    subtype_strategy=scalar_dtypes(),  # type: st.SearchStrategy[np.dtype]
    min_size=1,  # type: int
    max_size=5,  # type: int
    allow_subarrays=False,  # type: bool
):
    # type: (...) -> st.SearchStrategy[np.dtype]
    """Return a strategy for generating array (compound) dtypes, with members
    drawn from the given subtype strategy."""
    order_check("size", 0, min_size, max_size)
    # Field names must be native strings and the empty string is weird; see #1963.
    if PY2:
        field_names = st.binary(min_size=1)
    else:
        field_names = st.text(min_size=1)
    elements = st.tuples(field_names, subtype_strategy)
    if allow_subarrays:
        elements |= st.tuples(
            field_names, subtype_strategy, array_shapes(max_dims=2, max_side=2)
        )
    return st.lists(
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=lambda d: d[0],
    )


@st.defines_strategy
def nested_dtypes(
    subtype_strategy=scalar_dtypes(),  # type: st.SearchStrategy[np.dtype]
    max_leaves=10,  # type: int
    max_itemsize=None,  # type: int
):
    # type: (...) -> st.SearchStrategy[np.dtype]
    """Return the most-general dtype strategy.

    Elements drawn from this strategy may be simple (from the
    subtype_strategy), or several such values drawn from
    :func:`array_dtypes` with ``allow_subarrays=True``. Subdtypes in an
    array dtype may be nested to any depth, subject to the max_leaves
    argument.
    """
    return st.recursive(
        subtype_strategy, lambda x: array_dtypes(x, allow_subarrays=True), max_leaves
    ).filter(lambda d: max_itemsize is None or d.itemsize <= max_itemsize)


@st.defines_strategy
def valid_tuple_axes(ndim, min_size=0, max_size=None):
    # type: (int, int, int) -> st.SearchStrategy[Shape]
    """Return a strategy for generating permissible tuple-values for the
    ``axis`` argument for a numpy sequential function (e.g.
    :func:`numpy:numpy.sum`), given an array of the specified
    dimensionality.

    All tuples will have an length >= min_size and <= max_size. The default
    value for max_size is ``ndim``.

    Examples from this strategy shrink towards an empty tuple, which render
    most sequential functions as no-ops.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> [valid_tuple_axes(3).example() for i in range(4)]
        [(-3, 1), (0, 1, -1), (0, 2), (0, -2, 2)]

    ``valid_tuple_axes`` can be joined with other strategies to generate
    any type of valid axis object, i.e. integers, tuples, and ``None``:

    .. code-block:: pycon

        any_axis_strategy = none() | integers(-ndim, ndim - 1) | valid_tuple_axes(ndim)

    """
    if max_size is None:
        max_size = ndim

    check_type(integer_types, ndim, "ndim")
    check_type(integer_types, min_size, "min_size")
    check_type(integer_types, max_size, "max_size")
    order_check("size", 0, min_size, max_size)
    check_valid_interval(max_size, ndim, "max_size", "ndim")

    # shrink axis values from negative to positive
    axes = st.integers(0, max(0, 2 * ndim - 1)).map(
        lambda x: x if x < ndim else x - 2 * ndim
    )
    return st.lists(axes, min_size, max_size, unique_by=lambda x: x % ndim).map(tuple)


@st.defines_strategy
def broadcastable_shapes(shape, min_dims=0, max_dims=None, min_side=1, max_side=None):
    # type: (Shape, int, int, int, int) -> st.SearchStrategy[Shape]
    """Return a strategy for generating shapes that are broadcast-compatible
    with the provided shape.

    Examples from this strategy shrink towards a shape with length ``min_dims``.
    The size of an aligned dimension shrinks towards size ``1``. The
    size of an unaligned dimension shrink towards ``min_side``.

    * ``shape`` a tuple of integers
    * ``min_dims`` The smallest length that the generated shape can possess.
    * ``max_dims`` The largest length that the generated shape can possess.
      The default-value for ``max_dims`` is ``min(32, max(len(shape), min_dims) + 2)``.
    * ``min_side`` The smallest size that an unaligned dimension can possess.
    * ``max_side`` The largest size that an unaligned dimension can possess.
      The default value is 2 + 'size-of-largest-aligned-dimension'.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> [broadcastable_shapes(shape=(2, 3)).example() for i in range(5)]
        [(1, 3), (), (2, 3), (2, 1), (4, 1, 3), (3, )]

    """
    check_type(tuple, shape, "shape")
    strict_check = max_side is None or max_dims is None
    check_type(integer_types, min_side, "min_side")
    check_type(integer_types, min_dims, "min_dims")

    if max_dims is None:
        max_dims = min(32, max(len(shape), min_dims) + 2)
    else:
        check_type(integer_types, max_dims, "max_dims")

    if max_side is None:
        max_side = max(tuple(shape[-max_dims:]) + (min_side,)) + 2
    else:
        check_type(integer_types, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if 32 < max_dims:
        raise InvalidArgument("max_dims cannot exceed 32")

    dims, bnd_name = (max_dims, "max_dims") if strict_check else (min_dims, "min_dims")

    # check for unsatisfiable min_side
    if not all(min_side <= s for s in shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            "Given shape=%r, there are no broadcast-compatible "
            "shapes that satisfy: %s=%s and min_side=%s"
            % (shape, bnd_name, dims, min_side)
        )

    # check for unsatisfiable [min_side, max_side]
    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in shape[::-1][:dims])
    ):
        raise InvalidArgument(
            "Given shape=%r, there are no broadcast-compatible shapes "
            "that satisfy: %s=%s and [min_side=%s, max_side=%s]"
            % (shape, bnd_name, dims, min_side, max_side)
        )

    if not strict_check:
        # reduce max_dims to exclude unsatisfiable dimensions
        for n, s in zip(range(max_dims), reversed(shape)):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=1,
        base_shape=shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    ).map(lambda x: x.input_shapes[0])


class MutuallyBroadcastableShapesStrategy(SearchStrategy):
    def __init__(
        self,
        num_shapes,
        signature=None,
        base_shape=(),
        min_dims=0,
        max_dims=None,
        min_side=1,
        max_side=None,
    ):
        assert 0 <= min_side <= max_side
        assert 0 <= min_dims <= max_dims <= 32
        SearchStrategy.__init__(self)
        self.base_shape = base_shape
        self.side_strat = st.integers(min_side, max_side)
        self.num_shapes = num_shapes
        self.signature = signature
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.min_side = min_side
        self.max_side = max_side

        self.size_one_allowed = self.min_side <= 1 <= self.max_side

    def do_draw(self, data):
        # We don't usually have a gufunc signature; do the common case first & fast.
        if self.signature is None:
            return self._draw_loop_dimensions(data)

        # When we *do*, draw the core dims, then draw loop dims, and finally combine.
        core_in, core_res = self._draw_core_dimensions(data)

        # If some core shape has omitted optional dimensions, it's an error to add
        # loop dimensions to it.  We never omit core dims if min_dims >= 1.
        # This ensures that we respect Numpy's gufunc broadcasting semantics and user
        # constraints without needing to check whether the loop dims will be
        # interpreted as an invalid substitute for the omitted core dims.
        # We may implement this check later!
        use = [None not in shp for shp in core_in]
        loop_in, loop_res = self._draw_loop_dimensions(data, use=use)

        def add_shape(loop, core):
            return tuple(x for x in (loop + core)[-32:] if x is not None)

        return BroadcastableShapes(
            input_shapes=tuple(add_shape(l, c) for l, c in zip(loop_in, core_in)),
            result_shape=add_shape(loop_res, core_res),
        )

    def _draw_core_dimensions(self, data):
        # Draw gufunc core dimensions, with None standing for optional dimensions
        # that will not be present in the final shape.  We track omitted dims so
        # that we can do an accurate per-shape length cap.
        dims = {}
        shapes = []
        for shape in self.signature.input_shapes + (self.signature.result_shape,):
            shapes.append([])
            for name in shape:
                if name.isdigit():
                    shapes[-1].append(int(name))
                    continue
                if name not in dims:
                    dim = name.strip("?")
                    dims[dim] = data.draw(self.side_strat)
                    if self.min_dims == 0 and not data.draw_bits(3):
                        dims[dim + "?"] = None
                    else:
                        dims[dim + "?"] = dims[dim]
                shapes[-1].append(dims[name])
        return tuple(tuple(s) for s in shapes[:-1]), tuple(shapes[-1])

    def _draw_loop_dimensions(self, data, use=None):
        # All shapes are handled in column-major order; i.e. they are reversed
        base_shape = self.base_shape[::-1]
        result_shape = list(base_shape)
        shapes = [[] for _ in range(self.num_shapes)]
        if use is None:
            use = [True for _ in range(self.num_shapes)]
        else:
            assert len(use) == self.num_shapes
            assert all(isinstance(x, bool) for x in use)

        for dim_count in range(1, self.max_dims + 1):
            dim = dim_count - 1

            # We begin by drawing a valid dimension-size for the given
            # dimension. This restricts the variability across the shapes
            # at this dimension such that they can only choose between
            # this size and a singleton dimension.
            if len(base_shape) < dim_count or base_shape[dim] == 1:
                # dim is unrestricted by the base-shape: shrink to min_side
                dim_side = data.draw(self.side_strat)
            elif base_shape[dim] <= self.max_side:
                # dim is aligned with non-singleton base-dim
                dim_side = base_shape[dim]
            else:
                # only a singleton is valid in alignment with the base-dim
                dim_side = 1

            for shape_id, shape in enumerate(shapes):
                # Populating this dimension-size for each shape, either
                # the drawn size is used or, if permitted, a singleton
                # dimension.
                if dim_count <= len(base_shape) and self.size_one_allowed:
                    # aligned: shrink towards size 1
                    side = data.draw(st.sampled_from([1, dim_side]))
                else:
                    side = dim_side

                # Use a trick where where a biased coin is queried to see
                # if the given shape-tuple will continue to be grown. All
                # of the relevant draws will still be made for the given
                # shape-tuple even if it is no longer being added to.
                # This helps to ensure more stable shrinking behavior.
                if self.min_dims < dim_count:
                    use[shape_id] &= cu.biased_coin(
                        data, 1 - 1 / (1 + self.max_dims - dim)
                    )

                if use[shape_id]:
                    shape.append(side)
                    if len(result_shape) < len(shape):
                        result_shape.append(shape[-1])
                    elif shape[-1] != 1 and result_shape[dim] == 1:
                        result_shape[dim] = shape[-1]
            if not any(use):
                break

        result_shape = result_shape[: max(map(len, [self.base_shape] + shapes))]

        assert len(shapes) == self.num_shapes
        assert all(self.min_dims <= len(s) <= self.max_dims for s in shapes)
        assert all(self.min_side <= s <= self.max_side for side in shapes for s in side)

        return BroadcastableShapes(
            input_shapes=tuple(tuple(reversed(shape)) for shape in shapes),
            result_shape=tuple(reversed(result_shape)),
        )


# See https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
# Implementation based on numpy.lib.function_base._parse_gufunc_signature
# with minor upgrades to handle numeric and optional dimensions.  Examples:
#
#     add       (),()->()                   binary ufunc
#     sum1d     (i)->()                     reduction
#     inner1d   (i),(i)->()                 vector-vector multiplication
#     matmat    (m,n),(n,p)->(m,p)          matrix multiplication
#     vecmat    (n),(n,p)->(p)              vector-matrix multiplication
#     matvec    (m,n),(n)->(m)              matrix-vector multiplication
#     matmul    (m?,n),(n,p?)->(m?,p?)      combination of the four above
#     cross1d   (3),(3)->(3)                cross product with frozen dimensions
#
# Note that while no examples of such usage are given, Numpy does allow
# generalised ufuncs that have *multiple output arrays*.  This is not
# currently supported by Hypothesis - please contact us if you would use it!
#
# We are unsure if gufuncs allow frozen dimensions to be optional, but it's
# easy enough to support here - and so we will unless we learn otherwise.
#
_DIMENSION = r"\w+\??"  # Note that \w permits digits too!
_SHAPE = r"\((?:{0}(?:,{0})".format(_DIMENSION) + r"{0,31})?\)"
_ARGUMENT_LIST = "{0}(?:,{0})*".format(_SHAPE)
_SIGNATURE = r"^{}->{}$".format(_ARGUMENT_LIST, _SHAPE)
_SIGNATURE_MULTIPLE_OUTPUT = r"^{0}->{0}$".format(_ARGUMENT_LIST)

_GUfuncSig = namedtuple("_GUfuncSig", ["input_shapes", "result_shape"])


def _hypothesis_parse_gufunc_signature(signature, all_checks=True):
    # Disable all_checks to better match the Numpy version, for testing
    if not re.match(_SIGNATURE, signature):
        if re.match(_SIGNATURE_MULTIPLE_OUTPUT, signature):
            raise InvalidArgument(
                "Hypothesis does not yet support generalised ufunc signatures "
                "with multiple output arrays - mostly because we don't know of "
                "anyone who uses them!  Please get in touch with us to fix that."
                "\n (signature=%r)" % (signature,)
            )
        if re.match(np.lib.function_base._SIGNATURE, signature):
            raise InvalidArgument(
                "signature=%r matches Numpy's regex for gufunc signatures, but "
                "contains shapes with more than 32 dimensions and is thus invalid."
                % (signature,)
            )
        raise InvalidArgument("%r is not a valid gufunc signature" % (signature,))
    input_shapes, output_shapes = (
        tuple(tuple(re.findall(_DIMENSION, a)) for a in re.findall(_SHAPE, arg_list))
        for arg_list in signature.split("->")
    )
    assert len(output_shapes) == 1
    result_shape = output_shapes[0]
    if all_checks:
        # Check that there are no names in output shape that do not appear in inputs.
        # (kept out of parser function for easier generation of test values)
        # We also disallow frozen optional dimensions - this is ambiguous as there is
        # no way to share an un-named dimension between shapes.  Maybe just padding?
        # Anyway, we disallow it pending clarification from upstream.
        frozen_optional_err = (
            "Got dimension %r, but handling of frozen optional dimensions "
            "is ambiguous.  If you known how this should work, please "
            "contact us to get this fixed and documented (signature=%r)."
        )
        only_out_err = (
            "The %r dimension only appears in the output shape, and is "
            "not frozen, so the size is not determined (signature=%r)."
        )
        names_in = {n.strip("?") for shp in input_shapes for n in shp}
        names_out = {n.strip("?") for n in result_shape}
        for shape in input_shapes + (result_shape,):
            for name in shape:
                try:
                    int(name.strip("?"))
                    if "?" in name:
                        raise InvalidArgument(frozen_optional_err % (name, signature))
                except ValueError:
                    if name.strip("?") in (names_out - names_in):
                        quiet_raise(InvalidArgument(only_out_err % (name, signature)))
    return _GUfuncSig(input_shapes=input_shapes, result_shape=result_shape)


@st.defines_strategy
@reserved_means_kwonly_star
def mutually_broadcastable_shapes(
    __reserved=not_set,  # type: Any
    num_shapes=not_set,  # type: Union[UniqueIdentifier, int]
    signature=not_set,  # type: Union[UniqueIdentifier, str]
    base_shape=(),  # type: Shape
    min_dims=0,  # type: int
    max_dims=None,  # type: int
    min_side=1,  # type: int
    max_side=None,  # type: int
):
    # type: (...) -> st.SearchStrategy[BroadcastableShapes]
    """Return a strategy for generating a specified number of shapes, N, that are
    mutually-broadcastable with one another and with the provided "base-shape".

    The strategy will generate a named-tuple of:

    * input_shapes: the N generated shapes
    * result_shape: the resulting shape, produced by broadcasting the
      N shapes with the base-shape

    Each shape produced from this strategy shrinks towards a shape with length
    ``min_dims``. The size of an aligned dimension shrinks towards being having
    a size of 1. The size of an unaligned dimension shrink towards ``min_side``.

    * ``num_shapes`` The number of mutually broadcast-compatible shapes to generate.
    * ``base-shape`` The shape against which all generated shapes can broadcast.
      The default shape is empty, which corresponds to a scalar and thus does not
      constrain broadcasting at all.
    * ``min_dims`` The smallest length that any generated shape can possess.
    * ``max_dims`` The largest length that any generated shape can possess.
      It cannot exceed 32, which is the greatest supported dimensionality for a
      numpy array. The default-value for ``max_dims`` is
      ``2 + max(len(shape), min_dims)``, capped at 32.
    * ``min_side`` The smallest size that an unaligned dimension can possess.
    * ``max_side`` The largest size that an unaligned dimension can possess.
      The default value is 2 + 'size-of-largest-aligned-dimension'.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> # Draw three shapes, and each shape is broadcast-compatible with `(2, 3)`
        >>> for _ in range(5):
        ...     mutually_broadcastable_shapes(num_shapes=3, base_shape=(2, 3)).example()
        BroadcastableShapes(input_shapes=((4, 1, 3), (4, 2, 3), ()), result_shape=(4, 2, 3))
        BroadcastableShapes(input_shapes=((3,), (1,), (2, 1)), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((3,), (1, 3), (2, 3)), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((), (), ()), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((3,), (), (3,)), result_shape=(2, 3))

    **Use with Generalised Universal Function signatures**

    A :np-ref:`universal function <ufuncs.html>` (or ufunc for short) is a function
    that operates on ndarrays in an element-by-element fashion, supporting array
    broadcasting, type casting, and several other standard features.
    A :np-ref:`generalised ufunc <c-api.generalized-ufuncs.html>` operates on
    sub-arrays rather than elements, based on the "signature" of the function.
    Compare e.g. :obj:`numpy:numpy.add` (ufunc) to :obj:`numpy:numpy.matmul` (gufunc).

    To generate shapes for a gufunc, you can pass the ``signature`` argument instead of
    ``num_shapes``.  This must be a gufunc signature string; which you can write by
    hand or access as e.g. ``np.matmul.signature`` on generalised ufuncs.

    In this case, the ``side`` arguments are applied to the 'core dimensions' as well,
    ignoring any frozen dimensions.  ``base_shape``  and the ``dims`` arguments are
    applied to the 'loop dimensions', and if necessary, the dimensionality of each
    shape is silently capped to respect the 32-dimension limit.

    The generated ``result_shape`` is the real result shape of applying the gufunc
    to arrays of the generated ``input_shapes``, even where this is different to
    broadcasting the loop dimensions.

    gufunc-compatible shapes shrink their loop dimensions as above, towards omitting
    optional core dimensions, and smaller-size core dimensions.

    .. code-block:: pycon

        >>> # np.matmul.signature == "(m?,n),(n,p?)->(m?,p?)"
        >>> for _ in range(3):
        ...     mutually_broadcastable_shapes(signature=np.matmul.signature).example()
        BroadcastableShapes(input_shapes=((2,), (2,)), result_shape=())
        BroadcastableShapes(input_shapes=((3, 4, 2), (1, 2)), result_shape=(3, 4))
        BroadcastableShapes(input_shapes=((4, 2), (1, 2, 3)), result_shape=(4, 3))
    """
    if __reserved is not not_set:
        raise InvalidArgument("Do not pass the __reserved argument.")

    arg_msg = "Pass either the `num_shapes` or the `signature` argument, but not both."
    if num_shapes is not not_set:
        check_argument(signature is not_set, arg_msg)
        check_type(integer_types, num_shapes, "num_shapes")
        assert isinstance(num_shapes, integer_types)  # for mypy
        check_argument(num_shapes >= 1, "num_shapes={} must be at least 1", num_shapes)
        parsed_signature = None
        sig_dims = 0
    else:
        check_argument(signature is not not_set, arg_msg)
        if signature is None:
            raise InvalidArgument(
                "Expected a string, but got invalid signature=None.  "
                "(maybe .signature attribute of an element-wise ufunc?)"
            )
        check_type(string_types, signature, "signature")
        parsed_signature = _hypothesis_parse_gufunc_signature(signature)
        sig_dims = min(
            map(len, parsed_signature.input_shapes + (parsed_signature.result_shape,))
        )
        num_shapes = len(parsed_signature.input_shapes)
        assert num_shapes >= 1

    check_type(tuple, base_shape, "base_shape")
    strict_check = max_dims is not None
    check_type(integer_types, min_side, "min_side")
    check_type(integer_types, min_dims, "min_dims")

    if max_dims is None:
        max_dims = min(32 - sig_dims, max(len(base_shape), min_dims) + 2)
    else:
        check_type(integer_types, max_dims, "max_dims")

    if max_side is None:
        max_side = max(tuple(base_shape[-max_dims:]) + (min_side,)) + 2
    else:
        check_type(integer_types, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if 32 - sig_dims < max_dims:
        if sig_dims == 0:
            raise InvalidArgument("max_dims cannot exceed 32")
        raise InvalidArgument(
            "max_dims=%r would exceed the 32-dimension limit given signature=%r"
            % (signature, parsed_signature)
        )

    dims, bnd_name = (max_dims, "max_dims") if strict_check else (min_dims, "min_dims")

    # check for unsatisfiable min_side
    if not all(min_side <= s for s in base_shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            "Given base_shape=%r, there are no broadcast-compatible "
            "shapes that satisfy: %s=%s and min_side=%s"
            % (base_shape, bnd_name, dims, min_side)
        )

    # check for unsatisfiable [min_side, max_side]
    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in base_shape[::-1][:dims])
    ):
        raise InvalidArgument(
            "Given base_shape=%r, there are no broadcast-compatible shapes "
            "that satisfy all of %s=%s, min_side=%s, and max_side=%s"
            % (base_shape, bnd_name, dims, min_side, max_side)
        )

    if not strict_check:
        # reduce max_dims to exclude unsatisfiable dimensions
        for n, s in zip(range(max_dims), reversed(base_shape)):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=num_shapes,
        signature=parsed_signature,
        base_shape=base_shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    )


class BasicIndexStrategy(SearchStrategy):
    def __init__(self, shape, min_dims, max_dims, allow_ellipsis, allow_newaxis):
        assert 0 <= min_dims <= max_dims <= 32
        SearchStrategy.__init__(self)
        self.shape = shape
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.allow_ellipsis = allow_ellipsis
        self.allow_newaxis = allow_newaxis

    def do_draw(self, data):
        # General plan: determine the actual selection up front with a straightforward
        # approach that shrinks well, then complicate it by inserting other things.
        result = []
        for dim_size in self.shape:
            if dim_size == 0:
                result.append(slice(None))
                continue
            strategy = st.integers(-dim_size, dim_size - 1) | st.slices(dim_size)
            result.append(data.draw(strategy))
        # Insert some number of new size-one dimensions if allowed
        result_dims = sum(isinstance(idx, slice) for idx in result)
        while (
            self.allow_newaxis
            and result_dims < self.max_dims
            and (result_dims < self.min_dims or data.draw(st.booleans()))
        ):
            result.insert(data.draw(st.integers(0, len(result))), np.newaxis)
            result_dims += 1
        # Check that we'll have the right number of dimensions; reject if not.
        # It's easy to do this by construction iff you don't care about shrinking,
        # which is really important for array shapes.  So we filter instead.
        assume(self.min_dims <= result_dims <= self.max_dims)
        # This is a quick-and-dirty way to insert ..., xor shorten the indexer,
        # but it means we don't have to do any structural analysis.
        if self.allow_ellipsis and data.draw(st.booleans()):
            # Choose an index; then replace all adjacent whole-dimension slices.
            i = j = data.draw(st.integers(0, len(result)))
            while i > 0 and result[i - 1] == slice(None):
                i -= 1
            while j < len(result) and result[j] == slice(None):
                j += 1
            result[i:j] = [Ellipsis]
        else:
            while result[-1:] == [slice(None, None)] and data.draw(st.integers(0, 7)):
                result.pop()
        return tuple(result)


@st.defines_strategy
@reserved_means_kwonly_star
def basic_indices(
    shape,
    __reserved=not_set,
    min_dims=0,
    max_dims=None,
    allow_newaxis=False,
    allow_ellipsis=True,
):
    # type: (Shape, Any, int, int, bool, bool) -> st.SearchStrategy[BasicIndex]
    """
    The ``basic_indices`` strategy generates `basic indexes
    <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`__  for
    arrays of the specified shape, which may include dimensions of size zero.

    It generates tuples containing some mix of integers, :obj:`python:slice` objects,
    ``...`` (Ellipsis), and :obj:`numpy:numpy.newaxis`; which when used to index a
    ``shape``-shaped array will produce either a scalar or a shared-memory view.

    * ``shape``: the array shape that will be indexed, as a tuple of integers >= 0.
      This must be at least two-dimensional for a tuple to be a valid basic index;
      for one-dimensional arrays use :func:`~hypothesis.strategies.slices` instead.
    * ``min_dims``: the minimum dimensionality of the resulting view from use of
      the generated index.  When ``min_dims == 0``, scalars and zero-dimensional
      arrays are both allowed.
    * ``max_dims``: the maximum dimensionality of the resulting view.
      If not specified, it defaults to ``max(len(shape), min_dims) + 2``.
    * ``allow_ellipsis``: whether ``...``` is allowed in the index.
    * ``allow_newaxis``: whether :obj:`numpy:numpy.newaxis` is allowed in the index.

    Note that the length of the generated tuple may be anywhere between zero
    and ``min_dims``.  It may not match the length of ``shape``, or even the
    dimensionality of the array view resulting from its use!
    """
    # Arguments to exclude scalars, zero-dim arrays, and dims of size zero were
    # all considered and rejected.  We want users to explicitly consider those
    # cases if they're dealing in general indexers, and while it's fiddly we can
    # back-compatibly add them later (hence using __reserved to sim kwonlyargs).
    check_type(tuple, shape, "shape")
    if __reserved is not not_set:
        raise InvalidArgument("Do not pass the __reserved argument.")
    check_type(bool, allow_ellipsis, "allow_ellipsis")
    check_type(bool, allow_newaxis, "allow_newaxis")
    check_type(integer_types, min_dims, "min_dims")
    if max_dims is None:
        max_dims = min(max(len(shape), min_dims) + 2, 32)
    else:
        check_type(integer_types, max_dims, "max_dims")
    order_check("dims", 0, min_dims, max_dims)
    check_argument(
        max_dims <= 32,
        "max_dims=%r, but numpy arrays have at most 32 dimensions" % (max_dims,),
    )
    check_argument(
        all(isinstance(x, integer_types) and x >= 0 for x in shape),
        "shape=%r, but all dimensions must be of integer size >= 0" % (shape,),
    )
    return BasicIndexStrategy(
        shape,
        min_dims=min_dims,
        max_dims=max_dims,
        allow_ellipsis=allow_ellipsis,
        allow_newaxis=allow_newaxis,
    )


@st.defines_strategy
def integer_array_indices(shape, result_shape=array_shapes(), dtype="int"):
    # type: (Shape, SearchStrategy[Shape], np.dtype) -> st.SearchStrategy[Tuple[np.ndarray, ...]]
    """Return a search strategy for tuples of integer-arrays that, when used
    to index into an array of shape ``shape``, given an array whose shape
    was drawn from ``result_shape``.

    Examples from this strategy shrink towards the tuple of index-arrays::

        len(shape) * (np.zeros(drawn_result_shape, dtype), )

    * ``shape`` a tuple of integers that indicates the shape of the array,
      whose indices are being generated.
    * ``result_shape`` a strategy for generating tuples of integers, which
      describe the shape of the resulting index arrays. The default is
      :func:`~hypothesis.extra.numpy.array_shapes`.  The shape drawn from
      this strategy determines the shape of the array that will be produced
      when the corresponding example from ``integer_array_indices`` is used
      as an index.
    * ``dtype`` the integer data type of the generated index-arrays. Negative
      integer indices can be generated if a signed integer type is specified.

    Recall that an array can be indexed using a tuple of integer-arrays to
    access its members in an arbitrary order, producing an array with an
    arbitrary shape. For example:

    .. code-block:: pycon

        >>> from numpy import array
        >>> x = array([-0, -1, -2, -3, -4])
        >>> ind = (array([[4, 0], [0, 1]]),)  # a tuple containing a 2D integer-array
        >>> x[ind]  # the resulting array is commensurate with the indexing array(s)
        array([[-4,  0],
               [0, -1]])

    Note that this strategy does not accommodate all variations of so-called
    'advanced indexing', as prescribed by NumPy's nomenclature.  Combinations
    of basic and advanced indexes are too complex to usefully define in a
    standard strategy; we leave application-specific strategies to the user.
    Advanced-boolean indexing can be defined as ``arrays(shape=..., dtype=bool)``,
    and is similarly left to the user.
    """
    check_type(tuple, shape, "shape")
    check_argument(
        shape and all(isinstance(x, integer_types) and x > 0 for x in shape),
        "shape=%r must be a non-empty tuple of integers > 0" % (shape,),
    )
    check_type(SearchStrategy, result_shape, "result_shape")
    check_argument(
        np.issubdtype(dtype, np.integer), "dtype=%r must be an integer dtype" % (dtype,)
    )
    signed = np.issubdtype(dtype, np.signedinteger)

    def array_for(index_shape, size):
        return arrays(
            dtype=dtype,
            shape=index_shape,
            elements=st.integers(-size if signed else 0, size - 1),
        )

    return result_shape.flatmap(
        lambda index_shape: st.tuples(*[array_for(index_shape, size) for size in shape])
    )
