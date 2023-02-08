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

import datetime as dt
import enum
import math
import operator
import string
import sys
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import reduce
from inspect import isabstract, isclass
from uuid import UUID

import attr

from hypothesis._settings import note_deprecation
from hypothesis.control import cleanup, note, reject
from hypothesis.errors import InvalidArgument, ResolutionFailed
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.cathetus import cathetus
from hypothesis.internal.charmap import as_general_categories
from hypothesis.internal.compat import (
    ceil,
    floor,
    gcd,
    get_type_hints,
    getfullargspec,
    hrange,
    implements_iterator,
    string_types,
    typing_root_type,
)
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls,
    check_sample,
    integer_range,
)
from hypothesis.internal.entropy import get_seeder_and_restorer
from hypothesis.internal.floats import (
    count_between_floats,
    float_of,
    float_to_int,
    int_to_float,
    is_negative,
    next_down,
    next_up,
)
from hypothesis.internal.reflection import (
    define_function_signature,
    is_typed_named_tuple,
    nicerepr,
    proxies,
    required_args,
    reserved_means_kwonly_star,
)
from hypothesis.internal.validation import (
    check_type,
    check_valid_bound,
    check_valid_integer,
    check_valid_interval,
    check_valid_magnitude,
    check_valid_size,
    check_valid_sizes,
    try_convert,
)
from hypothesis.strategies._internal import SearchStrategy, check_strategy
from hypothesis.strategies._internal.collections import (
    FixedAndOptionalKeysDictStrategy,
    FixedKeysDictStrategy,
    ListStrategy,
    TupleStrategy,
    UniqueListStrategy,
    UniqueSampledListStrategy,
)
from hypothesis.strategies._internal.datetime import (
    DateStrategy,
    DatetimeStrategy,
    TimedeltaStrategy,
)
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.functions import FunctionStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy
from hypothesis.strategies._internal.misc import JustStrategy
from hypothesis.strategies._internal.numbers import (
    BoundedIntStrategy,
    FixedBoundedFloatStrategy,
    FloatStrategy,
    WideRangeIntStrategy,
)
from hypothesis.strategies._internal.recursive import RecursiveStrategy
from hypothesis.strategies._internal.shared import SharedStrategy
from hypothesis.strategies._internal.strategies import (
    OneOfStrategy,
    SampledFromStrategy,
)
from hypothesis.strategies._internal.strings import (
    BinaryStringStrategy,
    FixedSizeBytes,
    OneCharStringStrategy,
    StringStrategy,
)
from hypothesis.types import RandomWithSeed
from hypothesis.utils.conventions import infer, not_set

typing = None  # type: Union[None, ModuleType]

try:
    import typing as typing_module

    typing = typing_module
except ImportError:
    pass

if False:
    import random  # noqa
    from types import ModuleType  # noqa
    from typing import Any, Dict, Union, Sequence, Callable, Pattern  # noqa
    from typing import TypeVar, Tuple, Iterable, List, Set, FrozenSet, overload  # noqa
    from typing import Type, Text, AnyStr, Optional  # noqa

    from hypothesis.utils.conventions import InferType  # noqa
    from hypothesis.strategies._internal.strategies import T, Ex  # noqa

    K, V = TypeVar["K"], TypeVar["V"]
    # See https://github.com/python/mypy/issues/3186 - numbers.Real is wrong!
    Real = Union[int, float, Fraction, Decimal]
else:

    def overload(f):
        return f


_strategies = set()


class FloatKey(object):
    def __init__(self, f):
        self.value = float_to_int(f)

    def __eq__(self, other):
        return isinstance(other, FloatKey) and (other.value == self.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.value)


def convert_value(v):
    if isinstance(v, float):
        return FloatKey(v)
    return (type(v), v)


STRATEGY_CACHE = LRUReusedCache(1024)


def cacheable(fn):
    # type: (T) -> T
    @proxies(fn)
    def cached_strategy(*args, **kwargs):
        try:
            kwargs_cache_key = {(k, convert_value(v)) for k, v in kwargs.items()}
        except TypeError:
            return fn(*args, **kwargs)
        cache_key = (fn, tuple(map(convert_value, args)), frozenset(kwargs_cache_key))
        try:
            if cache_key in STRATEGY_CACHE:
                return STRATEGY_CACHE[cache_key]
        except TypeError:
            return fn(*args, **kwargs)
        else:
            result = fn(*args, **kwargs)
            if not isinstance(result, SearchStrategy) or result.is_cacheable:
                STRATEGY_CACHE[cache_key] = result
            return result

    cached_strategy.__clear_cache = STRATEGY_CACHE.clear
    return cached_strategy


def base_defines_strategy(force_reusable):
    # type: (bool) -> Callable[[T], T]
    """Returns a decorator for strategy functions.

    If force_reusable is True, the generated values are assumed to be
    reusable, i.e. immutable and safe to cache, across multiple test
    invocations.
    """

    def decorator(strategy_definition):
        """A decorator that registers the function as a strategy and makes it
        lazily evaluated."""
        _strategies.add(strategy_definition.__name__)

        @proxies(strategy_definition)
        def accept(*args, **kwargs):
            result = LazyStrategy(strategy_definition, args, kwargs)
            if force_reusable:
                result.force_has_reusable_values = True
                assert result.has_reusable_values
            return result

        accept.is_hypothesis_strategy_function = True
        return accept

    return decorator


defines_strategy = base_defines_strategy(False)
defines_strategy_with_reusable_values = base_defines_strategy(True)


class Nothing(SearchStrategy):
    def calc_is_empty(self, recur):
        return True

    def do_draw(self, data):
        # This method should never be called because draw() will mark the
        # data as invalid immediately because is_empty is True.
        raise NotImplementedError("This should never happen")  # pragma: no cover

    def calc_has_reusable_values(self, recur):
        return True

    def __repr__(self):
        return "nothing()"

    def map(self, f):
        return self

    def filter(self, f):
        return self

    def flatmap(self, f):
        return self


NOTHING = Nothing()


@cacheable
def nothing():
    # type: () -> SearchStrategy
    """This strategy never successfully draws a value and will always reject on
    an attempt to draw.

    Examples from this strategy do not shrink (because there are none).
    """
    return NOTHING


def just(value):
    # type: (T) -> SearchStrategy[T]
    """Return a strategy which only generates ``value``.

    Note: ``value`` is not copied. Be wary of using mutable values.

    If ``value`` is the result of a callable, you can use
    :func:`builds(callable) <hypothesis.strategies.builds>` instead
    of ``just(callable())`` to get a fresh value each time.

    Examples from this strategy do not shrink (because there is only one).
    """
    return JustStrategy(value)


@defines_strategy_with_reusable_values
def none():
    # type: () -> SearchStrategy[None]
    """Return a strategy which only generates None.

    Examples from this strategy do not shrink (because there is only
    one).
    """
    return just(None)


@overload
def one_of(args):
    # type: (Sequence[SearchStrategy[Any]]) -> SearchStrategy[Any]
    pass  # pragma: no cover


@overload  # noqa: F811
def one_of(*args):
    # type: (SearchStrategy[Any]) -> SearchStrategy[Any]
    pass  # pragma: no cover


def one_of(*args):  # noqa: F811
    # Mypy workaround alert:  Any is too loose above; the return parameter
    # should be the union of the input parameters.  Unfortunately, Mypy <=0.600
    # raises errors due to incompatible inputs instead.  See #1270 for links.
    # v0.610 doesn't error; it gets inference wrong for 2+ arguments instead.
    """Return a strategy which generates values from any of the argument
    strategies.

    This may be called with one iterable argument instead of multiple
    strategy arguments, in which case ``one_of(x)`` and ``one_of(*x)`` are
    equivalent.

    Examples from this strategy will generally shrink to ones that come from
    strategies earlier in the list, then shrink according to behaviour of the
    strategy that produced them. In order to get good shrinking behaviour,
    try to put simpler strategies first. e.g. ``one_of(none(), text())`` is
    better than ``one_of(text(), none())``.

    This is especially important when using recursive strategies. e.g.
    ``x = st.deferred(lambda: st.none() | st.tuples(x, x))`` will shrink well,
    but ``x = st.deferred(lambda: st.tuples(x, x) | st.none())`` will shrink
    very badly indeed.
    """
    if len(args) == 1 and not isinstance(args[0], SearchStrategy):
        try:
            args = tuple(args[0])
        except TypeError:
            pass
    return OneOfStrategy(args)


@cacheable
@defines_strategy_with_reusable_values
def integers(min_value=None, max_value=None):
    # type: (int, int) -> SearchStrategy[int]
    """Returns a strategy which generates integers; in Python 2 these may be
    ints or longs.

    If min_value is not None then all values will be >= min_value. If
    max_value is not None then all values will be <= max_value

    Examples from this strategy will shrink towards zero, and negative values
    will also shrink towards positive (i.e. -n may be replaced by +n).
    """

    check_valid_bound(min_value, "min_value")
    check_valid_bound(max_value, "max_value")
    check_valid_interval(min_value, max_value, "min_value", "max_value")

    min_int_value = None if min_value is None else ceil(min_value)
    max_int_value = None if max_value is None else floor(max_value)

    if min_value != min_int_value:
        note_deprecation(
            "min_value=%r of type %r cannot be exactly represented as an "
            "integer, which will be an error in a future version.  "
            "Use %r instead." % (min_value, type(min_value), min_int_value),
            since="2018-10-10",
        )
    if max_value != max_int_value:
        note_deprecation(
            "max_value=%r of type %r cannot be exactly represented as an "
            "integer, which will be an error in a future version.  "
            "Use %r instead." % (max_value, type(max_value), max_int_value),
            since="2018-10-10",
        )

    if (
        min_int_value is not None
        and max_int_value is not None
        and min_int_value > max_int_value
    ):
        raise InvalidArgument(
            "No integers between min_value=%r and "
            "max_value=%r" % (min_value, max_value)
        )

    if min_int_value is None:
        if max_int_value is None:
            return WideRangeIntStrategy()
        else:
            if max_int_value > 0:
                return WideRangeIntStrategy().filter(lambda x: x <= max_int_value)
            return WideRangeIntStrategy().map(lambda x: max_int_value - abs(x))
    else:
        if max_int_value is None:
            if min_int_value < 0:
                return WideRangeIntStrategy().filter(lambda x: x >= min_int_value)
            return WideRangeIntStrategy().map(lambda x: min_int_value + abs(x))
        else:
            assert min_int_value <= max_int_value
            if min_int_value == max_int_value:
                return just(min_int_value)
            elif min_int_value >= 0:
                return BoundedIntStrategy(min_int_value, max_int_value)
            elif max_int_value <= 0:
                return BoundedIntStrategy(-max_int_value, -min_int_value).map(
                    lambda t: -t
                )
            else:
                return integers(min_value=0, max_value=max_int_value) | integers(
                    min_value=min_int_value, max_value=0
                )


@cacheable
@defines_strategy
def booleans():
    # type: () -> SearchStrategy[bool]
    """Returns a strategy which generates instances of :class:`python:bool`.

    Examples from this strategy will shrink towards False (i.e.
    shrinking will try to replace True with False where possible).
    """
    return sampled_from([False, True])


@cacheable
@defines_strategy_with_reusable_values
def floats(
    min_value=None,  # type: Real
    max_value=None,  # type: Real
    allow_nan=None,  # type: bool
    allow_infinity=None,  # type: bool
    width=64,  # type: int
    exclude_min=False,  # type: bool
    exclude_max=False,  # type: bool
):
    # type: (...) -> SearchStrategy[float]
    """Returns a strategy which generates floats.

    - If min_value is not None, all values will be ``>= min_value``
      (or ``> min_value`` if ``exclude_min``).
    - If max_value is not None, all values will be ``<= max_value``
      (or ``< max_value`` if ``exclude_max``).
    - If min_value or max_value is not None, it is an error to enable
      allow_nan.
    - If both min_value and max_value are not None, it is an error to enable
      allow_infinity.

    Where not explicitly ruled out by the bounds, all of infinity, -infinity
    and NaN are possible values generated by this strategy.

    The width argument specifies the maximum number of bits of precision
    required to represent the generated float. Valid values are 16, 32, or 64.
    Passing ``width=32`` will still use the builtin 64-bit ``float`` class,
    but always for values which can be exactly represented as a 32-bit float.
    Half-precision floats (``width=16``) are only supported on Python 3.6, or
    if :pypi:`Numpy` is installed.

    The exclude_min and exclude_max argument can be used to generate numbers
    from open or half-open intervals, by excluding the respective endpoints.
    Excluding either signed zero will also exclude the other.
    Attempting to exclude an endpoint which is None will raise an error;
    use ``allow_infinity=False`` to generate finite floats.  You can however
    use e.g. ``min_value=float("-inf"), exclude_min=True`` to exclude only
    one infinite endpoint.

    Examples from this strategy have a complicated and hard to explain
    shrinking behaviour, but it tries to improve "human readability". Finite
    numbers will be preferred to infinity and infinity will be preferred to
    NaN.
    """
    check_type(bool, exclude_min, "exclude_min")
    check_type(bool, exclude_max, "exclude_max")

    if allow_nan is None:
        allow_nan = bool(min_value is None and max_value is None)
    elif allow_nan:
        if min_value is not None or max_value is not None:
            raise InvalidArgument(
                "Cannot have allow_nan=%r, with min_value or max_value" % (allow_nan)
            )

    if width not in (16, 32, 64):
        raise InvalidArgument(
            "Got width=%r, but the only valid values are the integers 16, "
            "32, and 64." % (width,)
        )
    if width == 16 and sys.version_info[:2] < (3, 6) and "numpy" not in sys.modules:
        raise InvalidArgument(  # pragma: no cover
            "width=16 requires either Numpy, or Python >= 3.6"
        )

    check_valid_bound(min_value, "min_value")
    check_valid_bound(max_value, "max_value")

    min_arg, max_arg = min_value, max_value
    if min_value is not None:
        min_value = float_of(min_value, width)
        assert isinstance(min_value, float)
    if max_value is not None:
        max_value = float_of(max_value, width)
        assert isinstance(max_value, float)

    if min_value != min_arg:
        note_deprecation(
            "min_value=%r cannot be exactly represented as a float of width "
            "%d, which will be an error in a future version. Use min_value=%r "
            "instead." % (min_arg, width, min_value),
            since="2018-10-10",
        )
    if max_value != max_arg:
        note_deprecation(
            "max_value=%r cannot be exactly represented as a float of width "
            "%d, which will be an error in a future version. Use max_value=%r "
            "instead" % (max_arg, width, max_value),
            since="2018-10-10",
        )

    if exclude_min and (min_value is None or min_value == float("inf")):
        raise InvalidArgument("Cannot exclude min_value=%r" % (min_value,))
    if exclude_max and (max_value is None or max_value == float("-inf")):
        raise InvalidArgument("Cannot exclude max_value=%r" % (max_value,))

    if min_value is not None and (
        exclude_min or (min_arg is not None and min_value < min_arg)
    ):
        min_value = next_up(min_value, width)
        if min_value == min_arg:
            assert min_value == min_arg == 0
            assert is_negative(min_arg) and not is_negative(min_value)
            min_value = next_up(min_value, width)
        assert min_value > min_arg  # type: ignore
    if max_value is not None and (
        exclude_max or (max_arg is not None and max_value > max_arg)
    ):
        max_value = next_down(max_value, width)
        if max_value == max_arg:
            assert max_value == max_arg == 0
            assert is_negative(max_value) and not is_negative(max_arg)
            max_value = next_down(max_value, width)
        assert max_value < max_arg  # type: ignore

    if min_value == float(u"-inf"):
        min_value = None
    if max_value == float(u"inf"):
        max_value = None

    bad_zero_bounds = (
        min_value == max_value == 0
        and is_negative(max_value)
        and not is_negative(min_value)
    )
    if (
        min_value is not None
        and max_value is not None
        and (min_value > max_value or bad_zero_bounds)
    ):
        # This is a custom alternative to check_valid_interval, because we want
        # to include the bit-width and exclusion information in the message.
        msg = (
            "There are no %s-bit floating-point values between min_value=%r "
            "and max_value=%r" % (width, min_arg, max_arg)
        )
        if exclude_min or exclude_max:
            msg += ", exclude_min=%r and exclude_max=%r" % (exclude_min, exclude_max)
        if bad_zero_bounds:
            note_deprecation(msg, since="2019-03-19")
        else:
            raise InvalidArgument(msg)

    if allow_infinity is None:
        allow_infinity = bool(min_value is None or max_value is None)
    elif allow_infinity:
        if min_value is not None and max_value is not None:
            raise InvalidArgument(
                "Cannot have allow_infinity=%r, with both min_value and "
                "max_value" % (allow_infinity)
            )
    elif min_value == float("inf"):
        raise InvalidArgument("allow_infinity=False excludes min_value=inf")
    elif max_value == float("-inf"):
        raise InvalidArgument("allow_infinity=False excludes max_value=-inf")

    unbounded_floats = FloatStrategy(
        allow_infinity=allow_infinity, allow_nan=allow_nan, width=width
    )

    if min_value is None and max_value is None:
        return unbounded_floats
    elif min_value is not None and max_value is not None:
        if min_value == max_value:
            assert isinstance(min_value, float)
            result = just(min_value)
        elif is_negative(min_value):
            if is_negative(max_value):
                return floats(
                    min_value=-max_value, max_value=-min_value, width=width
                ).map(operator.neg)
            else:
                return one_of(
                    floats(min_value=0.0, max_value=max_value, width=width),
                    floats(min_value=0.0, max_value=-min_value, width=width).map(
                        operator.neg
                    ),
                )
        elif count_between_floats(min_value, max_value) > 1000:
            return FixedBoundedFloatStrategy(
                lower_bound=min_value, upper_bound=max_value, width=width
            )
        else:
            ub_int = float_to_int(max_value, width)
            lb_int = float_to_int(min_value, width)
            assert lb_int <= ub_int
            result = integers(min_value=lb_int, max_value=ub_int).map(
                lambda x: int_to_float(x, width)
            )
    elif min_value is not None:
        assert isinstance(min_value, float)
        if is_negative(min_value):
            return one_of(
                unbounded_floats.map(abs),
                floats(min_value=min_value, max_value=-0.0, width=width),
            )
        else:
            result = unbounded_floats.map(lambda x: min_value + abs(x))
    else:
        assert isinstance(max_value, float)
        if not is_negative(max_value):
            return one_of(
                floats(min_value=0.0, max_value=max_value, width=width),
                unbounded_floats.map(lambda x: -abs(x)),
            )
        else:
            result = unbounded_floats.map(lambda x: max_value - abs(x))

    if width < 64:

        def downcast(x):
            try:
                return float_of(x, width)
            except OverflowError:  # pragma: no cover
                reject()

        result = result.map(downcast)
    if not allow_infinity:
        result = result.filter(lambda x: not math.isinf(x))
    return result


@cacheable
@defines_strategy
def tuples(*args):
    # type: (*SearchStrategy) -> SearchStrategy[tuple]
    """Return a strategy which generates a tuple of the same length as args by
    generating the value at index i from args[i].

    e.g. tuples(integers(), integers()) would generate a tuple of length
    two with both values an integer.

    Examples from this strategy shrink by shrinking their component parts.
    """
    for arg in args:
        check_strategy(arg)

    return TupleStrategy(args)


@overload
def sampled_from(elements):
    # type: (Sequence[T]) -> SearchStrategy[T]
    pass  # pragma: no cover


@overload  # noqa: F811
def sampled_from(elements):
    # type: (Type[enum.Enum]) -> SearchStrategy[Any]
    # `SearchStrategy[Enum]` is unreliable due to metaclass issues.
    pass  # pragma: no cover


@defines_strategy  # noqa: F811
def sampled_from(elements):
    """Returns a strategy which generates any value present in ``elements``.

    Note that as with :func:`~hypothesis.strategies.just`, values will not be
    copied and thus you should be careful of using mutable data.

    ``sampled_from`` supports ordered collections, as well as
    :class:`~python:enum.Enum` objects.  :class:`~python:enum.Flag` objects
    may also generate any combination of their members.

    Examples from this strategy shrink by replacing them with values earlier in
    the list. So e.g. sampled_from((10, 1)) will shrink by trying to replace
    1 values with 10, and sampled_from((1, 10)) will shrink by trying to
    replace 10 values with 1.
    """
    values = check_sample(elements, "sampled_from")
    if not values:
        note_deprecation(
            "sampled_from() with nothing to sample is deprecated and will be an "
            "error in a future version.  It currently returns `st.nothing()`, "
            "which if unexpected can make parts of a strategy silently vanish.",
            since="2019-03-12",
        )
        return nothing()
    if len(values) == 1:
        return just(values[0])
    if hasattr(enum, "Flag") and isclass(elements) and issubclass(elements, enum.Flag):
        # Combinations of enum.Flag members are also members.  We generate
        # these dynamically, because static allocation takes O(2^n) memory.
        return sets(sampled_from(values), min_size=1).map(
            lambda s: reduce(operator.or_, s)
        )
    return SampledFromStrategy(values)


@cacheable
@defines_strategy
def lists(
    elements,  # type: SearchStrategy[Ex]
    min_size=0,  # type: int
    max_size=None,  # type: int
    unique_by=None,  # type: Union[Callable, Tuple[Callable, ...]]
    unique=False,  # type: bool
):
    # type: (...) -> SearchStrategy[List[Ex]]
    """Returns a list containing values drawn from elements with length in the
    interval [min_size, max_size] (no bounds in that direction if these are
    None). If max_size is 0, only the empty list will be drawn.

    If ``unique`` is True (or something that evaluates to True), we compare direct
    object equality, as if unique_by was ``lambda x: x``. This comparison only
    works for hashable types.

    If ``unique_by`` is not None it must be a callable or tuple of callables
    returning a hashable type when given a value drawn from elements. The
    resulting list will satisfy the condition that for ``i`` != ``j``,
    ``unique_by(result[i])`` != ``unique_by(result[j])``.

    If ``unique_by`` is a tuple of callables the uniqueness will be respective
    to each callable.

    For example, the following will produce two columns of integers with both
    columns being unique respectively.
    .. code-block:: pycon

        >>> twoints = st.tuples(st.integers(), st.integers())
        >>> st.lists(twoints, unique_by=(lambda x: x[0], lambda x: x[1]))

    Examples from this strategy shrink by trying to remove elements from the
    list, and by shrinking each individual element of the list.
    """
    check_valid_sizes(min_size, max_size)
    check_strategy(elements, "elements")
    if unique:
        if unique_by is not None:
            raise InvalidArgument(
                "cannot specify both unique and unique_by "
                "(you probably only want to set unique_by)"
            )
        else:

            def unique_by(x):
                return x

    if max_size == 0:
        return builds(list)
    if unique_by is not None:
        if not (callable(unique_by) or isinstance(unique_by, tuple)):
            raise InvalidArgument(
                "unique_by=%r is not a callable or tuple of callables" % (unique_by)
            )
        if callable(unique_by):
            unique_by = (unique_by,)
        if len(unique_by) == 0:
            raise InvalidArgument("unique_by is empty")
        for i, f in enumerate(unique_by):
            if not callable(f):
                raise InvalidArgument("unique_by[%i]=%r is not a callable" % (i, f))
        # Note that lazy strategies automatically unwrap when passed to a defines_strategy
        # function.
        if isinstance(elements, SampledFromStrategy):
            element_count = len(elements.elements)
            if min_size > element_count:
                raise InvalidArgument(
                    "Cannot create a collection of min_size=%r unique elements with "
                    "values drawn from only %d distinct elements"
                    % (min_size, element_count)
                )

            if max_size is not None:
                max_size = min(max_size, element_count)
            else:
                max_size = element_count

            return UniqueSampledListStrategy(
                elements=elements, max_size=max_size, min_size=min_size, keys=unique_by
            )

        return UniqueListStrategy(
            elements=elements, max_size=max_size, min_size=min_size, keys=unique_by
        )
    return ListStrategy(elements, min_size=min_size, max_size=max_size)


@cacheable
@defines_strategy
def sets(
    elements,  # type: SearchStrategy[Ex]
    min_size=0,  # type: int
    max_size=None,  # type: int
):
    # type: (...) -> SearchStrategy[Set[Ex]]
    """This has the same behaviour as lists, but returns sets instead.

    Note that Hypothesis cannot tell if values are drawn from elements
    are hashable until running the test, so you can define a strategy
    for sets of an unhashable type but it will fail at test time.

    Examples from this strategy shrink by trying to remove elements from the
    set, and by shrinking each individual element of the set.
    """
    return lists(
        elements=elements, min_size=min_size, max_size=max_size, unique=True
    ).map(set)


@cacheable
@defines_strategy
def frozensets(
    elements,  # type: SearchStrategy[Ex]
    min_size=0,  # type: int
    max_size=None,  # type: int
):
    # type: (...) -> SearchStrategy[FrozenSet[Ex]]
    """This is identical to the sets function but instead returns
    frozensets."""
    return lists(
        elements=elements, min_size=min_size, max_size=max_size, unique=True
    ).map(frozenset)


@implements_iterator
class PrettyIter(object):
    def __init__(self, values):
        self._values = values
        self._iter = iter(self._values)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def __repr__(self):
        return "iter({!r})".format(self._values)


@defines_strategy
def iterables(
    elements,  # type: SearchStrategy[Ex]
    min_size=0,  # type: int
    max_size=None,  # type: int
    unique_by=None,  # type: Union[Callable, Tuple[Callable, ...]]
    unique=False,  # type: bool
):
    # type: (...) -> SearchStrategy[Iterable[Ex]]
    """This has the same behaviour as lists, but returns iterables instead.

    Some iterables cannot be indexed (e.g. sets) and some do not have a
    fixed length (e.g. generators). This strategy produces iterators,
    which cannot be indexed and do not have a fixed length. This ensures
    that you do not accidentally depend on sequence behaviour.
    """
    return lists(
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=unique_by,
        unique=unique,
    ).map(PrettyIter)


@defines_strategy
@reserved_means_kwonly_star
def fixed_dictionaries(
    mapping,  # type: Dict[T, SearchStrategy[Ex]]
    __reserved=not_set,  # type: Any
    optional=None,  # type: Dict[T, SearchStrategy[Ex]]
):
    # type: (...) -> SearchStrategy[Dict[T, Ex]]
    """Generates a dictionary of the same type as mapping with a fixed set of
    keys mapping to strategies. mapping must be a dict subclass.

    Generated values have all keys present in mapping, with the
    corresponding values drawn from mapping[key]. If mapping is an
    instance of OrderedDict the keys will also be in the same order,
    otherwise the order is arbitrary.

    Examples from this strategy shrink by shrinking each individual value in
    the generated dictionary.
    """
    check_type(dict, mapping, "mapping")
    for k, v in mapping.items():
        check_strategy(v, "mapping[%r]" % (k,))
    if __reserved is not not_set:
        raise InvalidArgument("Do not pass __reserved; got %r" % (__reserved,))
    if optional is not None:
        check_type(dict, optional, "optional")
        for k, v in optional.items():
            check_strategy(v, "optional[%r]" % (k,))
        if type(mapping) != type(optional):
            raise InvalidArgument(
                "Got arguments of different types: mapping=%s, optional=%s"
                % (nicerepr(type(mapping)), nicerepr(type(optional)))
            )
        if set(mapping) & set(optional):
            raise InvalidArgument(
                "The following keys were in both mapping and optional, "
                "which is invalid: %r" % (set(mapping) & set(optional))
            )
        return FixedAndOptionalKeysDictStrategy(mapping, optional)
    return FixedKeysDictStrategy(mapping)


@cacheable
@defines_strategy
def dictionaries(
    keys,  # type: SearchStrategy[Ex]
    values,  # type: SearchStrategy[T]
    dict_class=dict,  # type: type
    min_size=0,  # type: int
    max_size=None,  # type: int
):
    # type: (...) -> SearchStrategy[Dict[Ex, T]]
    # Describing the exact dict_class to Mypy drops the key and value types,
    # so we report Dict[K, V] instead of Mapping[Any, Any] for now.  Sorry!
    """Generates dictionaries of type ``dict_class`` with keys drawn from the ``keys``
    argument and values drawn from the ``values`` argument.

    The size parameters have the same interpretation as for
    :func:`~hypothesis.strategies.lists`.

    Examples from this strategy shrink by trying to remove keys from the
    generated dictionary, and by shrinking each generated key and value.
    """
    check_valid_sizes(min_size, max_size)
    if max_size == 0:
        return fixed_dictionaries(dict_class())
    check_strategy(keys)
    check_strategy(values)

    return lists(
        tuples(keys, values),
        min_size=min_size,
        max_size=max_size,
        unique_by=lambda x: x[0],
    ).map(dict_class)


@cacheable
@defines_strategy_with_reusable_values
def characters(
    whitelist_categories=None,  # type: Sequence[Text]
    blacklist_categories=None,  # type: Sequence[Text]
    blacklist_characters=None,  # type: Sequence[Text]
    min_codepoint=None,  # type: int
    max_codepoint=None,  # type: int
    whitelist_characters=None,  # type: Sequence[Text]
):
    # type: (...) -> SearchStrategy[Text]
    """Generates unicode text type (unicode on python 2, str on python 3)
    characters following specified filtering rules.

    - When no filtering rules are specified, any character can be produced.
    - If ``min_codepoint`` or ``max_codepoint`` is specified, then only
      characters having a codepoint in that range will be produced.
    - If ``whitelist_categories`` is specified, then only characters from those
      Unicode categories will be produced. This is a further restriction,
      characters must also satisfy ``min_codepoint`` and ``max_codepoint``.
    - If ``blacklist_categories`` is specified, then any character from those
      categories will not be produced.  Any overlap between
      ``whitelist_categories`` and ``blacklist_categories`` will raise an
      exception, as each character can only belong to a single class.
    - If ``whitelist_characters`` is specified, then any additional characters
      in that list will also be produced.
    - If ``blacklist_characters`` is specified, then any characters in
      that list will be not be produced. Any overlap between
      ``whitelist_characters`` and ``blacklist_characters`` will raise an
      exception.

    The ``_codepoint`` arguments must be integers between zero and
    :obj:`python:sys.maxunicode`.  The ``_characters`` arguments must be
    collections of length-one unicode strings, such as a unicode string.

    The ``_categories`` arguments must be used to specify either the
    one-letter Unicode major category or the two-letter Unicode
    `general category`_.  For example, ``('Nd', 'Lu')`` signifies "Number,
    decimal digit" and "Letter, uppercase".  A single letter ('major category')
    can be given to match all corresponding categories, for example ``'P'``
    for characters in any punctuation category.

    .. _general category: https://wikipedia.org/wiki/Unicode_character_property

    Examples from this strategy shrink towards the codepoint for ``'0'``,
    or the first allowable codepoint after it if ``'0'`` is excluded.
    """
    check_valid_size(min_codepoint, "min_codepoint")
    check_valid_size(max_codepoint, "max_codepoint")
    check_valid_interval(min_codepoint, max_codepoint, "min_codepoint", "max_codepoint")
    if (
        min_codepoint is None
        and max_codepoint is None
        and whitelist_categories is None
        and blacklist_categories is None
        and whitelist_characters is not None
    ):
        raise InvalidArgument(
            "Nothing is excluded by other arguments, so passing only "
            "whitelist_characters=%(chars)r would have no effect.  Also pass "
            "whitelist_categories=(), or use sampled_from(%(chars)r) instead."
            % {"chars": whitelist_characters}
        )
    blacklist_characters = blacklist_characters or ""
    whitelist_characters = whitelist_characters or ""
    overlap = set(blacklist_characters).intersection(whitelist_characters)
    if overlap:
        raise InvalidArgument(
            "Characters %r are present in both whitelist_characters=%r, and "
            "blacklist_characters=%r"
            % (sorted(overlap), whitelist_characters, blacklist_characters)
        )
    blacklist_categories = as_general_categories(
        blacklist_categories, "blacklist_categories"
    )
    if (
        whitelist_categories is not None
        and not whitelist_categories
        and not whitelist_characters
    ):
        raise InvalidArgument(
            "When whitelist_categories is an empty collection and there are "
            "no characters specified in whitelist_characters, nothing can "
            "be generated by the characters() strategy."
        )
    whitelist_categories = as_general_categories(
        whitelist_categories, "whitelist_categories"
    )
    both_cats = set(blacklist_categories or ()).intersection(whitelist_categories or ())
    if both_cats:
        raise InvalidArgument(
            "Categories %r are present in both whitelist_categories=%r, and "
            "blacklist_categories=%r"
            % (sorted(both_cats), whitelist_categories, blacklist_categories)
        )

    return OneCharStringStrategy(
        whitelist_categories=whitelist_categories,
        blacklist_categories=blacklist_categories,
        blacklist_characters=blacklist_characters,
        min_codepoint=min_codepoint,
        max_codepoint=max_codepoint,
        whitelist_characters=whitelist_characters,
    )


@cacheable
@defines_strategy_with_reusable_values
def text(
    alphabet=characters(
        blacklist_categories=("Cs",)
    ),  # type: Union[Sequence[Text], SearchStrategy[Text]]
    min_size=0,  # type: int
    max_size=None,  # type: int
):
    # type: (...) -> SearchStrategy[Text]
    """Generates values of a unicode text type (unicode on python 2, str on
    python 3) with values drawn from ``alphabet``, which should be an iterable of
    length one strings or a strategy generating such strings.

    The default alphabet strategy can generate the full unicode range but
    excludes surrogate characters because they are invalid in the UTF-8
    encoding.  You can use :func:`~hypothesis.strategies.characters` without
    arguments to find surrogate-related bugs such as :bpo:`34454`.

    ``min_size`` and ``max_size`` have the usual interpretations.
    Note that Python measures string length by counting codepoints: U+00C5
    ``Å`` is a single character, while U+0041 U+030A ``Å`` is two - the ``A``,
    and a combining ring above.

    Examples from this strategy shrink towards shorter strings, and with the
    characters in the text shrinking as per the alphabet strategy.
    This strategy does not :func:`~python:unicodedata.normalize` examples,
    so generated strings may be in any or none of the 'normal forms'.
    """
    check_valid_sizes(min_size, max_size)
    if alphabet is None:
        note_deprecation(
            "alphabet=None is deprecated; just omit the argument", since="2018-10-05"
        )
        char_strategy = characters(blacklist_categories=("Cs",))
    elif isinstance(alphabet, SearchStrategy):
        char_strategy = alphabet
    else:
        non_string = [c for c in alphabet if not isinstance(c, string_types)]
        if non_string:
            raise InvalidArgument(
                "The following elements in alphabet are not unicode "
                "strings:  %r" % (non_string,)
            )
        not_one_char = [c for c in alphabet if len(c) != 1]
        if not_one_char:
            raise InvalidArgument(
                "The following elements in alphabet are not of length "
                "one, which leads to violation of size constraints:  %r"
                % (not_one_char,)
            )
        char_strategy = (
            characters(whitelist_categories=(), whitelist_characters=alphabet)
            if alphabet
            else nothing()
        )
    if (max_size == 0 or char_strategy.is_empty) and not min_size:
        return just(u"")
    return StringStrategy(lists(char_strategy, min_size=min_size, max_size=max_size))


@cacheable
@defines_strategy
def from_regex(regex, fullmatch=False):
    # type: (Union[AnyStr, Pattern[AnyStr]], bool) -> SearchStrategy[AnyStr]
    r"""Generates strings that contain a match for the given regex (i.e. ones
    for which :func:`python:re.search` will return a non-None result).

    ``regex`` may be a pattern or :func:`compiled regex <python:re.compile>`.
    Both byte-strings and unicode strings are supported, and will generate
    examples of the same type.

    You can use regex flags such as :obj:`python:re.IGNORECASE` or
    :obj:`python:re.DOTALL` to control generation. Flags can be passed either
    in compiled regex or inside the pattern with a ``(?iLmsux)`` group.

    Some regular expressions are only partly supported - the underlying
    strategy checks local matching and relies on filtering to resolve
    context-dependent expressions.  Using too many of these constructs may
    cause health-check errors as too many examples are filtered out. This
    mainly includes (positive or negative) lookahead and lookbehind groups.

    If you want the generated string to match the whole regex you should use
    boundary markers. So e.g. ``r"\A.\Z"`` will return a single character
    string, while ``"."`` will return any string, and ``r"\A.$"`` will return
    a single character optionally followed by a ``"\n"``.
    Alternatively, passing ``fullmatch=True`` will ensure that the whole
    string is a match, as if you had used the ``\A`` and ``\Z`` markers.

    Examples from this strategy shrink towards shorter strings and lower
    character values, with exact behaviour that may depend on the pattern.
    """
    check_type(bool, fullmatch, "fullmatch")
    # TODO: We would like to move this to the top level, but pending some major
    # refactoring it's hard to do without creating circular imports.
    from hypothesis.strategies._internal.regex import regex_strategy

    return regex_strategy(regex, fullmatch)


@cacheable
@defines_strategy_with_reusable_values
def binary(min_size=0, max_size=None):
    # type: (int, int) -> SearchStrategy[bytes]
    """Generates the appropriate binary type (str in python 2, bytes in python
    3).

    min_size and max_size have the usual interpretations.

    Examples from this strategy shrink towards smaller strings and lower byte
    values.
    """
    check_valid_sizes(min_size, max_size)
    if min_size == max_size is not None:
        return FixedSizeBytes(min_size)
    return BinaryStringStrategy(
        lists(
            integers(min_value=0, max_value=255), min_size=min_size, max_size=max_size
        )
    )


@cacheable
@defines_strategy
def randoms():
    # type: () -> SearchStrategy[random.Random]
    """Generates instances of ``random.Random``, tweaked to show the seed value
    in the repr for reproducibility.

    Examples from this strategy shrink to seeds closer to zero.
    """
    return integers().map(RandomWithSeed)


class RandomSeeder(object):
    def __init__(self, seed):
        self.seed = seed

    def __repr__(self):
        return "RandomSeeder(%r)" % (self.seed,)


class RandomModule(SearchStrategy):
    def do_draw(self, data):
        seed = data.draw(integers(0, 2 ** 32 - 1))
        seed_all, restore_all = get_seeder_and_restorer(seed)
        seed_all()
        cleanup(restore_all)
        return RandomSeeder(seed)


@cacheable
@defines_strategy
def random_module():
    # type: () -> SearchStrategy[RandomSeeder]
    """The Hypothesis engine handles PRNG state for the stdlib and Numpy random
    modules internally, always seeding them to zero and restoring the previous
    state after the test.

    If having a fixed seed would unacceptably weaken your tests, and you
    cannot use a ``random.Random`` instance provided by
    :func:`~hypothesis.strategies.randoms`, this strategy calls
    :func:`python:random.seed` with an arbitrary integer and passes you
    an opaque object whose repr displays the seed value for debugging.
    If ``numpy.random`` is available, that state is also managed.

    Examples from these strategy shrink to seeds closer to zero.
    """
    return shared(RandomModule(), "hypothesis.strategies.random_module()")


@cacheable
@defines_strategy
def builds(
    *callable_and_args,  # type: Any
    **kwargs  # type: Union[SearchStrategy[Any], InferType]
):
    # type: (...) -> SearchStrategy[Any]
    """Generates values by drawing from ``args`` and ``kwargs`` and passing
    them to the callable (provided as the first positional argument) in the
    appropriate argument position.

    e.g. ``builds(target, integers(), flag=booleans())`` would draw an
    integer ``i`` and a boolean ``b`` and call ``target(i, flag=b)``.

    If the callable has type annotations, they will be used to infer a strategy
    for required arguments that were not passed to builds.  You can also tell
    builds to infer a strategy for an optional argument by passing the special
    value :const:`hypothesis.infer` as a keyword argument to
    builds, instead of a strategy for that argument to the callable.

    If the callable is a class defined with :pypi:`attrs`, missing required
    arguments will be inferred from the attribute on a best-effort basis,
    e.g. by checking :ref:`attrs standard validators <attrs:api_validators>`.
    Dataclasses are handled natively by the inference from type hints.

    Examples from this strategy shrink by shrinking the argument values to
    the callable.
    """
    if not callable_and_args:
        raise InvalidArgument(
            "builds() must be passed a callable as the first positional "
            "argument, but no positional arguments were given."
        )
    target, args = callable_and_args[0], callable_and_args[1:]
    if not callable(target):
        raise InvalidArgument(
            "The first positional argument to builds() must be a callable "
            "target to construct."
        )

    if infer in args:
        # Avoid an implementation nightmare juggling tuples and worse things
        raise InvalidArgument(
            "infer was passed as a positional argument to "
            "builds(), but is only allowed as a keyword arg"
        )
    required = required_args(target, args, kwargs) or set()
    to_infer = {k for k, v in kwargs.items() if v is infer}
    if required or to_infer:
        if isclass(target) and attr.has(target):
            # Use our custom introspection for attrs classes
            from hypothesis.strategies._internal.attrs import from_attrs

            return from_attrs(target, args, kwargs, required | to_infer)
        # Otherwise, try using type hints
        if isclass(target):
            if is_typed_named_tuple(target):
                # Special handling for typing.NamedTuple
                hints = target._field_types
            else:
                hints = get_type_hints(target.__init__)
        else:
            hints = get_type_hints(target)
        if to_infer - set(hints):
            raise InvalidArgument(
                "passed infer for %s, but there is no type annotation"
                % (", ".join(sorted(to_infer - set(hints))))
            )
        for kw in set(hints) & (required | to_infer):
            kwargs[kw] = from_type(hints[kw])
    # Mypy doesn't realise that `infer` is gone from kwargs now
    kwarg_strat = fixed_dictionaries(kwargs)  # type: ignore
    return tuples(tuples(*args), kwarg_strat).map(
        lambda value: target(*value[0], **value[1])
    )


def _defer_from_type(func):
    # type: (T) -> T
    """Decorator to make from_type lazy to support recursive definitions."""

    @proxies(func)
    def inner(*args, **kwargs):
        return deferred(lambda: func(*args, **kwargs))

    return inner


@cacheable
@_defer_from_type
def from_type(thing):
    # type: (Type[Ex]) -> SearchStrategy[Ex]
    """Looks up the appropriate search strategy for the given type.

    ``from_type`` is used internally to fill in missing arguments to
    :func:`~hypothesis.strategies.builds` and can be used interactively
    to explore what strategies are available or to debug type resolution.

    You can use :func:`~hypothesis.strategies.register_type_strategy` to
    handle your custom types, or to globally redefine certain strategies -
    for example excluding NaN from floats, or use timezone-aware instead of
    naive time and datetime strategies.

    The resolution logic may be changed in a future version, but currently
    tries these five options:

    1. If ``thing`` is in the default lookup mapping or user-registered lookup,
       return the corresponding strategy.  The default lookup covers all types
       with Hypothesis strategies, including extras where possible.
    2. If ``thing`` is from the :mod:`python:typing` module, return the
       corresponding strategy (special logic).
    3. If ``thing`` has one or more subtypes in the merged lookup, return
       the union of the strategies for those types that are not subtypes of
       other elements in the lookup.
    4. Finally, if ``thing`` has type annotations for all required arguments,
       and is not an abstract class, it is resolved via
       :func:`~hypothesis.strategies.builds`.
    5. Because :mod:`abstract types <python:abc>` cannot be instantiated,
       we treat abstract types as the union of their concrete subclasses.
       Note that this lookup works via inheritance but not via
       :obj:`~python:abc.ABCMeta.register`, so you may still need to use
       :func:`~hypothesis.strategies.register_type_strategy`.

    There is a valuable recipe for leveraging ``from_type()`` to generate
    "everything except" values from a specified type. I.e.

    .. code-block:: python

        def everything_except(excluded_types):
            return (
                from_type(type).flatmap(from_type)
                .filter(lambda x: not isinstance(x, excluded_types))
            )

    For example, ``everything_except(int)`` returns a strategy that can
    generate anything that ``from_type()`` can ever generate, except for
    instances of :class:`python:int`, and excluding instances of types
    added via :func:`~hypothesis.strategies.register_type_strategy`.

    This is useful when writing tests which check that invalid input is
    rejected in a certain way.
    """
    if (
        hasattr(typing, "_TypedDictMeta")
        and type(thing) is typing._TypedDictMeta  # type: ignore
    ):  # pragma: no cover
        # The __optional_keys__ attribute may or may not be present, but if there's no
        # way to tell and we just have to assume that everything is required.
        # See https://github.com/python/cpython/pull/17214 for details.
        optional = getattr(thing, "__optional_keys__", ())
        anns = {k: from_type(v) for k, v in thing.__annotations__.items()}
        return fixed_dictionaries(  # type: ignore
            mapping={k: v for k, v in anns.items() if k not in optional},
            optional={k: v for k, v in anns.items() if k in optional},
        )

    # TODO: We would like to move this to the top level, but pending some major
    # refactoring it's hard to do without creating circular imports.
    from hypothesis.strategies._internal import types

    def as_strategy(strat_or_callable, thing, final=True):
        # User-provided strategies need some validation, and callables even more
        # of it.  We do this in three places, hence the helper function
        if not isinstance(strat_or_callable, SearchStrategy):
            assert callable(strat_or_callable)  # Validated in register_type_strategy
            try:
                # On Python 3.6, typing.Hashable is just an alias for abc.Hashable,
                # and the resolver function for Type throws an AttributeError because
                # Hashable has no __args__.  We discard such errors when attempting
                # to resolve subclasses, because the function was passed a weird arg.
                strategy = strat_or_callable(thing)
            except Exception:  # pragma: no cover
                if not final:
                    return NOTHING
                raise
        else:
            strategy = strat_or_callable
        if not isinstance(strategy, SearchStrategy):
            raise ResolutionFailed(
                "Error: %s was registered for %r, but returned non-strategy %r"
                % (thing, nicerepr(strat_or_callable), strategy)
            )
        if strategy.is_empty:
            raise ResolutionFailed("Error: %r resolved to an empty strategy" % (thing,))
        return strategy

    if typing is not None:  # pragma: no branch
        if not isinstance(thing, type):
            if types.is_a_new_type(thing):
                # Check if we have an explicitly registered strategy for this thing,
                # resolve it so, and otherwise resolve as for the base type.
                if thing in types._global_type_lookup:
                    return as_strategy(types._global_type_lookup[thing], thing)
                return from_type(thing.__supertype__)
            # Under Python 3.6, Unions are not instances of `type` - but we
            # still want to resolve them!
            if getattr(thing, "__origin__", None) is typing.Union:
                args = sorted(thing.__args__, key=types.type_sorting_key)
                return one_of([from_type(t) for t in args])
        # We can't resolve forward references, and under Python 3.5 (only)
        # a forward reference is an instance of type.  Hence, explicit check:
        elif type(thing) == getattr(typing, "_ForwardRef", None):  # pragma: no cover
            raise ResolutionFailed(
                "thing=%s cannot be resolved.  Upgrading to python>=3.6 may "
                "fix this problem via improvements to the typing module." % (thing,)
            )
    if not types.is_a_type(thing):
        raise InvalidArgument("thing=%s must be a type" % (thing,))
    # Now that we know `thing` is a type, the first step is to check for an
    # explicitly registered strategy.  This is the best (and hopefully most
    # common) way to resolve a type to a strategy.  Note that the value in the
    # lookup may be a strategy or a function from type -> strategy; and we
    # convert empty results into an explicit error.
    if thing in types._global_type_lookup:
        return as_strategy(types._global_type_lookup[thing], thing)
    # If there's no explicitly registered strategy, maybe a subtype of thing
    # is registered - if so, we can resolve it to the subclass strategy.
    # We'll start by checking if thing is from from the typing module,
    # because there are several special cases that don't play well with
    # subclass and instance checks.
    if typing is not None:  # pragma: no branch
        if isinstance(thing, typing_root_type):
            return types.from_typing_type(thing)
    # If it's not from the typing module, we get all registered types that are
    # a subclass of `thing` and are not themselves a subtype of any other such
    # type.  For example, `Number -> integers() | floats()`, but bools() is
    # not included because bool is a subclass of int as well as Number.
    strategies = [
        as_strategy(v, thing, final=False)
        for k, v in sorted(types._global_type_lookup.items(), key=repr)
        if isinstance(k, type)
        and issubclass(k, thing)
        and sum(types.try_issubclass(k, typ) for typ in types._global_type_lookup) == 1
    ]
    if any(not s.is_empty for s in strategies):
        return one_of(strategies)
    # If we don't have a strategy registered for this type or any subtype, we
    # may be able to fall back on type annotations.
    if issubclass(thing, enum.Enum):
        return sampled_from(thing)
    # If we know that builds(thing) will fail, give a better error message
    required = required_args(thing)
    if required and not any(
        [
            required.issubset(get_type_hints(thing.__init__)),
            attr.has(thing),
            # NamedTuples are weird enough that we need a specific check for them.
            is_typed_named_tuple(thing),
        ]
    ):
        raise ResolutionFailed(
            "Could not resolve %r to a strategy; consider "
            "using register_type_strategy" % (thing,)
        )
    # Finally, try to build an instance by calling the type object
    if not isabstract(thing):
        return builds(thing)
    subclasses = thing.__subclasses__()
    if not subclasses:
        raise ResolutionFailed(
            "Could not resolve %r to a strategy, because it is an abstract type "
            "without any subclasses. Consider using register_type_strategy" % (thing,)
        )
    return sampled_from(subclasses).flatmap(from_type)


@cacheable
@defines_strategy_with_reusable_values
def fractions(
    min_value=None,  # type: Union[Real, AnyStr]
    max_value=None,  # type: Union[Real, AnyStr]
    max_denominator=None,  # type: int
):
    # type: (...) -> SearchStrategy[Fraction]
    """Returns a strategy which generates Fractions.

    If ``min_value`` is not None then all generated values are no less than
    ``min_value``.  If ``max_value`` is not None then all generated values are no
    greater than ``max_value``.  ``min_value`` and ``max_value`` may be anything accepted
    by the :class:`~fractions.Fraction` constructor.

    If ``max_denominator`` is not None then the denominator of any generated
    values is no greater than ``max_denominator``. Note that ``max_denominator`` must
    be None or a positive integer.

    Examples from this strategy shrink towards smaller denominators, then
    closer to zero.
    """
    min_value = try_convert(Fraction, min_value, "min_value")
    max_value = try_convert(Fraction, max_value, "max_value")
    # These assertions tell Mypy what happened in try_convert
    assert min_value is None or isinstance(min_value, Fraction)
    assert max_value is None or isinstance(max_value, Fraction)

    check_valid_interval(min_value, max_value, "min_value", "max_value")
    check_valid_integer(max_denominator)

    if max_denominator is not None:
        if max_denominator < 1:
            raise InvalidArgument("max_denominator=%r must be >= 1" % max_denominator)

        def fraction_bounds(value):
            # type: (Fraction) -> Tuple[Fraction, Fraction]
            """Find the best lower and upper approximation for value."""
            # Adapted from CPython's Fraction.limit_denominator here:
            # https://github.com/python/cpython/blob/3.6/Lib/fractions.py#L219
            assert max_denominator is not None
            if value is None or value.denominator <= max_denominator:
                return value, value

            p0, q0, p1, q1 = 0, 1, 1, 0
            n, d = value.numerator, value.denominator
            while True:
                a = n // d
                q2 = q0 + a * q1
                if q2 > max_denominator:
                    break
                p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
                n, d = d, n - a * d
            k = (max_denominator - q0) // q1
            low, high = Fraction(p1, q1), Fraction(p0 + k * p1, q0 + k * q1)
            assert low < value < high
            return low, high

        # Take the high approximation for min_value and low for max_value
        bounds = (max_denominator, min_value, max_value)
        if min_value is not None:
            if min_value.denominator > max_denominator:
                note_deprecation(
                    "The min_value=%r has a denominator greater than the "
                    "max_denominator=%r, which will be an error in a future "
                    "version." % (min_value, max_denominator),
                    since="2018-10-12",
                )
            _, min_value = fraction_bounds(min_value)
        if max_value is not None:
            if max_value.denominator > max_denominator:
                note_deprecation(
                    "The max_value=%r has a denominator greater than the "
                    "max_denominator=%r, which will be an error in a future "
                    "version." % (max_value, max_denominator),
                    since="2018-10-12",
                )
            max_value, _ = fraction_bounds(max_value)

        if min_value is not None and max_value is not None and min_value > max_value:
            raise InvalidArgument(
                "There are no fractions with a denominator <= %r between "
                "min_value=%r and max_value=%r" % bounds
            )

    if min_value is not None and min_value == max_value:
        return just(min_value)

    def dm_func(denom):
        """Take denom, construct numerator strategy, and build fraction."""
        # Four cases of algebra to get integer bounds and scale factor.
        min_num, max_num = None, None
        if max_value is None and min_value is None:
            pass
        elif min_value is None:
            max_num = denom * max_value.numerator
            denom *= max_value.denominator
        elif max_value is None:
            min_num = denom * min_value.numerator
            denom *= min_value.denominator
        else:
            low = min_value.numerator * max_value.denominator
            high = max_value.numerator * min_value.denominator
            scale = min_value.denominator * max_value.denominator
            # After calculating our integer bounds and scale factor, we remove
            # the gcd to avoid drawing more bytes for the example than needed.
            # Note that `div` can be at most equal to `scale`.
            div = gcd(scale, gcd(low, high))
            min_num = denom * low // div
            max_num = denom * high // div
            denom *= scale // div

        return builds(
            Fraction, integers(min_value=min_num, max_value=max_num), just(denom)
        )

    if max_denominator is None:
        return integers(min_value=1).flatmap(dm_func)

    return (
        integers(1, max_denominator)
        .flatmap(dm_func)
        .map(lambda f: f.limit_denominator(max_denominator))
    )


def _as_finite_decimal(
    value,  # type: Union[Real, AnyStr, None]
    name,  # type: str
    allow_infinity,  # type: Optional[bool]
):
    # type: (...) -> Optional[Decimal]
    """Convert decimal bounds to decimals, carefully."""
    assert name in ("min_value", "max_value")
    if value is None:
        return None
    if not isinstance(value, Decimal):
        with localcontext(Context()):  # ensure that default traps are enabled
            value = try_convert(Decimal, value, name)
    assert isinstance(value, Decimal)
    if value.is_finite():
        return value
    if value.is_infinite() and (value < 0 if "min" in name else value > 0):
        if allow_infinity or allow_infinity is None:
            return None
        raise InvalidArgument(
            "allow_infinity=%r, but %s=%r" % (allow_infinity, name, value)
        )
    # This could be infinity, quiet NaN, or signalling NaN
    raise InvalidArgument(u"Invalid %s=%r" % (name, value))


@cacheable
@defines_strategy_with_reusable_values
def decimals(
    min_value=None,  # type: Union[Real, AnyStr]
    max_value=None,  # type: Union[Real, AnyStr]
    allow_nan=None,  # type: bool
    allow_infinity=None,  # type: bool
    places=None,  # type: int
):
    # type: (...) -> SearchStrategy[Decimal]
    """Generates instances of :class:`python:decimal.Decimal`, which may be:

    - A finite rational number, between ``min_value`` and ``max_value``.
    - Not a Number, if ``allow_nan`` is True.  None means "allow NaN, unless
      ``min_value`` and ``max_value`` are not None".
    - Positive or negative infinity, if ``max_value`` and ``min_value``
      respectively are None, and ``allow_infinity`` is not False.  None means
      "allow infinity, unless excluded by the min and max values".

    Note that where floats have one ``NaN`` value, Decimals have four: signed,
    and either *quiet* or *signalling*.  See `the decimal module docs
    <https://docs.python.org/3/library/decimal.html#special-values>`_ for
    more information on special values.

    If ``places`` is not None, all finite values drawn from the strategy will
    have that number of digits after the decimal place.

    Examples from this strategy do not have a well defined shrink order but
    try to maximize human readability when shrinking.
    """
    # Convert min_value and max_value to Decimal values, and validate args
    check_valid_integer(places)
    if places is not None and places < 0:
        raise InvalidArgument("places=%r may not be negative" % places)
    min_value = _as_finite_decimal(min_value, "min_value", allow_infinity)
    max_value = _as_finite_decimal(max_value, "max_value", allow_infinity)
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if allow_infinity and (None not in (min_value, max_value)):
        raise InvalidArgument("Cannot allow infinity between finite bounds")
    # Set up a strategy for finite decimals.  Note that both floating and
    # fixed-point decimals require careful handling to remain isolated from
    # any external precision context - in short, we always work out the
    # required precision for lossless operation and use context methods.
    if places is not None:
        # Fixed-point decimals are basically integers with a scale factor
        def ctx(val):
            """Return a context in which this value is lossless."""
            precision = ceil(math.log10(abs(val) or 1)) + places + 1
            return Context(prec=max([precision, 1]))

        def int_to_decimal(val):
            context = ctx(val)
            return context.quantize(context.multiply(val, factor), factor)

        factor = Decimal(10) ** -places
        min_num, max_num = None, None
        if min_value is not None:
            min_num = ceil(ctx(min_value).divide(min_value, factor))
        if max_value is not None:
            max_num = floor(ctx(max_value).divide(max_value, factor))
        if min_num is not None and max_num is not None and min_num > max_num:
            raise InvalidArgument(
                "There are no decimals with %d places between min_value=%r "
                "and max_value=%r " % (places, min_value, max_value)
            )
        strat = integers(min_num, max_num).map(int_to_decimal)
    else:
        # Otherwise, they're like fractions featuring a power of ten
        def fraction_to_decimal(val):
            precision = (
                ceil(math.log10(abs(val.numerator) or 1) + math.log10(val.denominator))
                + 1
            )
            return Context(prec=precision or 1).divide(
                Decimal(val.numerator), val.denominator
            )

        strat = fractions(min_value, max_value).map(fraction_to_decimal)
    # Compose with sampled_from for infinities and NaNs as appropriate
    special = []  # type: List[Decimal]
    if allow_nan or (allow_nan is None and (None in (min_value, max_value))):
        special.extend(map(Decimal, ("NaN", "-NaN", "sNaN", "-sNaN")))
    if allow_infinity or (allow_infinity is max_value is None):
        special.append(Decimal("Infinity"))
    if allow_infinity or (allow_infinity is min_value is None):
        special.append(Decimal("-Infinity"))
    return strat | (sampled_from(special) if special else nothing())


def recursive(
    base,  # type: SearchStrategy[Ex]
    extend,  # type: Callable[[SearchStrategy[Any]], SearchStrategy[T]]
    max_leaves=100,  # type: int
):
    # type: (...) -> SearchStrategy[Union[T, Ex]]
    """base: A strategy to start from.

    extend: A function which takes a strategy and returns a new strategy.

    max_leaves: The maximum number of elements to be drawn from base on a given
    run.

    This returns a strategy ``S`` such that ``S = extend(base | S)``. That is,
    values may be drawn from base, or from any strategy reachable by mixing
    applications of | and extend.

    An example may clarify: ``recursive(booleans(), lists)`` would return a
    strategy that may return arbitrarily nested and mixed lists of booleans.
    So e.g. ``False``, ``[True]``, ``[False, []]``, and ``[[[[True]]]]`` are
    all valid values to be drawn from that strategy.

    Examples from this strategy shrink by trying to reduce the amount of
    recursion and by shrinking according to the shrinking behaviour of base
    and the result of extend.

    """

    return RecursiveStrategy(base, extend, max_leaves)


class PermutationStrategy(SearchStrategy):
    def __init__(self, values):
        self.values = values

    def do_draw(self, data):
        # Reversed Fisher-Yates shuffle: swap each element with itself or with
        # a later element.  This shrinks i==j for each element, i.e. to no
        # change.  We don't consider the last element as it's always a no-op.
        result = list(self.values)
        for i in hrange(len(result) - 1):
            j = integer_range(data, i, len(result) - 1)
            result[i], result[j] = result[j], result[i]
        return result


@defines_strategy
def permutations(values):
    # type: (Sequence[T]) -> SearchStrategy[List[T]]
    """Return a strategy which returns permutations of the ordered collection
    ``values``.

    Examples from this strategy shrink by trying to become closer to the
    original order of values.
    """
    values = check_sample(values, "permutations")
    if not values:
        return builds(list)

    return PermutationStrategy(values)


@defines_strategy_with_reusable_values
def datetimes(
    min_value=dt.datetime.min,  # type: dt.datetime
    max_value=dt.datetime.max,  # type: dt.datetime
    timezones=none(),  # type: SearchStrategy[Optional[dt.tzinfo]]
):
    # type: (...) -> SearchStrategy[dt.datetime]
    """A strategy for generating datetimes, which may be timezone-aware.

    This strategy works by drawing a naive datetime between ``min_value``
    and ``max_value``, which must both be naive (have no timezone).

    ``timezones`` must be a strategy that generates
    :class:`~python:datetime.tzinfo` objects (or None,
    which is valid for naive datetimes).  A value drawn from this strategy
    will be added to a naive datetime, and the resulting tz-aware datetime
    returned.

    .. note::
        tz-aware datetimes from this strategy may be ambiguous or non-existent
        due to daylight savings, leap seconds, timezone and calendar
        adjustments, etc.  This is intentional, as malformed timestamps are a
        common source of bugs.

    :py:func:`hypothesis.extra.pytz.timezones` requires the :pypi:`pytz`
    package, but provides all timezones in the Olsen database.  If you want to
    allow naive datetimes, combine strategies like ``none() | timezones()``.

    :py:func:`hypothesis.extra.dateutil.timezones` requires the
    :pypi:`python-dateutil` package, and similarly provides all timezones
    there.

    Alternatively, you can create a list of the timezones you wish to allow
    (e.g. from the standard library, :pypi:`dateutil`, or :pypi:`pytz`) and use
    :py:func:`sampled_from`.  Ensure that simple values such as None or UTC
    are at the beginning of the list for proper minimisation.

    Examples from this strategy shrink towards midnight on January 1st 2000.
    """
    # Why must bounds be naive?  In principle, we could also write a strategy
    # that took aware bounds, but the API and validation is much harder.
    # If you want to generate datetimes between two particular moments in
    # time I suggest (a) just filtering out-of-bounds values; (b) if bounds
    # are very close, draw a value and subtract its UTC offset, handling
    # overflows and nonexistent times; or (c) do something customised to
    # handle datetimes in e.g. a four-microsecond span which is not
    # representable in UTC.  Handling (d), all of the above, leads to a much
    # more complex API for all users and a useful feature for very few.
    check_type(dt.datetime, min_value, "min_value")
    check_type(dt.datetime, max_value, "max_value")
    if min_value.tzinfo is not None:
        raise InvalidArgument("min_value=%r must not have tzinfo" % (min_value,))
    if max_value.tzinfo is not None:
        raise InvalidArgument("max_value=%r must not have tzinfo" % (max_value,))
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if not isinstance(timezones, SearchStrategy):
        raise InvalidArgument(
            "timezones=%r must be a SearchStrategy that can provide tzinfo "
            "for datetimes (either None or dt.tzinfo objects)" % (timezones,)
        )
    return DatetimeStrategy(min_value, max_value, timezones)


@defines_strategy_with_reusable_values
def dates(min_value=dt.date.min, max_value=dt.date.max):
    # type: (dt.date, dt.date) -> SearchStrategy[dt.date]
    """A strategy for dates between ``min_value`` and ``max_value``.

    Examples from this strategy shrink towards January 1st 2000.
    """
    check_type(dt.date, min_value, "min_value")
    check_type(dt.date, max_value, "max_value")
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if min_value == max_value:
        return just(min_value)
    return DateStrategy(min_value, max_value)


@defines_strategy_with_reusable_values
def times(min_value=dt.time.min, max_value=dt.time.max, timezones=none()):
    # type: (dt.time, dt.time, SearchStrategy) -> SearchStrategy[dt.time]
    """A strategy for times between ``min_value`` and ``max_value``.

    The ``timezones`` argument is handled as for :py:func:`datetimes`.

    Examples from this strategy shrink towards midnight, with the timezone
    component shrinking as for the strategy that provided it.
    """
    check_type(dt.time, min_value, "min_value")
    check_type(dt.time, max_value, "max_value")
    if min_value.tzinfo is not None:
        raise InvalidArgument("min_value=%r must not have tzinfo" % min_value)
    if max_value.tzinfo is not None:
        raise InvalidArgument("max_value=%r must not have tzinfo" % max_value)
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    day = dt.date(2000, 1, 1)
    return datetimes(
        min_value=dt.datetime.combine(day, min_value),
        max_value=dt.datetime.combine(day, max_value),
        timezones=timezones,
    ).map(lambda t: t.timetz())


@defines_strategy_with_reusable_values
def timedeltas(min_value=dt.timedelta.min, max_value=dt.timedelta.max):
    # type: (dt.timedelta, dt.timedelta) -> SearchStrategy[dt.timedelta]
    """A strategy for timedeltas between ``min_value`` and ``max_value``.

    Examples from this strategy shrink towards zero.
    """
    check_type(dt.timedelta, min_value, "min_value")
    check_type(dt.timedelta, max_value, "max_value")
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if min_value == max_value:
        return just(min_value)
    return TimedeltaStrategy(min_value=min_value, max_value=max_value)


class CompositeStrategy(SearchStrategy):
    def __init__(self, definition, label, args, kwargs):
        self.definition = definition
        self.__label = label
        self.args = args
        self.kwargs = kwargs

    def do_draw(self, data):
        return self.definition(data.draw, *self.args, **self.kwargs)

    def calc_label(self):
        return self.__label


@cacheable
def composite(f):
    # type: (Callable[..., Ex]) -> Callable[..., SearchStrategy[Ex]]
    """Defines a strategy that is built out of potentially arbitrarily many
    other strategies.

    This is intended to be used as a decorator. See
    :ref:`the full documentation for more details <composite-strategies>`
    about how to use this function.

    Examples from this strategy shrink by shrinking the output of each draw
    call.
    """
    argspec = getfullargspec(f)

    if argspec.defaults is not None and len(argspec.defaults) == len(argspec.args):
        raise InvalidArgument("A default value for initial argument will never be used")
    if len(argspec.args) == 0 and not argspec.varargs:
        raise InvalidArgument(
            "Functions wrapped with composite must take at least one "
            "positional argument."
        )

    annots = {
        k: v
        for k, v in argspec.annotations.items()
        if k in (argspec.args + argspec.kwonlyargs + ["return"])
    }
    new_argspec = argspec._replace(args=argspec.args[1:], annotations=annots)

    label = calc_label_from_cls(f)

    @defines_strategy
    @define_function_signature(f.__name__, f.__doc__, new_argspec)
    def accept(*args, **kwargs):
        return CompositeStrategy(f, label, args, kwargs)

    accept.__module__ = f.__module__
    return accept


@defines_strategy_with_reusable_values
@cacheable
def complex_numbers(
    min_magnitude=0, max_magnitude=None, allow_infinity=None, allow_nan=None
):
    # type: (Optional[Real], Real, bool, bool) -> SearchStrategy[complex]
    """Returns a strategy that generates complex numbers.

    This strategy draws complex numbers with constrained magnitudes.
    The ``min_magnitude`` and ``max_magnitude`` parameters should be
    non-negative :class:`~python:numbers.Real` numbers; values
    of ``None`` correspond to zero and infinite values respectively.

    If ``min_magnitude`` is positive or ``max_magnitude`` is finite, it
    is an error to enable ``allow_nan``.  If ``max_magnitude`` is finite,
    it is an error to enable ``allow_infinity``.

    The magnitude contraints are respected up to a relative error
    of (around) floating-point epsilon, due to implementation via
    the system ``sqrt`` function.

    Examples from this strategy shrink by shrinking their real and
    imaginary parts, as :func:`~hypothesis.strategies.floats`.

    If you need to generate complex numbers with particular real and
    imaginary parts or relationships between parts, consider using
    :func:`builds(complex, ...) <hypothesis.strategies.builds>` or
    :func:`@composite <hypothesis.strategies.composite>` respectively.
    """
    check_valid_magnitude(min_magnitude, "min_magnitude")
    check_valid_magnitude(max_magnitude, "max_magnitude")
    check_valid_interval(min_magnitude, max_magnitude, "min_magnitude", "max_magnitude")
    if max_magnitude == float("inf"):
        max_magnitude = None
    if min_magnitude == 0:
        min_magnitude = None

    if allow_infinity is None:
        allow_infinity = bool(max_magnitude is None)
    elif allow_infinity and max_magnitude is not None:
        raise InvalidArgument(
            "Cannot have allow_infinity=%r with max_magnitude=%r"
            % (allow_infinity, max_magnitude)
        )
    if allow_nan is None:
        allow_nan = bool(min_magnitude is None and max_magnitude is None)
    elif allow_nan and not (min_magnitude is None and max_magnitude is None):
        raise InvalidArgument(
            "Cannot have allow_nan=%r, min_magnitude=%r max_magnitude=%r"
            % (allow_nan, min_magnitude, max_magnitude)
        )
    allow_kw = {"allow_nan": allow_nan, "allow_infinity": allow_infinity}

    if min_magnitude is None and max_magnitude is None:
        # In this simple but common case, there are no constraints on the
        # magnitude and therefore no relationship between the real and
        # imaginary parts.
        return builds(complex, floats(**allow_kw), floats(**allow_kw))

    @composite
    def constrained_complex(draw):
        # Draw the imaginary part, and determine the maximum real part given
        # this and the max_magnitude
        if max_magnitude is None:
            zi = draw(floats(**allow_kw))
            rmax = None
        else:
            zi = draw(floats(-max_magnitude, max_magnitude, **allow_kw))
            rmax = cathetus(max_magnitude, zi)
        # Draw the real part from the allowed range given the imaginary part
        if min_magnitude is None or math.fabs(zi) >= min_magnitude:
            zr = draw(floats(None if rmax is None else -rmax, rmax, **allow_kw))
        else:
            zr = draw(floats(cathetus(min_magnitude, zi), rmax, **allow_kw))
        # Order of conditions carefully tuned so that for a given pair of
        # magnitude arguments, we always either draw or do not draw the bool
        # (crucial for good shrinking behaviour) but only invert when needed.
        if (
            min_magnitude is not None
            and draw(booleans())
            and math.fabs(zi) <= min_magnitude
        ):
            zr = -zr
        return complex(zr, zi)

    return constrained_complex()


def shared(base, key=None):
    # type: (SearchStrategy[Ex], Any) -> SearchStrategy[Ex]
    """Returns a strategy that draws a single shared value per run, drawn from
    base. Any two shared instances with the same key will share the same value,
    otherwise the identity of this strategy will be used. That is:

    >>> s = integers()  # or any other strategy
    >>> x = shared(s)
    >>> y = shared(s)

    In the above x and y may draw different (or potentially the same) values.
    In the following they will always draw the same:

    >>> x = shared(s, key="hi")
    >>> y = shared(s, key="hi")

    Examples from this strategy shrink as per their base strategy.
    """
    return SharedStrategy(base, key)


@cacheable
@defines_strategy_with_reusable_values
def uuids(version=None):
    # type: (int) -> SearchStrategy[UUID]
    """Returns a strategy that generates :class:`UUIDs <uuid.UUID>`.

    If the optional version argument is given, value is passed through
    to :class:`~python:uuid.UUID` and only UUIDs of that version will
    be generated.

    All returned values from this will be unique, so e.g. if you do
    ``lists(uuids())`` the resulting list will never contain duplicates.

    Examples from this strategy don't have any meaningful shrink order.
    """
    if version not in (None, 1, 2, 3, 4, 5):
        raise InvalidArgument(
            (
                "version=%r, but version must be in (None, 1, 2, 3, 4, 5) "
                "to pass to the uuid.UUID constructor."
            )
            % (version,)
        )
    return shared(randoms(), key="hypothesis.strategies.uuids.generator").map(
        lambda r: UUID(version=version, int=r.getrandbits(128))
    )


class RunnerStrategy(SearchStrategy):
    def __init__(self, default):
        self.default = default

    def do_draw(self, data):
        runner = getattr(data, "hypothesis_runner", not_set)
        if runner is not_set:
            if self.default is not_set:
                raise InvalidArgument(
                    "Cannot use runner() strategy with no "
                    "associated runner or explicit default."
                )
            else:
                return self.default
        else:
            return runner


@defines_strategy_with_reusable_values
def runner(default=not_set):
    # type: (Any) -> SearchStrategy[Any]
    """A strategy for getting "the current test runner", whatever that may be.
    The exact meaning depends on the entry point, but it will usually be the
    associated 'self' value for it.

    If there is no current test runner and a default is provided, return
    that default. If no default is provided, raises InvalidArgument.

    Examples from this strategy do not shrink (because there is only one).
    """
    return RunnerStrategy(default)


class DataObject(object):
    """This type only exists so that you can write type hints for tests using
    the :func:`~hypothesis.strategies.data` strategy.  Do not use it directly!
    """

    # Note that "only exists" here really means "is only exported to users",
    # but we want to treat it as "semi-stable", not document it as "public API".

    def __init__(self, data):
        self.count = 0
        self.conjecture_data = data

    def __repr__(self):
        return "data(...)"

    def draw(self, strategy, label=None):
        # type: (SearchStrategy[Ex], Any) -> Ex
        check_type(SearchStrategy, strategy, "strategy")
        result = self.conjecture_data.draw(strategy)
        self.count += 1
        if label is not None:
            note("Draw %d (%s): %r" % (self.count, label, result))
        else:
            note("Draw %d: %r" % (self.count, result))
        return result


class DataStrategy(SearchStrategy):
    supports_find = False

    def do_draw(self, data):
        if not hasattr(data, "hypothesis_shared_data_strategy"):
            data.hypothesis_shared_data_strategy = DataObject(data)
        return data.hypothesis_shared_data_strategy

    def __repr__(self):
        return "data()"

    def map(self, f):
        self.__not_a_first_class_strategy("map")

    def filter(self, f):
        self.__not_a_first_class_strategy("filter")

    def flatmap(self, f):
        self.__not_a_first_class_strategy("flatmap")

    def example(self):
        self.__not_a_first_class_strategy("example")

    def __not_a_first_class_strategy(self, name):
        raise InvalidArgument(
            "Cannot call %s on a DataStrategy. You should probably be using "
            "@composite for whatever it is you're trying to do." % (name,)
        )


@cacheable
def data():
    # type: () -> SearchStrategy[DataObject]
    """This isn't really a normal strategy, but instead gives you an object
    which can be used to draw data interactively from other strategies.

    See :ref:`the rest of the documentation <interactive-draw>` for more
    complete information.

    Examples from this strategy do not shrink (because there is only one),
    but the result of calls to each draw() call shrink as they normally would.
    """
    return DataStrategy()


def register_type_strategy(
    custom_type,  # type: type
    strategy,  # type: Union[SearchStrategy, Callable[[type], SearchStrategy]]
):
    # type: (...) -> None
    """Add an entry to the global type-to-strategy lookup.

    This lookup is used in :func:`~hypothesis.strategies.builds` and
    :func:`@given <hypothesis.given>`.

    :func:`~hypothesis.strategies.builds` will be used automatically for
    classes with type annotations on ``__init__`` , so you only need to
    register a strategy if one or more arguments need to be more tightly
    defined than their type-based default, or if you want to supply a strategy
    for an argument with a default value.

    ``strategy`` may be a search strategy, or a function that takes a type and
    returns a strategy (useful for generic types).
    """
    # TODO: We would like to move this to the top level, but pending some major
    # refactoring it's hard to do without creating circular imports.
    from hypothesis.strategies._internal import types

    if not types.is_a_type(custom_type):
        raise InvalidArgument("custom_type=%r must be a type")
    elif not (isinstance(strategy, SearchStrategy) or callable(strategy)):
        raise InvalidArgument(
            "strategy=%r must be a SearchStrategy, or a function that takes "
            "a generic type and returns a specific SearchStrategy"
        )
    elif isinstance(strategy, SearchStrategy) and strategy.is_empty:
        raise InvalidArgument("strategy=%r must not be empty")
    types._global_type_lookup[custom_type] = strategy
    from_type.__clear_cache()  # type: ignore


@cacheable
def deferred(definition):
    # type: (Callable[[], SearchStrategy[Ex]]) -> SearchStrategy[Ex]
    """A deferred strategy allows you to write a strategy that references other
    strategies that have not yet been defined. This allows for the easy
    definition of recursive and mutually recursive strategies.

    The definition argument should be a zero-argument function that returns a
    strategy. It will be evaluated the first time the strategy is used to
    produce an example.

    Example usage:

    >>> import hypothesis.strategies as st
    >>> x = st.deferred(lambda: st.booleans() | st.tuples(x, x))
    >>> x.example()
    (((False, (True, True)), (False, True)), (True, True))
    >>> x.example()
    True

    Mutual recursion also works fine:

    >>> a = st.deferred(lambda: st.booleans() | b)
    >>> b = st.deferred(lambda: st.tuples(a, a))
    >>> a.example()
    True
    >>> b.example()
    (False, (False, ((False, True), False)))

    Examples from this strategy shrink as they normally would from the strategy
    returned by the definition.
    """
    return DeferredStrategy(definition)


@defines_strategy_with_reusable_values
def emails():
    # type: () -> SearchStrategy[Text]
    """A strategy for generating email addresses as unicode strings. The
    address format is specified in :rfc:`5322#section-3.4.1`. Values shrink
    towards shorter local-parts and host domains.

    This strategy is useful for generating "user data" for tests, as
    mishandling of email addresses is a common source of bugs.
    """
    from hypothesis.provisional import domains

    local_chars = string.ascii_letters + string.digits + "!#$%&'*+-/=^_`{|}~"
    local_part = text(local_chars, min_size=1, max_size=64)
    # TODO: include dot-atoms, quoted strings, escaped chars, etc in local part
    return builds(u"{}@{}".format, local_part, domains()).filter(
        lambda addr: len(addr) <= 254
    )


@defines_strategy
def functions(
    like=lambda: None,  # type: Callable[..., Any]
    returns=none(),  # type: SearchStrategy[Any]
):
    # type: (...) -> SearchStrategy[Callable[..., Any]]
    # The proper type signature of `functions()` would have T instead of Any, but mypy
    # disallows default args for generics: https://github.com/python/mypy/issues/3737
    """A strategy for functions, which can be used in callbacks.

    The generated functions will mimic the interface of ``like``, which must
    be a callable (including a class, method, or function).  The return value
    for the function is drawn from the ``returns`` argument, which must be a
    strategy.

    Note that the generated functions do not validate their arguments, and
    may return a different value if called again with the same arguments.

    Generated functions can only be called within the scope of the ``@given``
    which created them.  This strategy does not support ``.example()``.
    """
    check_type(SearchStrategy, returns)
    if not callable(like):
        raise InvalidArgument(
            "The first argument to functions() must be a callable to imitate, "
            "but got non-callable like=%r" % (nicerepr(like),)
        )
    return FunctionStrategy(like, returns)


@composite
def slices(draw, size):
    # type: (Any, int) -> slice
    """Generates slices that will select indices up to the supplied size

    Generated slices will have start and stop indices that range from 0 to size - 1
    and will step in the appropriate direction. Slices should only produce an empty selection
    if the start and end are the same.

    Examples from this strategy shrink toward 0 and smaller values
    """
    check_valid_integer(size)
    if size is None or size < 1:
        raise InvalidArgument("size=%r must be at least one" % size)

    min_start = min_stop = 0
    max_start = max_stop = size
    min_step = 1
    # For slices start is inclusive and stop is exclusive
    start = draw(integers(min_start, max_start) | none())
    stop = draw(integers(min_stop, max_stop) | none())

    # Limit step size to be reasonable
    if start is None and stop is None:
        max_step = size
    elif start is None:
        max_step = stop
    elif stop is None:
        max_step = start
    else:
        max_step = abs(start - stop)

    step = draw(integers(min_step, max_step or 1))

    if (stop or 0) < (start or 0):
        step *= -1

    return slice(start, stop, step)
