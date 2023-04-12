# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import threading
from inspect import signature
from typing import TYPE_CHECKING, Callable, Dict

from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.floats import float_to_int
from hypothesis.internal.reflection import proxies

if TYPE_CHECKING:
    from hypothesis.strategies._internal.strategies import SearchStrategy, T

_strategies: Dict[str, Callable[..., "SearchStrategy"]] = {}


class FloatKey:
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


_CACHE = threading.local()


def get_cache() -> LRUReusedCache:
    try:
        return _CACHE.STRATEGY_CACHE
    except AttributeError:
        _CACHE.STRATEGY_CACHE = LRUReusedCache(1024)
        return _CACHE.STRATEGY_CACHE


def clear_cache() -> None:
    cache = get_cache()
    cache.clear()


def cacheable(fn: "T") -> "T":
    from hypothesis.strategies._internal.strategies import SearchStrategy

    @proxies(fn)
    def cached_strategy(*args, **kwargs):
        try:
            kwargs_cache_key = {(k, convert_value(v)) for k, v in kwargs.items()}
        except TypeError:
            return fn(*args, **kwargs)
        cache_key = (fn, tuple(map(convert_value, args)), frozenset(kwargs_cache_key))
        cache = get_cache()
        try:
            if cache_key in cache:
                return cache[cache_key]
        except TypeError:
            return fn(*args, **kwargs)
        else:
            result = fn(*args, **kwargs)
            if not isinstance(result, SearchStrategy) or result.is_cacheable:
                cache[cache_key] = result
            return result

    cached_strategy.__clear_cache = clear_cache  # type: ignore
    return cached_strategy


def defines_strategy(
    *,
    force_reusable_values: bool = False,
    try_non_lazy: bool = False,
    never_lazy: bool = False,
) -> Callable[["T"], "T"]:
    """Returns a decorator for strategy functions.

    If ``force_reusable_values`` is True, the returned strategy will be marked
    with ``.has_reusable_values == True`` even if it uses maps/filters or
    non-reusable strategies internally. This tells our numpy/pandas strategies
    that they can implicitly use such strategies as background values.

    If ``try_non_lazy`` is True, attempt to execute the strategy definition
    function immediately, so that a LazyStrategy is only returned if this
    raises an exception.

    If ``never_lazy`` is True, the decorator performs no lazy-wrapping at all,
    and instead returns the original function.
    """

    def decorator(strategy_definition):
        """A decorator that registers the function as a strategy and makes it
        lazily evaluated."""
        _strategies[strategy_definition.__name__] = signature(strategy_definition)

        if never_lazy:
            assert not try_non_lazy
            # We could potentially support never_lazy + force_reusable_values
            # with a suitable wrapper, but currently there are no callers that
            # request this combination.
            assert not force_reusable_values
            return strategy_definition

        from hypothesis.strategies._internal.lazy import LazyStrategy

        @proxies(strategy_definition)
        def accept(*args, **kwargs):
            if try_non_lazy:
                # Why not try this unconditionally?  Because we'd end up with very
                # deep nesting of recursive strategies - better to be lazy unless we
                # *know* that eager evaluation is the right choice.
                try:
                    return strategy_definition(*args, **kwargs)
                except Exception:
                    # If invoking the strategy definition raises an exception,
                    # wrap that up in a LazyStrategy so it happens again later.
                    pass
            result = LazyStrategy(strategy_definition, args, kwargs)
            if force_reusable_values:
                # Setting `force_has_reusable_values` here causes the recursive
                # property code to set `.has_reusable_values == True`.
                result.force_has_reusable_values = True
                assert result.has_reusable_values
            return result

        accept.is_hypothesis_strategy_function = True
        return accept

    return decorator
