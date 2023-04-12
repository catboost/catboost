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

import collections
import datetime
import decimal
import fractions
import functools
import inspect
import io
import numbers
import sys
import uuid

import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument, ResolutionFailed
from hypothesis.internal.compat import (
    PY2,
    ForwardRef,
    abc,
    binary_type,
    text_type,
    typing_root_type,
)
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.strategies import OneOfStrategy


def type_sorting_key(t):
    """Minimise to None, then non-container types, then container types."""
    if not is_a_type(t):
        raise InvalidArgument("thing=%s must be a type" % (t,))
    if t is None or t is type(None):  # noqa: E721
        return (-1, repr(t))
    if not isinstance(t, type):  # pragma: no cover
        # Some generics in the typing module are not actually types in 3.7
        return (2, repr(t))
    return (int(issubclass(t, abc.Container)), repr(t))


def try_issubclass(thing, superclass):
    thing = getattr(thing, "__origin__", None) or thing
    superclass = getattr(superclass, "__origin__", None) or superclass
    try:
        return issubclass(thing, superclass)
    except (AttributeError, TypeError):  # pragma: no cover
        # Some types can't be the subject or object of an instance or
        # subclass check under Python 3.5
        return False


def is_a_new_type(thing):
    # At runtime, `typing.NewType` returns an identity function rather
    # than an actual type, but we can check whether that thing matches.
    return (
        hasattr(thing, "__supertype__")
        and hasattr(typing, "NewType")
        and inspect.isfunction(thing)
        and getattr(thing, "__module__", None) == "typing"
    )


def is_a_type(thing):
    """Return True if thing is a type or a generic type like thing."""
    return (
        isinstance(thing, type)
        or isinstance(thing, typing_root_type)
        or is_a_new_type(thing)
    )


def is_typing_literal(thing):
    return (
        hasattr(typing, "Literal")
        and getattr(thing, "__origin__", None) == typing.Literal
    )


def from_typing_type(thing):
    # We start with special-case support for Union and Tuple - the latter
    # isn't actually a generic type. Then we handle Literal since it doesn't
    # support `isinstance`. Support for Callable may be added to this section
    # later.
    # We then explicitly error on non-Generic types, which don't carry enough
    # information to sensibly resolve to strategies at runtime.
    # Finally, we run a variation of the subclass lookup in st.from_type
    # among generic types in the lookup.
    import typing

    # Under 3.6 Union is handled directly in st.from_type, as the argument is
    # not an instance of `type`. However, under Python 3.5 Union *is* a type
    # and we have to handle it here, including failing if it has no parameters.
    if hasattr(thing, "__union_params__"):  # pragma: no cover
        args = sorted(thing.__union_params__ or (), key=type_sorting_key)
        if not args:
            raise ResolutionFailed("Cannot resolve Union of no types.")
        return st.one_of([st.from_type(t) for t in args])
    if getattr(thing, "__origin__", None) == tuple or isinstance(
        thing, getattr(typing, "TupleMeta", ())
    ):
        elem_types = getattr(thing, "__tuple_params__", None) or ()
        elem_types += getattr(thing, "__args__", None) or ()
        if (
            getattr(thing, "__tuple_use_ellipsis__", False)
            or len(elem_types) == 2
            and elem_types[-1] is Ellipsis
        ):
            return st.lists(st.from_type(elem_types[0])).map(tuple)
        elif len(elem_types) == 1 and elem_types[0] == ():
            return st.tuples()  # Empty tuple; see issue #1583
        return st.tuples(*map(st.from_type, elem_types))
    if (
        hasattr(typing, "Final") and getattr(thing, "__origin__", None) == typing.Final
    ):  # pragma: no cover  # new in Python 3.8
        return st.one_of([st.from_type(t) for t in thing.__args__])
    if is_typing_literal(thing):  # pragma: no cover  # new in Python 3.8
        args_dfs_stack = list(thing.__args__)
        literals = []
        while args_dfs_stack:
            arg = args_dfs_stack.pop()
            if is_typing_literal(arg):
                args_dfs_stack.extend(reversed(arg.__args__))
            else:
                literals.append(arg)
        return st.sampled_from(literals)
    if isinstance(thing, typing.TypeVar):
        if getattr(thing, "__bound__", None) is not None:
            strat = unwrap_strategies(st.from_type(thing.__bound__))
            if not isinstance(strat, OneOfStrategy):
                return strat
            # The bound was a union, or we resolved it as a union of subtypes,
            # so we need to unpack the strategy to ensure consistency across uses.
            # This incantation runs a sampled_from over the strategies inferred for
            # each part of the union, wraps that in shared so that we only generate
            # from one type per testcase, and flatmaps that back to instances.
            return st.shared(
                st.sampled_from(strat.original_strategies), key="typevar=%r" % (thing,)
            ).flatmap(lambda s: s)
        if getattr(thing, "__constraints__", None):
            return st.shared(
                st.sampled_from(thing.__constraints__), key="typevar=%r" % (thing,)
            ).flatmap(st.from_type)
        # Constraints may be None or () on various Python versions.
        return st.text()  # An arbitrary type for the typevar
    # Now, confirm that we're dealing with a generic type as we expected
    if not isinstance(thing, typing_root_type):  # pragma: no cover
        raise ResolutionFailed("Cannot resolve %s to a strategy" % (thing,))

    # Some "generic" classes are not generic *in* anything - for example both
    # Hashable and Sized have `__args__ == ()` on Python 3.7 or later.
    # (In 3.6 they're just aliases for the collections.abc classes)
    origin = getattr(thing, "__origin__", thing)
    if (
        typing.Hashable is not abc.Hashable
        and origin in vars(abc).values()
        and len(getattr(thing, "__args__", None) or []) == 0
    ):  # pragma: no cover  # impossible on 3.6 where we measure coverage.
        return st.from_type(origin)

    # Parametrised generic types have their __origin__ attribute set to the
    # un-parametrised version, which we need to use in the subclass checks.
    # e.g.:     typing.List[int].__origin__ == typing.List
    mapping = {
        k: v
        for k, v in _global_type_lookup.items()
        if isinstance(k, typing_root_type) and try_issubclass(k, thing)
    }
    if typing.Dict in mapping:
        # The subtype relationships between generic and concrete View types
        # are sometimes inconsistent under Python 3.5, so we pop them out to
        # preserve our invariant that all examples of from_type(T) are
        # instances of type T - and simplify the strategy for abstract types
        # such as Container
        for t in (typing.KeysView, typing.ValuesView, typing.ItemsView):
            mapping.pop(t, None)
    if len(mapping) > 1:
        # issubclass treats bytestring as a kind of sequence, which it is,
        # but treating it as such breaks everything else when it is presumed
        # to be a generic sequence or container that could hold any item.
        # Except for sequences of integers, or unions which include integer!
        # See https://github.com/HypothesisWorks/hypothesis/issues/2257
        #
        # This block drops ByteString from the types that can be generated
        # if there is more than one allowed type, and the element type is
        # not either `int` or a Union with `int` as one of its elements.
        elem_type = (getattr(thing, "__args__", None) or ["not int"])[0]
        if getattr(elem_type, "__origin__", None) is typing.Union:
            union_elems = elem_type.__args__
        elif hasattr(elem_type, "__union_params__"):  # pragma: no cover
            union_elems = elem_type.__union_params__  # python 3.5 only
        else:
            union_elems = ()
        if PY2 or not any(
            isinstance(T, type) and issubclass(int, T)
            for T in list(union_elems) + [elem_type]
        ):
            mapping.pop(typing.ByteString, None)
    strategies = [
        v if isinstance(v, st.SearchStrategy) else v(thing)
        for k, v in mapping.items()
        if sum(try_issubclass(k, T) for T in mapping) == 1
    ]
    empty = ", ".join(repr(s) for s in strategies if s.is_empty)
    if empty or not strategies:  # pragma: no cover
        raise ResolutionFailed(
            "Could not resolve %s to a strategy; consider using "
            "register_type_strategy" % (empty or thing,)
        )
    return st.one_of(strategies)


def can_cast(type, value):
    """Determine if value can be cast to type."""
    try:
        type(value)
        return True
    except Exception:
        return False


_global_type_lookup = {
    # Types with core Hypothesis strategies
    type(None): st.none(),
    bool: st.booleans(),
    int: st.integers(),
    float: st.floats(),
    complex: st.complex_numbers(),
    fractions.Fraction: st.fractions(),
    decimal.Decimal: st.decimals(),
    text_type: st.text(),
    binary_type: st.binary(),
    datetime.datetime: st.datetimes(),
    datetime.date: st.dates(),
    datetime.time: st.times(),
    datetime.timedelta: st.timedeltas(),
    uuid.UUID: st.uuids(),
    tuple: st.builds(tuple),
    list: st.builds(list),
    set: st.builds(set),
    abc.MutableSet: st.builds(set),
    frozenset: st.builds(frozenset),
    dict: st.builds(dict),
    type(lambda: None): st.functions(),
    # Built-in types
    type(Ellipsis): st.just(Ellipsis),
    type(NotImplemented): st.just(NotImplemented),
    bytearray: st.binary().map(bytearray),
    memoryview: st.binary().map(memoryview),
    numbers.Real: st.floats(),
    numbers.Rational: st.fractions(),
    numbers.Number: st.complex_numbers(),
    numbers.Integral: st.integers(),
    numbers.Complex: st.complex_numbers(),
    slice: st.builds(
        slice,
        st.none() | st.integers(),
        st.none() | st.integers(),
        st.none() | st.integers(),
    )
    # Pull requests with more types welcome!
}

if PY2:
    # xrange's |stop - start| must fit in a C long
    int64_strat = st.integers(-sys.maxint // 2, sys.maxint // 2)  # noqa
    _global_type_lookup.update(
        {
            int: st.integers().filter(lambda x: isinstance(x, int)),
            long: st.integers().map(long),  # noqa
            xrange: st.integers(min_value=0, max_value=sys.maxint).map(xrange)  # noqa
            | st.builds(xrange, int64_strat, int64_strat)  # noqa
            | st.builds(
                xrange, int64_strat, int64_strat, int64_strat.filter(bool)  # noqa
            ),
        }
    )
else:
    _global_type_lookup.update(
        {
            range: st.integers(min_value=0).map(range)
            | st.builds(range, st.integers(), st.integers())
            | st.builds(range, st.integers(), st.integers(), st.integers().filter(bool))
        }
    )

_global_type_lookup[type] = st.sampled_from(
    [type(None)] + sorted(_global_type_lookup, key=str)
)

try:
    from hypothesis.extra.pytz import timezones

    _global_type_lookup[datetime.tzinfo] = timezones()
except ImportError:  # pragma: no cover
    pass
try:  # pragma: no cover
    import numpy as np
    from hypothesis.extra.numpy import (
        arrays,
        array_shapes,
        scalar_dtypes,
        nested_dtypes,
        from_dtype,
        integer_dtypes,
        unsigned_integer_dtypes,
    )

    _global_type_lookup.update(
        {
            np.dtype: nested_dtypes(),
            np.ndarray: arrays(scalar_dtypes(), array_shapes(max_dims=2)),
        }
    )
except ImportError:  # pragma: no cover
    np = None

try:
    import typing
except ImportError:  # pragma: no cover
    pass
else:
    _global_type_lookup.update(
        {
            typing.ByteString: st.binary() | st.binary().map(bytearray),
            # Reversible is somehow a subclass of Hashable, so we tuplize it.
            # See also the discussion at https://bugs.python.org/issue39046
            typing.Reversible: st.lists(st.integers()).map(tuple),  # type: ignore
            typing.SupportsAbs: st.one_of(
                st.booleans(),
                st.integers(),
                st.floats(),
                st.complex_numbers(),
                st.fractions(),
                st.decimals(),
                st.timedeltas(),
            ),
            typing.SupportsComplex: st.one_of(
                st.booleans(),
                st.integers(),
                st.floats(),
                st.complex_numbers(),
                st.decimals(),
                st.fractions(),
            ),
            typing.SupportsFloat: st.one_of(
                st.booleans(),
                st.integers(),
                st.floats(),
                st.decimals(),
                st.fractions(),
                # with floats its far more annoying to capture all
                # the magic in a regex. so we just stringify some floats
                st.floats().map(str),
            ),
            typing.SupportsInt: st.one_of(
                st.booleans(),
                st.integers(),
                st.floats(),
                st.uuids(),
                st.decimals(),
                # this generates strings that should able to be parsed into integers
                st.from_regex(r"-?\d+", fullmatch=True).filter(
                    lambda value: can_cast(int, value)
                ),
            ),
            # xIO are only available in .io on Python 3.5, but available directly
            # as typing.*IO from 3.6 onwards and mypy 0.730 errors on the compat form.
            typing.io.BinaryIO: st.builds(io.BytesIO, st.binary()),  # type: ignore
            typing.io.TextIO: st.builds(io.StringIO, st.text()),  # type: ignore
        }
    )

    try:
        # These aren't present in the typing module backport.
        _global_type_lookup[typing.SupportsBytes] = st.one_of(
            st.booleans(),
            st.binary(),
            st.integers(0, 255),
            # As with Reversible, we tuplize this for compatibility with Hashable.
            st.lists(st.integers(0, 255)).map(tuple),  # type: ignore
        )
        _global_type_lookup[typing.SupportsRound] = st.one_of(
            st.booleans(), st.integers(), st.floats(), st.decimals(), st.fractions()
        )
    except AttributeError:  # pragma: no cover
        pass
    try:
        strat = st.integers() | st.booleans()
        if np is not None:  # pragma: no branch
            strat |= (unsigned_integer_dtypes() | integer_dtypes()).flatmap(from_dtype)
        _global_type_lookup[typing.SupportsIndex] = strat  # type: ignore
    except AttributeError:  # pragma: no cover
        pass

    def register(type_, fallback=None):
        if isinstance(type_, str):
            # Use the name of generic types which are not available on all
            # versions, and the function just won't be added to the registry
            type_ = getattr(typing, type_, None)
            if type_ is None:  # pragma: no cover
                return lambda f: f

        def inner(func):
            if fallback is None:
                _global_type_lookup[type_] = func
                return func

            @functools.wraps(func)
            def really_inner(thing):
                if getattr(thing, "__args__", None) is None:
                    return fallback
                return func(thing)

            _global_type_lookup[type_] = really_inner
            return really_inner

        return inner

    @register("Type")
    def resolve_Type(thing):
        if thing.__args__ is None:
            return st.just(type)
        args = (thing.__args__[0],)
        if getattr(args[0], "__origin__", None) is typing.Union:
            args = args[0].__args__
        elif hasattr(args[0], "__union_params__"):  # pragma: no cover
            args = args[0].__union_params__
        if isinstance(ForwardRef, type):  # pragma: no cover
            # Duplicate check from from_type here - only paying when needed.
            for a in args:
                if type(a) == ForwardRef:
                    raise ResolutionFailed(
                        "thing=%s cannot be resolved.  Upgrading to "
                        "python>=3.6 may fix this problem via improvements "
                        "to the typing module." % (thing,)
                    )
        return st.sampled_from(sorted(args, key=type_sorting_key))

    @register(typing.List, st.builds(list))
    def resolve_List(thing):
        return st.lists(st.from_type(thing.__args__[0]))

    @register(typing.Set, st.builds(set))
    def resolve_Set(thing):
        return st.sets(st.from_type(thing.__args__[0]))

    @register(typing.FrozenSet, st.builds(frozenset))
    def resolve_FrozenSet(thing):
        return st.frozensets(st.from_type(thing.__args__[0]))

    @register(typing.Dict, st.builds(dict))
    def resolve_Dict(thing):
        # If thing is a Collection instance, we need to fill in the values
        keys_vals = [st.from_type(t) for t in thing.__args__] * 2
        return st.dictionaries(keys_vals[0], keys_vals[1])

    @register("DefaultDict", st.builds(collections.defaultdict))
    def resolve_DefaultDict(thing):
        return resolve_Dict(thing).map(lambda d: collections.defaultdict(None, d))

    @register(typing.ItemsView, st.builds(dict).map(dict.items))
    def resolve_ItemsView(thing):
        return resolve_Dict(thing).map(dict.items)

    @register(typing.KeysView, st.builds(dict).map(dict.keys))
    def resolve_KeysView(thing):
        return st.dictionaries(st.from_type(thing.__args__[0]), st.none()).map(
            dict.keys
        )

    @register(typing.ValuesView, st.builds(dict).map(dict.values))
    def resolve_ValuesView(thing):
        return st.dictionaries(st.integers(), st.from_type(thing.__args__[0])).map(
            dict.values
        )

    @register(typing.Iterator, st.iterables(st.nothing()))
    def resolve_Iterator(thing):
        return st.iterables(st.from_type(thing.__args__[0]))

    @register(typing.Callable, st.functions())
    def resolve_Callable(thing):
        # Generated functions either accept no arguments, or arbitrary arguments.
        # This is looser than ideal, but anything tighter would generally break
        # use of keyword arguments and we'd rather not force positional-only.
        if not thing.__args__:  # pragma: no cover  # varies by minor version
            return st.functions()
        return st.functions(
            like=(lambda: None) if len(thing.__args__) == 1 else (lambda *a, **k: None),
            returns=st.from_type(thing.__args__[-1]),
        )
