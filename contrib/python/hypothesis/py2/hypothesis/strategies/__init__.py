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

from hypothesis.strategies._internal import SearchStrategy
from hypothesis.strategies._internal.core import (
    DataObject,
    _strategies,
    binary,
    booleans,
    builds,
    characters,
    complex_numbers,
    composite,
    data,
    dates,
    datetimes,
    decimals,
    deferred,
    dictionaries,
    emails,
    fixed_dictionaries,
    floats,
    fractions,
    from_regex,
    from_type,
    frozensets,
    functions,
    integers,
    iterables,
    just,
    lists,
    none,
    nothing,
    one_of,
    permutations,
    random_module,
    randoms,
    recursive,
    register_type_strategy,
    runner,
    sampled_from,
    sets,
    shared,
    slices,
    text,
    timedeltas,
    times,
    tuples,
    uuids,
)

# The implementation of all of these lives in `_strategies.py`, but we
# re-export them via this module to avoid exposing implementation details
# to over-zealous tab completion in editors that do not respect __all__.


__all__ = [
    "binary",
    "booleans",
    "builds",
    "characters",
    "complex_numbers",
    "composite",
    "data",
    "DataObject",
    "dates",
    "datetimes",
    "decimals",
    "deferred",
    "dictionaries",
    "emails",
    "fixed_dictionaries",
    "floats",
    "fractions",
    "from_regex",
    "from_type",
    "frozensets",
    "functions",
    "integers",
    "iterables",
    "just",
    "lists",
    "none",
    "nothing",
    "one_of",
    "permutations",
    "random_module",
    "randoms",
    "recursive",
    "register_type_strategy",
    "runner",
    "sampled_from",
    "sets",
    "shared",
    "slices",
    "text",
    "timedeltas",
    "times",
    "tuples",
    "uuids",
    "SearchStrategy",
]

assert _strategies.issubset(set(__all__)), (
    _strategies - set(__all__),
    set(__all__) - _strategies,
)
del absolute_import, division, print_function
_public = {n for n in dir() if n[0] not in "_@"}
assert set(__all__) == _public, (set(__all__) - _public, _public - set(__all__))
del _public
