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

import contextlib
import random
import sys

from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import integer_types

RANDOMS_TO_MANAGE = [random]  # type: list


class NumpyRandomWrapper(object):
    def __init__(self):
        assert "numpy" in sys.modules
        # This class provides a shim that matches the numpy to stdlib random,
        # and lets us avoid importing Numpy until it's already in use.
        import numpy.random

        self.seed = numpy.random.seed
        self.getstate = numpy.random.get_state
        self.setstate = numpy.random.set_state


def register_random(r):
    # type: (random.Random) -> None
    """Register the given Random instance for management by Hypothesis.

    You can pass ``random.Random`` instances (or other objects with seed,
    getstate, and setstate methods) to ``register_random(r)`` to have their
    states seeded and restored in the same way as the global PRNGs from the
    ``random`` and ``numpy.random`` modules.

    All global PRNGs, from e.g. simulation or scheduling frameworks, should
    be registered to prevent flaky tests.  Hypothesis will ensure that the
    PRNG state is consistent for all test runs, or reproducibly varied if you
    choose to use the :func:`~hypothesis.strategies.random_module` strategy.
    """
    if not (hasattr(r, "seed") and hasattr(r, "getstate") and hasattr(r, "setstate")):
        raise InvalidArgument("r=%r does not have all the required methods" % (r,))
    if r not in RANDOMS_TO_MANAGE:
        RANDOMS_TO_MANAGE.append(r)


def get_seeder_and_restorer(seed=0):
    """Return a pair of functions which respectively seed all and restore
    the state of all registered PRNGs.

    This is used by the core engine via `deterministic_PRNG`, and by users
    via `register_random`.  We support registration of additional random.Random
    instances (or other objects with seed, getstate, and setstate methods)
    to force determinism on simulation or scheduling frameworks which avoid
    using the global random state.  See e.g. #1709.
    """
    assert isinstance(seed, integer_types) and 0 <= seed < 2 ** 32
    states = []  # type: list

    if "numpy" in sys.modules and not any(
        isinstance(x, NumpyRandomWrapper) for x in RANDOMS_TO_MANAGE
    ):
        RANDOMS_TO_MANAGE.append(NumpyRandomWrapper())

    def seed_all():
        assert not states
        for r in RANDOMS_TO_MANAGE:
            states.append(r.getstate())
            r.seed(seed)

    def restore_all():
        assert len(states) == len(RANDOMS_TO_MANAGE)
        for r, state in zip(RANDOMS_TO_MANAGE, states):
            r.setstate(state)
        del states[:]

    return seed_all, restore_all


@contextlib.contextmanager
def deterministic_PRNG(seed=0):
    """Context manager that handles random.seed without polluting global state.

    See issue #1255 and PR #1295 for details and motivation - in short,
    leaving the global pseudo-random number generator (PRNG) seeded is a very
    bad idea in principle, and breaks all kinds of independence assumptions
    in practice.
    """
    seed_all, restore_all = get_seeder_and_restorer(seed)
    seed_all()
    try:
        yield
    finally:
        restore_all()
