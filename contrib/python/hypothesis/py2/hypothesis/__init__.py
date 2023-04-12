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

"""Hypothesis is a library for writing unit tests which are parametrized by
some source of data.

It verifies your code against a wide range of input and minimizes any
failing examples it finds.
"""

from __future__ import absolute_import, division, print_function

from hypothesis._settings import (
    HealthCheck,
    Phase,
    PrintSettings,
    Verbosity,
    settings,
    unlimited,
)
from hypothesis.control import assume, event, note, reject, target
from hypothesis.core import example, find, given, reproduce_failure, seed
from hypothesis.internal.entropy import register_random
from hypothesis.utils.conventions import infer
from hypothesis.version import __version__, __version_info__

__all__ = [
    "settings",
    "Verbosity",
    "HealthCheck",
    "Phase",
    "PrintSettings",
    "assume",
    "reject",
    "seed",
    "given",
    "unlimited",
    "reproduce_failure",
    "find",
    "example",
    "note",
    "event",
    "infer",
    "register_random",
    "target",
    "__version__",
    "__version_info__",
]
