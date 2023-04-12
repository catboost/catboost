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

from hypothesis.control import note
from hypothesis.errors import InvalidState
from hypothesis.internal.reflection import arg_string, nicerepr, proxies
from hypothesis.strategies._internal.strategies import SearchStrategy


class FunctionStrategy(SearchStrategy):
    supports_find = False

    def __init__(self, like, returns):
        super(FunctionStrategy, self).__init__()
        self.like = like
        self.returns = returns

    def calc_is_empty(self, recur):
        return recur(self.returns)

    def do_draw(self, data):
        @proxies(self.like)
        def inner(*args, **kwargs):
            if data.frozen:
                raise InvalidState(
                    "This generated %s function can only be called within the "
                    "scope of the @given that created it." % (nicerepr(self.like),)
                )
            val = data.draw(self.returns)
            note(
                "Called function: %s(%s) -> %r"
                % (nicerepr(self.like), arg_string(self.like, args, kwargs), val)
            )
            return val

        return inner
