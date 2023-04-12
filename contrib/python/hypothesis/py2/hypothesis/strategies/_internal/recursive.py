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

from contextlib import contextmanager

from hypothesis.errors import InvalidArgument
from hypothesis.internal.lazyformat import lazyformat
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.strategies import OneOfStrategy, SearchStrategy


class LimitReached(BaseException):
    pass


class LimitedStrategy(SearchStrategy):
    def __init__(self, strategy):
        super(LimitedStrategy, self).__init__()
        self.base_strategy = strategy
        self.marker = 0
        self.currently_capped = False

    def __repr__(self):
        return "LimitedStrategy(%r)" % (self.base_strategy,)

    def do_validate(self):
        self.base_strategy.validate()

    def do_draw(self, data):
        assert self.currently_capped
        if self.marker <= 0:
            raise LimitReached()
        self.marker -= 1
        return data.draw(self.base_strategy)

    @contextmanager
    def capped(self, max_templates):
        assert not self.currently_capped
        try:
            self.currently_capped = True
            self.marker = max_templates
            yield
        finally:
            self.currently_capped = False


class RecursiveStrategy(SearchStrategy):
    def __init__(self, base, extend, max_leaves):
        self.max_leaves = max_leaves
        self.base = base
        self.limited_base = LimitedStrategy(base)
        self.extend = extend

        strategies = [self.limited_base, self.extend(self.limited_base)]
        while 2 ** (len(strategies) - 1) <= max_leaves:
            strategies.append(extend(OneOfStrategy(tuple(strategies))))
        self.strategy = OneOfStrategy(strategies)

    def __repr__(self):
        if not hasattr(self, "_cached_repr"):
            self._cached_repr = "recursive(%r, %s, max_leaves=%d)" % (
                self.base,
                get_pretty_function_description(self.extend),
                self.max_leaves,
            )
        return self._cached_repr

    def do_validate(self):
        if not isinstance(self.base, SearchStrategy):
            raise InvalidArgument(
                "Expected base to be SearchStrategy but got %r" % (self.base,)
            )
        extended = self.extend(self.limited_base)
        if not isinstance(extended, SearchStrategy):
            raise InvalidArgument(
                "Expected extend(%r) to be a SearchStrategy but got %r"
                % (self.limited_base, extended)
            )
        self.limited_base.validate()
        self.extend(self.limited_base).validate()

    def do_draw(self, data):
        count = 0
        while True:
            try:
                with self.limited_base.capped(self.max_leaves):
                    return data.draw(self.strategy)
            except LimitReached:
                # Workaround for possible coverage bug - this branch is definitely
                # covered but for some reason is showing up as not covered.
                if count == 0:  # pragma: no branch
                    data.note_event(
                        lazyformat(
                            "Draw for %r exceeded max_leaves and had to be retried",
                            self,
                        )
                    )
                count += 1
