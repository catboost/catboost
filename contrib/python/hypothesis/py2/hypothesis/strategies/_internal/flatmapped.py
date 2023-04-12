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

from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.strategies import SearchStrategy


class FlatMapStrategy(SearchStrategy):
    def __init__(self, strategy, expand):
        super(FlatMapStrategy, self).__init__()
        self.flatmapped_strategy = strategy
        self.expand = expand

    def calc_is_empty(self, recur):
        return recur(self.flatmapped_strategy)

    def __repr__(self):
        if not hasattr(self, u"_cached_repr"):
            self._cached_repr = u"%r.flatmap(%s)" % (
                self.flatmapped_strategy,
                get_pretty_function_description(self.expand),
            )
        return self._cached_repr

    def do_draw(self, data):
        source = data.draw(self.flatmapped_strategy)
        expanded_source = self.expand(source)
        check_type(SearchStrategy, expanded_source)
        return data.draw(expanded_source)

    @property
    def branches(self):
        return [
            FlatMapStrategy(strategy=strategy, expand=self.expand)
            for strategy in self.flatmapped_strategy.branches
        ]
