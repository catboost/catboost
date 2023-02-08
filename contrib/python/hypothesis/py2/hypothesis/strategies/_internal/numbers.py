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

import hypothesis.internal.conjecture.floats as flt
import hypothesis.internal.conjecture.utils as d
from hypothesis.control import assume, reject
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.floats import float_of
from hypothesis.strategies._internal.strategies import SearchStrategy


class WideRangeIntStrategy(SearchStrategy):

    distribution = d.Sampler([4.0, 8.0, 1.0, 1.0, 0.5])

    sizes = [8, 16, 32, 64, 128]

    def __repr__(self):
        return "WideRangeIntStrategy()"

    def do_draw(self, data):
        size = self.sizes[self.distribution.sample(data)]
        r = data.draw_bits(size)
        sign = r & 1
        r >>= 1
        if sign:
            r = -r
        return int(r)


class BoundedIntStrategy(SearchStrategy):
    """A strategy for providing integers in some interval with inclusive
    endpoints."""

    def __init__(self, start, end):
        SearchStrategy.__init__(self)
        self.start = start
        self.end = end

    def __repr__(self):
        return "BoundedIntStrategy(%d, %d)" % (self.start, self.end)

    def do_draw(self, data):
        return d.integer_range(data, self.start, self.end)


NASTY_FLOATS = sorted(
    [
        0.0,
        0.5,
        1.1,
        1.5,
        1.9,
        1.0 / 3,
        10e6,
        10e-6,
        1.175494351e-38,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        3.402823466e38,
        9007199254740992,
        1 - 10e-6,
        2 + 10e-6,
        1.192092896e-07,
        2.2204460492503131e-016,
    ]
    + [float("inf"), float("nan")] * 5,
    key=flt.float_to_lex,
)
NASTY_FLOATS = list(map(float, NASTY_FLOATS))
NASTY_FLOATS.extend([-x for x in NASTY_FLOATS])

FLOAT_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "getting another float in FloatStrategy"
)


class FloatStrategy(SearchStrategy):
    """Generic superclass for strategies which produce floats."""

    def __init__(self, allow_infinity, allow_nan, width):
        SearchStrategy.__init__(self)
        assert isinstance(allow_infinity, bool)
        assert isinstance(allow_nan, bool)
        assert width in (16, 32, 64)
        self.allow_infinity = allow_infinity
        self.allow_nan = allow_nan
        self.width = width

        self.nasty_floats = [
            float_of(f, self.width) for f in NASTY_FLOATS if self.permitted(f)
        ]
        weights = [0.2 * len(self.nasty_floats)] + [0.8] * len(self.nasty_floats)
        self.sampler = d.Sampler(weights)

    def __repr__(self):
        return "{}(allow_infinity={}, allow_nan={}, width={})".format(
            self.__class__.__name__, self.allow_infinity, self.allow_nan, self.width
        )

    def permitted(self, f):
        assert isinstance(f, float)
        if not self.allow_infinity and math.isinf(f):
            return False
        if not self.allow_nan and math.isnan(f):
            return False
        if self.width < 64:
            try:
                float_of(f, self.width)
                return True
            except OverflowError:  # pragma: no cover
                return False
        return True

    def do_draw(self, data):
        while True:
            data.start_example(FLOAT_STRATEGY_DO_DRAW_LABEL)
            i = self.sampler.sample(data)
            if i == 0:
                result = flt.draw_float(data)
            else:
                result = self.nasty_floats[i - 1]
                flt.write_float(data, result)
            if self.permitted(result):
                data.stop_example()
                if self.width < 64:
                    return float_of(result, self.width)
                return result
            data.stop_example(discard=True)


class FixedBoundedFloatStrategy(SearchStrategy):
    """A strategy for floats distributed between two endpoints.

    The conditional distribution tries to produce values clustered
    closer to one of the ends.
    """

    def __init__(self, lower_bound, upper_bound, width):
        SearchStrategy.__init__(self)
        assert isinstance(lower_bound, float)
        assert isinstance(upper_bound, float)
        assert 0 <= lower_bound < upper_bound
        assert math.copysign(1, lower_bound) == 1, "lower bound may not be -0.0"
        assert width in (16, 32, 64)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width

    def __repr__(self):
        return "FixedBoundedFloatStrategy(%s, %s, %s)" % (
            self.lower_bound,
            self.upper_bound,
            self.width,
        )

    def do_draw(self, data):
        f = self.lower_bound + (
            self.upper_bound - self.lower_bound
        ) * d.fractional_float(data)
        if self.width < 64:
            try:
                f = float_of(f, self.width)
            except OverflowError:  # pragma: no cover
                reject()
        assume(self.lower_bound <= f <= self.upper_bound)
        return f
