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

import enum
import hashlib
import heapq
import sys
from collections import OrderedDict
from fractions import Fraction

from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import (
    abc,
    bit_length,
    floor,
    hrange,
    int_from_bytes,
    qualname,
    str_to_bytes,
)
from hypothesis.internal.floats import int_to_float

LABEL_MASK = 2 ** 64 - 1


def calc_label_from_name(name):
    hashed = hashlib.sha384(str_to_bytes(name)).digest()
    return int_from_bytes(hashed[:8])


def calc_label_from_cls(cls):
    return calc_label_from_name(qualname(cls))


def combine_labels(*labels):
    label = 0
    for l in labels:
        label = (label << 1) & LABEL_MASK
        label ^= l
    return label


INTEGER_RANGE_DRAW_LABEL = calc_label_from_name("another draw in integer_range()")
BIASED_COIN_LABEL = calc_label_from_name("biased_coin()")
SAMPLE_IN_SAMPLER_LABLE = calc_label_from_name("a sample() in Sampler")
ONE_FROM_MANY_LABEL = calc_label_from_name("one more from many()")


def integer_range(data, lower, upper, center=None):
    assert lower <= upper
    if lower == upper:
        # Write a value even when this is trivial so that when a bound depends
        # on other values we don't suddenly disappear when the gap shrinks to
        # zero - if that happens then often the data stream becomes misaligned
        # and we fail to shrink in cases where we really should be able to.
        data.draw_bits(1, forced=0)
        return int(lower)

    if center is None:
        center = lower
    center = min(max(center, lower), upper)

    if center == upper:
        above = False
    elif center == lower:
        above = True
    else:
        above = boolean(data)

    if above:
        gap = upper - center
    else:
        gap = center - lower

    assert gap > 0

    bits = bit_length(gap)
    probe = gap + 1

    if bits > 24 and data.draw_bits(3):
        # For large ranges, we combine the uniform random distribution from draw_bits
        # with the weighting scheme used by WideRangeIntStrategy with moderate chance.
        # Cutoff at 2 ** 24 so unicode choice is uniform but 32bit distribution is not.
        idx = Sampler([4.0, 8.0, 1.0, 1.0, 0.5]).sample(data)
        sizes = [8, 16, 32, 64, 128]
        bits = min(bits, sizes[idx])

    while probe > gap:
        data.start_example(INTEGER_RANGE_DRAW_LABEL)
        probe = data.draw_bits(bits)
        data.stop_example(discard=probe > gap)

    if above:
        result = center + probe
    else:
        result = center - probe

    assert lower <= result <= upper
    return int(result)


def check_sample(values, strategy_name):
    if "numpy" in sys.modules and isinstance(values, sys.modules["numpy"].ndarray):
        if values.ndim != 1:
            raise InvalidArgument(
                (
                    "Only one-dimensional arrays are supported for sampling, "
                    "and the given value has {ndim} dimensions (shape "
                    "{shape}).  This array would give samples of array slices "
                    "instead of elements!  Use np.ravel(values) to convert "
                    "to a one-dimensional array, or tuple(values) if you "
                    "want to sample slices."
                ).format(ndim=values.ndim, shape=values.shape)
            )
    elif not isinstance(values, (OrderedDict, abc.Sequence, enum.EnumMeta)):
        raise InvalidArgument(
            "Cannot sample from {values}, not an ordered collection. "
            "Hypothesis goes to some length to ensure that the {strategy} "
            "strategy has stable results between runs. To replay a saved "
            "example, the sampled values must have the same iteration order "
            "on every run - ruling out sets, dicts, etc due to hash "
            "randomisation. Most cases can simply use `sorted(values)`, but "
            "mixed types or special values such as math.nan require careful "
            "handling - and note that when simplifying an example, "
            "Hypothesis treats earlier values as simpler.".format(
                values=repr(values), strategy=strategy_name
            )
        )
    return tuple(values)


def choice(data, values):
    return values[integer_range(data, 0, len(values) - 1)]


FLOAT_PREFIX = 0b1111111111 << 52
FULL_FLOAT = int_to_float(FLOAT_PREFIX | ((2 << 53) - 1)) - 1


def fractional_float(data):
    return (int_to_float(FLOAT_PREFIX | data.draw_bits(52)) - 1) / FULL_FLOAT


def boolean(data):
    return bool(data.draw_bits(1))


def biased_coin(data, p):
    """Return True with probability p (assuming a uniform generator),
    shrinking towards False."""
    data.start_example(BIASED_COIN_LABEL)
    while True:
        # The logic here is a bit complicated and special cased to make it
        # play better with the shrinker.

        # We imagine partitioning the real interval [0, 1] into 256 equal parts
        # and looking at each part and whether its interior is wholly <= p
        # or wholly >= p. At most one part can be neither.

        # We then pick a random part. If it's wholly on one side or the other
        # of p then we use that as the answer. If p is contained in the
        # interval then we start again with a new probability that is given
        # by the fraction of that interval that was <= our previous p.

        # We then take advantage of the fact that we have control of the
        # labelling to make this shrink better, using the following tricks:

        # If p is <= 0 or >= 1 the result of this coin is certain. We make sure
        # to write a byte to the data stream anyway so that these don't cause
        # difficulties when shrinking.
        if p <= 0:
            data.draw_bits(1, forced=0)
            result = False
        elif p >= 1:
            data.draw_bits(1, forced=1)
            result = True
        else:
            falsey = floor(256 * (1 - p))
            truthy = floor(256 * p)
            remainder = 256 * p - truthy

            if falsey + truthy == 256:
                if isinstance(p, Fraction):
                    m = p.numerator
                    n = p.denominator
                else:
                    m, n = p.as_integer_ratio()
                assert n & (n - 1) == 0, n  # n is a power of 2
                assert n > m > 0
                truthy = m
                falsey = n - m
                bits = bit_length(n) - 1
                partial = False
            else:
                bits = 8
                partial = True

            i = data.draw_bits(bits)

            # We always label the region that causes us to repeat the loop as
            # 255 so that shrinking this byte never causes us to need to draw
            # more data.
            if partial and i == 255:
                p = remainder
                continue
            if falsey == 0:
                # Every other partition is truthy, so the result is true
                result = True
            elif truthy == 0:
                # Every other partition is falsey, so the result is false
                result = False
            elif i <= 1:
                # We special case so that zero is always false and 1 is always
                # true which makes shrinking easier because we can always
                # replace a truthy block with 1. This has the slightly weird
                # property that shrinking from 2 to 1 can cause the result to
                # grow, but the shrinker always tries 0 and 1 first anyway, so
                # this will usually be fine.
                result = bool(i)
            else:
                # Originally everything in the region 0 <= i < falsey was false
                # and everything above was true. We swapped one truthy element
                # into this region, so the region becomes 0 <= i <= falsey
                # except for i = 1. We know i > 1 here, so the test for truth
                # becomes i > falsey.
                result = i > falsey
        break
    data.stop_example()
    return result


class Sampler(object):
    """Sampler based on Vose's algorithm for the alias method. See
    http://www.keithschwarz.com/darts-dice-coins/ for a good explanation.

    The general idea is that we store a table of triples (base, alternate, p).
    base. We then pick a triple uniformly at random, and choose its alternate
    value with probability p and else choose its base value. The triples are
    chosen so that the resulting mixture has the right distribution.

    We maintain the following invariants to try to produce good shrinks:

    1. The table is in lexicographic (base, alternate) order, so that choosing
       an earlier value in the list always lowers (or at least leaves
       unchanged) the value.
    2. base[i] < alternate[i], so that shrinking the draw always results in
       shrinking the chosen element.
    """

    def __init__(self, weights):

        n = len(weights)

        self.table = [[i, None, None] for i in hrange(n)]

        total = sum(weights)

        num_type = type(total)

        zero = num_type(0)
        one = num_type(1)

        small = []
        large = []

        probabilities = [w / total for w in weights]
        scaled_probabilities = []

        for i, p in enumerate(probabilities):
            scaled = p * n
            scaled_probabilities.append(scaled)
            if scaled == 1:
                self.table[i][2] = zero
            elif scaled < 1:
                small.append(i)
            else:
                large.append(i)
        heapq.heapify(small)
        heapq.heapify(large)

        while small and large:
            lo = heapq.heappop(small)
            hi = heapq.heappop(large)

            assert lo != hi
            assert scaled_probabilities[hi] > one
            assert self.table[lo][1] is None
            self.table[lo][1] = hi
            self.table[lo][2] = one - scaled_probabilities[lo]
            scaled_probabilities[hi] = (
                scaled_probabilities[hi] + scaled_probabilities[lo]
            ) - one

            if scaled_probabilities[hi] < 1:
                heapq.heappush(small, hi)
            elif scaled_probabilities[hi] == 1:
                self.table[hi][2] = zero
            else:
                heapq.heappush(large, hi)
        while large:
            self.table[large.pop()][2] = zero
        while small:
            self.table[small.pop()][2] = zero

        for entry in self.table:
            assert entry[2] is not None
            if entry[1] is None:
                entry[1] = entry[0]
            elif entry[1] < entry[0]:
                entry[0], entry[1] = entry[1], entry[0]
                entry[2] = one - entry[2]
        self.table.sort()

    def sample(self, data):
        data.start_example(SAMPLE_IN_SAMPLER_LABLE)
        i = integer_range(data, 0, len(self.table) - 1)
        base, alternate, alternate_chance = self.table[i]
        use_alternate = biased_coin(data, alternate_chance)
        data.stop_example()
        if use_alternate:
            return alternate
        else:
            return base


class many(object):
    """Utility class for collections. Bundles up the logic we use for "should I
    keep drawing more values?" and handles starting and stopping examples in
    the right place.

    Intended usage is something like:

    elements = many(data, ...)
    while elements.more():
        add_stuff_to_result()
    """

    def __init__(self, data, min_size, max_size, average_size):
        assert 0 <= min_size <= average_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        self.data = data
        self.stopping_value = 1 - 1.0 / (1 + average_size)
        self.count = 0
        self.rejections = 0
        self.drawn = False
        self.force_stop = False
        self.rejected = False

    def more(self):
        """Should I draw another element to add to the collection?"""
        if self.drawn:
            self.data.stop_example(discard=self.rejected)

        self.drawn = True
        self.rejected = False

        self.data.start_example(ONE_FROM_MANY_LABEL)

        if self.min_size == self.max_size:
            should_continue = self.count < self.min_size
        elif self.force_stop:
            should_continue = False
        else:
            if self.count < self.min_size:
                p_continue = 1.0
            elif self.count >= self.max_size:
                p_continue = 0.0
            else:
                p_continue = self.stopping_value
            should_continue = biased_coin(self.data, p_continue)

        if should_continue:
            self.count += 1
            return True
        else:
            self.data.stop_example()
            return False

    def reject(self):
        """Reject the last example (i.e. don't count it towards our budget of
        elements because it's not going to go in the final collection)."""
        assert self.count > 0
        self.count -= 1
        self.rejections += 1
        self.rejected = True
        # We set a minimum number of rejections before we give up to avoid
        # failing too fast when we reject the first draw.
        if self.rejections > max(3, 2 * self.count):
            if self.count < self.min_size:
                self.data.mark_invalid()
            else:
                self.force_stop = True
