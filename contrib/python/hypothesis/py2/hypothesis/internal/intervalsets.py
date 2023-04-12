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


class IntervalSet(object):
    def __init__(self, intervals):
        self.intervals = tuple(intervals)
        self.offsets = [0]
        for u, v in self.intervals:
            self.offsets.append(self.offsets[-1] + v - u + 1)
        self.size = self.offsets.pop()

    def __len__(self):
        return self.size

    def __iter__(self):
        for u, v in self.intervals:
            for i in range(u, v + 1):
                yield i

    def __getitem__(self, i):
        if i < 0:
            i = self.size + i
        if i < 0 or i >= self.size:
            raise IndexError("Invalid index %d for [0, %d)" % (i, self.size))
        # Want j = maximal such that offsets[j] <= i

        j = len(self.intervals) - 1
        if self.offsets[j] > i:
            hi = j
            lo = 0
            # Invariant: offsets[lo] <= i < offsets[hi]
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self.offsets[mid] <= i:
                    lo = mid
                else:
                    hi = mid
            j = lo
        t = i - self.offsets[j]
        u, v = self.intervals[j]
        r = u + t
        assert r <= v
        return r

    def __repr__(self):
        return "IntervalSet(%r)" % (self.intervals,)

    def index(self, value):
        for offset, (u, v) in zip(self.offsets, self.intervals):
            if u == value:
                return offset
            elif u > value:
                raise ValueError("%d is not in list" % (value,))
            if value <= v:
                return offset + (value - u)
        raise ValueError("%d is not in list" % (value,))

    def index_above(self, value):
        for offset, (u, v) in zip(self.offsets, self.intervals):
            if u >= value:
                return offset
            if value <= v:
                return offset + (value - u)
        return self.size
