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

from math import fabs, isinf, isnan, sqrt
from sys import float_info


def cathetus(h, a):
    """Given the lengths of the hypotenuse and a side of a right triangle,
    return the length of the other side.

    A companion to the C99 hypot() function.  Some care is needed to avoid
    underflow in the case of small arguments, and overflow in the case of
    large arguments as would occur for the naive implementation as
    sqrt(h*h - a*a).  The behaviour with respect the non-finite arguments
    (NaNs and infinities) is designed to be as consistent as possible with
    the C99 hypot() specifications.

    This function relies on the system ``sqrt`` function and so, like it,
    may be inaccurate up to a relative error of (around) floating-point
    epsilon.

    Based on the C99 implementation https://github.com/jjgreen/cathetus
    """
    if isnan(h):
        return float(u"nan")

    if isinf(h):
        if isinf(a):
            return float(u"nan")
        else:
            # Deliberately includes the case when isnan(a), because the
            # C99 standard mandates that hypot(inf, nan) == inf
            return float(u"inf")

    h = fabs(h)
    a = fabs(a)

    if h < a:
        return float(u"nan")

    # Thanks to floating-point precision issues when performing multiple
    # operations on extremely large or small values, we may rarely calculate
    # a side length that is longer than the hypotenuse.  This is clearly an
    # error, so we clip to the hypotenuse as the best available estimate.
    if h > sqrt(float_info.max):
        if h > float_info.max / 2:
            b = sqrt(h - a) * sqrt(h / 2 + a / 2) * sqrt(2)
        else:
            b = sqrt(h - a) * sqrt(h + a)
    elif h < sqrt(float_info.min):
        b = sqrt(h - a) * sqrt(h + a)
    else:
        b = sqrt((h - a) * (h + a))
    return min(b, h)
