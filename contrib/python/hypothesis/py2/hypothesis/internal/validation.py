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

import decimal
import math
from numbers import Rational, Real

from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import integer_types
from hypothesis.internal.coverage import check_function


@check_function
def check_type(typ, arg, name=""):
    if name:
        name += "="
    if not isinstance(arg, typ):
        if isinstance(typ, tuple) and len(typ) == 1:
            typ = typ[0]
        if isinstance(typ, tuple):
            typ_string = "one of %s" % (", ".join(t.__name__ for t in typ))
        else:
            typ_string = typ.__name__
        raise InvalidArgument(
            "Expected %s but got %s%r (type=%s)"
            % (typ_string, name, arg, type(arg).__name__)
        )


@check_function
def check_valid_integer(value):
    """Checks that value is either unspecified, or a valid integer.

    Otherwise raises InvalidArgument.
    """
    if value is None:
        return
    check_type(integer_types, value)


@check_function
def check_valid_bound(value, name):
    """Checks that value is either unspecified, or a valid interval bound.

    Otherwise raises InvalidArgument.
    """
    if value is None or isinstance(value, integer_types + (Rational,)):
        return
    if not isinstance(value, (Real, decimal.Decimal)):
        raise InvalidArgument("%s=%r must be a real number." % (name, value))
    if math.isnan(value):
        raise InvalidArgument(u"Invalid end point %s=%r" % (name, value))


@check_function
def check_valid_magnitude(value, name):
    """Checks that value is either unspecified, or a non-negative valid
    interval bound.

    Otherwise raises InvalidArgument.
    """
    check_valid_bound(value, name)
    if value is not None and value < 0:
        raise InvalidArgument("%s=%r must not be negative." % (name, value))


@check_function
def try_convert(typ, value, name):
    if value is None:
        return None
    if isinstance(value, typ):
        return value
    try:
        return typ(value)
    except (TypeError, OverflowError, ValueError, ArithmeticError):
        raise InvalidArgument(
            "Cannot convert %s=%r of type %s to type %s"
            % (name, value, type(value).__name__, typ.__name__)
        )


@check_function
def check_valid_size(value, name):
    """Checks that value is either unspecified, or a valid non-negative size
    expressed as an integer/float.

    Otherwise raises InvalidArgument.
    """
    if value is None:
        if name == "min_size":
            from hypothesis._settings import note_deprecation

            note_deprecation(
                "min_size=None is deprecated; use min_size=0 instead.",
                since="2018-10-06",
            )
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise InvalidArgument(u"Invalid size %s=%r" % (name, value))
        from hypothesis._settings import note_deprecation

        note_deprecation(
            "Float size are deprecated: "
            "%s should be an integer, got %r" % (name, value),
            since="2018-10-11",
        )
    else:
        check_type(integer_types, value, name)
    if value < 0:
        raise InvalidArgument(u"Invalid size %s=%r < 0" % (name, value))


@check_function
def check_valid_interval(lower_bound, upper_bound, lower_name, upper_name):
    """Checks that lower_bound and upper_bound are either unspecified, or they
    define a valid interval on the number line.

    Otherwise raises InvalidArgument.
    """
    if lower_bound is None or upper_bound is None:
        return
    if upper_bound < lower_bound:
        raise InvalidArgument(
            "Cannot have %s=%r < %s=%r"
            % (upper_name, upper_bound, lower_name, lower_bound)
        )


@check_function
def check_valid_sizes(min_size, max_size):
    check_valid_size(min_size, "min_size")
    check_valid_size(max_size, "max_size")
    check_valid_interval(min_size, max_size, "min_size", "max_size")
