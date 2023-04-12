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

import inspect

from hypothesis._settings import Verbosity, settings
from hypothesis.internal.compat import (
    binary_type,
    escape_unicode_characters,
    print_unicode,
)
from hypothesis.utils.dynamicvariables import DynamicVariable


def silent(value):
    pass


def default(value):
    try:
        print_unicode(value)
    except UnicodeEncodeError:
        print_unicode(escape_unicode_characters(value))


reporter = DynamicVariable(default)


def current_reporter():
    return reporter.value


def with_reporter(new_reporter):
    return reporter.with_value(new_reporter)


def current_verbosity():
    return settings.default.verbosity


def to_text(textish):
    if inspect.isfunction(textish):
        textish = textish()
    if isinstance(textish, binary_type):
        textish = textish.decode("utf-8")
    return textish


def verbose_report(text):
    if current_verbosity() >= Verbosity.verbose:
        base_report(text)


def debug_report(text):
    if current_verbosity() >= Verbosity.debug:
        base_report(text)


def report(text):
    if current_verbosity() >= Verbosity.normal:
        base_report(text)


def base_report(text):
    current_reporter()(to_text(text))
