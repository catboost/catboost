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

import os
import sys
import traceback
from inspect import getframeinfo

import hypothesis
from hypothesis.errors import (
    DeadlineExceeded,
    HypothesisException,
    MultipleFailures,
    StopTest,
    UnsatisfiedAssumption,
)
from hypothesis.internal.compat import binary_type, encoded_filepath, text_type

if False:
    from typing import Dict  # noqa


def belongs_to(package):
    if not hasattr(package, "__file__"):  # pragma: no cover
        return lambda filepath: False

    root = os.path.dirname(package.__file__)
    cache = {text_type: {}, binary_type: {}}

    def accept(filepath):
        ftype = type(filepath)
        try:
            return cache[ftype][filepath]
        except KeyError:
            pass
        new_filepath = encoded_filepath(filepath)
        result = os.path.abspath(new_filepath).startswith(root)
        cache[ftype][filepath] = result
        cache[type(new_filepath)][new_filepath] = result
        return result

    accept.__name__ = "is_%s_file" % (package.__name__,)
    return accept


PREVENT_ESCALATION = os.getenv("HYPOTHESIS_DO_NOT_ESCALATE") == "true"

FILE_CACHE = {}  # type: Dict[bytes, bool]


is_hypothesis_file = belongs_to(hypothesis)

HYPOTHESIS_CONTROL_EXCEPTIONS = (DeadlineExceeded, StopTest, UnsatisfiedAssumption)


def mark_for_escalation(e):
    if not isinstance(e, HYPOTHESIS_CONTROL_EXCEPTIONS):
        e.hypothesis_internal_always_escalate = True


def escalate_hypothesis_internal_error():
    if PREVENT_ESCALATION:
        return

    error_type, e, tb = sys.exc_info()

    if getattr(e, "hypothesis_internal_never_escalate", False):
        return

    if getattr(e, "hypothesis_internal_always_escalate", False):
        raise
    filepath = traceback.extract_tb(tb)[-1][0]
    if is_hypothesis_file(filepath) and not isinstance(
        e, (HypothesisException,) + HYPOTHESIS_CONTROL_EXCEPTIONS
    ):
        raise


def get_trimmed_traceback():
    """Return the current traceback, minus any frames added by Hypothesis."""
    error_type, _, tb = sys.exc_info()
    # Avoid trimming the traceback if we're in verbose mode, or the error
    # was raised inside Hypothesis (and is not a MultipleFailures)
    if hypothesis.settings.default.verbosity >= hypothesis.Verbosity.debug or (
        is_hypothesis_file(traceback.extract_tb(tb)[-1][0])
        and not isinstance(error_type, MultipleFailures)
    ):
        return tb
    while tb is not None and (
        # If the frame is from one of our files, it's been added by Hypothesis.
        is_hypothesis_file(getframeinfo(tb.tb_frame)[0])
        # But our `@proxies` decorator overrides the source location,
        # so we check for an attribute it injects into the frame too.
        or tb.tb_frame.f_globals.get("__hypothesistracebackhide__") is True
    ):
        tb = tb.tb_next
    return tb
