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

"""
-----------------------
hypothesis[dpcontracts]
-----------------------

This module provides tools for working with the :pypi:`dpcontracts` library,
because `combining contracts and property-based testing works really well
<https://hillelwayne.com/talks/beyond-unit-tests/>`_.

It requires ``dpcontracts >= 0.4``.
"""

from __future__ import absolute_import, division, print_function

from dpcontracts import PreconditionError

from hypothesis import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import proxies


def fulfill(contract_func):
    """Decorate ``contract_func`` to reject calls which violate preconditions,
    and retry them with different arguments.

    This is a convenience function for testing internal code that uses
    :pypi:`dpcontracts`, to automatically filter out arguments that would be
    rejected by the public interface before triggering a contract error.

    This can be used as ``builds(fulfill(func), ...)`` or in the body of the
    test e.g. ``assert fulfill(func)(*args)``.
    """
    if not hasattr(contract_func, "__contract_wrapped_func__"):
        raise InvalidArgument(
            "There are no dpcontracts preconditions associated with %s"
            % (contract_func.__name__,)
        )

    @proxies(contract_func)
    def inner(*args, **kwargs):
        try:
            return contract_func(*args, **kwargs)
        except PreconditionError:
            reject()

    return inner
