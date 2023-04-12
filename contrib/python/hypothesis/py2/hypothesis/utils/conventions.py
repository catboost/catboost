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

# Notes: we use instances of these objects as singletons which serve as
# identifiers in various patches of code.  The more specific types
# (DefaultValueType and InferType) exist so that typecheckers such as Mypy
# can distinguish them from the others.  DefaultValueType is only used in
# the Django extra.


class UniqueIdentifier(object):
    def __init__(self, identifier):
        self.identifier = identifier

    def __repr__(self):
        return self.identifier


class DefaultValueType(UniqueIdentifier):
    pass


class InferType(UniqueIdentifier):
    pass


infer = InferType(u"infer")
not_set = UniqueIdentifier(u"not_set")
