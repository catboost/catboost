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

import hypothesis.internal.conjecture.utils as cu
from hypothesis.strategies._internal.strategies import SearchStrategy

FEATURE_LABEL = cu.calc_label_from_name("feature flag")


class FeatureFlags(object):
    """Object that can be used to control a number of feature flags for a
    given test run.

    This enables an approach to data generation called swarm testing (
    see Groce, Alex, et al. "Swarm testing." Proceedings of the 2012
    International Symposium on Software Testing and Analysis. ACM, 2012), in
    which generation is biased by selectively turning some features off for
    each test case generated. When there are many interacting features this can
    find bugs that a pure generation strategy would otherwise have missed.

    FeatureFlags are designed to "shrink open", so that during shrinking they
    become less restrictive. This allows us to potentially shrink to smaller
    test cases that were forbidden during the generation phase because they
    required disabled features.
    """

    def __init__(self, data=None, enabled=(), disabled=()):
        self.__data = data
        self.__decisions = {}

        for f in enabled:
            self.__decisions[f] = 0

        for f in disabled:
            self.__decisions[f] = 255

        # In the original swarm testing paper they turn features on or off
        # uniformly at random. Instead we decide the probability with which to
        # enable features up front. This can allow for scenarios where all or
        # no features are enabled, which are vanishingly unlikely in the
        # original model.
        #
        # We implement this as a single 8-bit integer and enable features which
        # score >= that value. In particular when self.__baseline is 0, all
        # features will be enabled. This is so that we shrink in the direction
        # of more features being enabled.
        if self.__data is not None:
            self.__baseline = data.draw_bits(8)
        else:
            # If data is None we're in example mode so all that matters is the
            # enabled/disabled lists above. We set this up so that
            self.__baseline = 1

    def is_enabled(self, name):
        """Tests whether the feature named ``name`` should be enabled on this
        test run."""
        if self.__data is None or self.__data.frozen:
            # Feature set objects might hang around after data generation has
            # finished. If this happens then we just report all new features as
            # enabled, because that's our shrinking direction and they have no
            # impact on data generation if they weren't used while it was
            # running.
            try:
                return self.__is_value_enabled(self.__decisions[name])
            except KeyError:
                return True

        data = self.__data

        data.start_example(label=FEATURE_LABEL)
        if name in self.__decisions:
            # If we've already decided on this feature then we don't actually
            # need to draw anything, but we do write the same decision to the
            # input stream. This allows us to lazily decide whether a feature
            # is enabled, because it means that if we happen to delete the part
            # of the test case where we originally decided, the next point at
            # which we make this decision just makes the decision it previously
            # made.
            value = self.__decisions[name]
            data.draw_bits(8, forced=value)
        else:
            # If the baseline is 0 then everything is enabled so it doesn't
            # matter what we have here and we might as well make the shrinker's
            # life easier by forcing it to zero.
            if self.__baseline == 0:
                value = 0
                data.draw_bits(8, forced=0)
            else:
                value = data.draw_bits(8)
            self.__decisions[name] = value
        data.stop_example()
        return self.__is_value_enabled(value)

    def __is_value_enabled(self, value):
        """Check if a given value drawn for a feature counts as enabled. Note
        that low values are more likely to be enabled. This is again in aid of
        shrinking open. In particular a value of 255 is always enabled."""
        return (255 - value) >= self.__baseline

    def __repr__(self):
        enabled = []
        disabled = []
        for k, v in self.__decisions.items():
            if self.__is_value_enabled(v):
                enabled.append(k)
            else:
                disabled.append(k)
        return "FeatureFlags(enabled=%r, disabled=%r)" % (enabled, disabled)


class FeatureStrategy(SearchStrategy):
    def do_draw(self, data):
        return FeatureFlags(data)
