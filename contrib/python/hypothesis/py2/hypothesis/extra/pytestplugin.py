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

from distutils.version import LooseVersion

import pytest

from hypothesis import Verbosity, core, settings
from hypothesis._settings import note_deprecation
from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import text_type
from hypothesis.internal.detection import is_hypothesis_test
from hypothesis.reporting import default as default_reporter, with_reporter
from hypothesis.statistics import collector

LOAD_PROFILE_OPTION = "--hypothesis-profile"
VERBOSITY_OPTION = "--hypothesis-verbosity"
PRINT_STATISTICS_OPTION = "--hypothesis-show-statistics"
SEED_OPTION = "--hypothesis-seed"


class StoringReporter(object):
    def __init__(self, config):
        self.config = config
        self.results = []

    def __call__(self, msg):
        if self.config.getoption("capture", "fd") == "no":
            default_reporter(msg)
        if not isinstance(msg, text_type):
            msg = repr(msg)
        self.results.append(msg)


if LooseVersion(pytest.__version__) < "4.3":  # pragma: no cover
    import warnings
    from hypothesis.errors import HypothesisWarning

    PYTEST_TOO_OLD_MESSAGE = """
        You are using Pytest version %s.  Hypothesis tests work with any test
        runner, but our Pytest plugin requires Pytest  4.3 or newer.
        Note that the Pytest developers no longer support this version either!
        Disabling the Hypothesis pytest plugin...
    """
    warnings.warn(PYTEST_TOO_OLD_MESSAGE % (pytest.__version__,), HypothesisWarning)

else:

    def pytest_addoption(parser):
        group = parser.getgroup("hypothesis", "Hypothesis")
        group.addoption(
            LOAD_PROFILE_OPTION,
            action="store",
            help="Load in a registered hypothesis.settings profile",
        )
        group.addoption(
            VERBOSITY_OPTION,
            action="store",
            choices=[opt.name for opt in Verbosity],
            help="Override profile with verbosity setting specified",
        )
        group.addoption(
            PRINT_STATISTICS_OPTION,
            action="store_true",
            help="Configure when statistics are printed",
            default=False,
        )
        group.addoption(
            SEED_OPTION,
            action="store",
            help="Set a seed to use for all Hypothesis tests",
        )

    def pytest_report_header(config):
        profile = config.getoption(LOAD_PROFILE_OPTION)
        if not profile:
            profile = settings._current_profile
        settings_str = settings.get_profile(profile).show_changed()
        if settings_str != "":
            settings_str = " -> %s" % (settings_str)
        if (
            config.option.verbose >= 1
            or settings.default.verbosity >= Verbosity.verbose
        ):
            return "hypothesis profile %r%s" % (profile, settings_str)

    def pytest_configure(config):
        core.running_under_pytest = True
        profile = config.getoption(LOAD_PROFILE_OPTION)
        if profile:
            settings.load_profile(profile)
        verbosity_name = config.getoption(VERBOSITY_OPTION)
        if verbosity_name:
            verbosity_value = Verbosity[verbosity_name]
            profile_name = "%s-with-%s-verbosity" % (
                settings._current_profile,
                verbosity_name,
            )
            # register_profile creates a new profile, exactly like the current one,
            # with the extra values given (in this case 'verbosity')
            settings.register_profile(profile_name, verbosity=verbosity_value)
            settings.load_profile(profile_name)
        seed = config.getoption(SEED_OPTION)
        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                pass
            core.global_force_seed = seed
        config.addinivalue_line("markers", "hypothesis: Tests which use hypothesis.")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item):
        if not hasattr(item, "obj"):
            yield
        elif not is_hypothesis_test(item.obj):
            # If @given was not applied, check whether other hypothesis
            # decorators were applied, and raise an error if they were.
            message = "Using `@%s` on a test without `@given` is completely pointless."
            if getattr(item.obj, "_hypothesis_internal_settings_applied", False):
                raise InvalidArgument(message % ("settings",))
            if getattr(item.obj, "is_hypothesis_strategy_function", False):
                note_deprecation(
                    "%s is a function that returns a Hypothesis strategy, but pytest "
                    "has collected it as a test function.  This is useless as the "
                    "function body will never be executed.  To define a test "
                    "function, use @given instead of @composite." % (item.nodeid,),
                    since="2018-11-02",
                )
            for name, attribute in [
                ("example", "hypothesis_explicit_examples"),
                ("seed", "_hypothesis_internal_use_seed"),
                ("reproduce_example", "_hypothesis_internal_use_reproduce_failure"),
            ]:
                if hasattr(item.obj, attribute):
                    note_deprecation(message % (name,), since="2019-12-07")
            yield
        else:
            if item.get_closest_marker("parametrize") is not None:
                # Give every parametrized test invocation a unique database key
                key = item.nodeid.encode("utf-8")
                item.obj.hypothesis.inner_test._hypothesis_internal_add_digest = key

            store = StoringReporter(item.config)

            def note_statistics(stats):
                lines = [item.nodeid + ":", ""] + stats.get_description() + [""]
                item.hypothesis_statistics = lines

            with collector.with_value(note_statistics):
                with with_reporter(store):
                    yield
            if store.results:
                item.hypothesis_report_information = list(store.results)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):
        report = (yield).get_result()
        if hasattr(item, "hypothesis_report_information"):
            report.sections.append(
                ("Hypothesis", "\n".join(item.hypothesis_report_information))
            )
        if hasattr(item, "hypothesis_statistics") and report.when == "teardown":
            val = ("hypothesis-stats", item.hypothesis_statistics)
            report.user_properties.append(val)

    def pytest_terminal_summary(terminalreporter):
        if not terminalreporter.config.getoption(PRINT_STATISTICS_OPTION):
            return
        terminalreporter.section("Hypothesis Statistics")
        # terminalreporter.stats is a dict, where the empty string appears to
        # always be the key for a list of _pytest.reports.TestReport objects
        # (where we stored the statistics data in pytest_runtest_makereport above)
        for test_report in terminalreporter.stats.get("", []):
            for name, lines in test_report.user_properties:
                if name == "hypothesis-stats" and test_report.when == "teardown":
                    for li in lines:
                        terminalreporter.write_line(li)

    def pytest_collection_modifyitems(items):
        for item in items:
            if isinstance(item, pytest.Function) and is_hypothesis_test(item.obj):
                item.add_marker("hypothesis")


def load():
    """Required for `pluggy` to load a plugin from setuptools entrypoints."""
