"""
This conftest should be counted a non-initial conftest, see
https://docs.pytest.org/en/stable/how-to/writing_plugins.html#pluginorder

Still, with CONFTEST_LOCAL_POLICY != LOCAL, we will load it as an initial conftest.
"""

import pytest


def pytest_configure(config):
    config._my_conftests = getattr(config, '_my_conftests', []) + ['conftest_non_local/tests/lib_child/conftest.py']


def pytest_sessionstart(session):
    session._my_conftests = getattr(session, '_my_conftests', []) + ['conftest_non_local/tests/lib_child/conftest.py']


@pytest.fixture
def fixture_order(fixture_order):
    return fixture_order + ['conftest_non_local/tests/lib_child/conftest.py']
