"""
This conftest SHOULD be counted an initial conftest, see
https://docs.pytest.org/en/stable/how-to/writing_plugins.html#pluginorder

This is because this conftest is located in the directory of the test module.
"""

import pytest


def pytest_configure(config):
    config._my_conftests = getattr(config, '_my_conftests', []) + ['conftest_non_local/tests/conftest.py']


def pytest_sessionstart(session):
    session._my_conftests = getattr(session, '_my_conftests', []) + ['conftest_non_local/tests/conftest.py']


@pytest.fixture
def fixture_order(fixture_order):
    return fixture_order + ['conftest_non_local/tests/conftest.py']
