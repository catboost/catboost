"""
This conftest should not be loaded according to vanilla pytest rules.
Still, with CONFTEST_LOCAL_POLICY != LOCAL, we will load it as an initial conftest.
"""

import pytest


def pytest_configure(config):
    config._my_conftests = getattr(config, '_my_conftests', []) + ['conftest_non_local/lib2/conftest.py']


def pytest_sessionstart(session):
    session._my_conftests = getattr(session, '_my_conftests', []) + ['conftest_non_local/lib2/conftest.py']


@pytest.fixture
def fixture_order(fixture_order):
    return fixture_order + ['conftest_non_local/lib2/conftest.py']
