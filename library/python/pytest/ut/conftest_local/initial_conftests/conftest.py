"""
This conftest SHOULD be counted an initial conftest, see
https://docs.pytest.org/en/stable/how-to/writing_plugins.html#pluginorder

This is because this conftest is located in a parent directory of the test module.
"""

import pytest


sessionstart_called = False


def pytest_sessionstart(session):
    global sessionstart_called
    sessionstart_called = True


@pytest.fixture
def is_sessionstart_called_root():
    return sessionstart_called
