"""
This conftest should NOT be counted an initial conftest, see
https://docs.pytest.org/en/stable/how-to/writing_plugins.html#pluginorder

This is because this conftest is located in a subdirectory of the test module.
"""

import pytest


sessionstart_called = False


def pytest_sessionstart(session):
    global sessionstart_called
    sessionstart_called = True


@pytest.fixture
def is_sessionstart_called_bar():
    return sessionstart_called
