"""Conftest for foo_bar directory - should be loaded by foo_bar tests."""

import pytest


@pytest.fixture
def foo_bar_fixture():
    """Fixture from foo_bar/conftest.py that should be available in foo_bar tests."""
    return 'foo_bar_fixture_value'
