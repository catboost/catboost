"""Conftest for foo directory - should NOT be loaded by foo_bar tests."""

import pytest


@pytest.fixture
def foo_fixture():
    """Fixture from foo/conftest.py that should NOT be available in foo_bar."""
    return 'foo_fixture_value'
