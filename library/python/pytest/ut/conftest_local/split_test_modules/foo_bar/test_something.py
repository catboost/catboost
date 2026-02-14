"""Test file to reproduce pytest conftest bug.

This test should only load foo_bar/conftest.py, but pytest incorrectly
also loads foo/conftest.py because 'foo_bar' starts with 'foo'.
"""

import pytest


def test_fixture_availability(request):
    """Test which fixtures are available - should only have foo_bar_fixture."""
    # This should work - foo_bar/conftest.py should be loaded
    foo_bar_value = request.getfixturevalue('foo_bar_fixture')
    assert foo_bar_value == 'foo_bar_fixture_value'

    # This should fail - foo/conftest.py should NOT be loaded
    with pytest.raises(LookupError):
        request.getfixturevalue('foo_fixture')


def test_simple():
    """Simple test to verify basic functionality."""
    assert True
