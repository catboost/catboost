"""
Tests ``from __future__ import absolute_import`` (only important for
Python 2.X)
"""
from parso import parse


def test_explicit_absolute_imports():
    """
    Detect modules with ``from __future__ import absolute_import``.
    """
    module = parse("from __future__ import absolute_import")
    assert module._has_explicit_absolute_import()


def test_no_explicit_absolute_imports():
    """
     Detect modules without ``from __future__ import absolute_import``.
    """
    assert not parse("1")._has_explicit_absolute_import()


def test_dont_break_imports_without_namespaces():
    """
    The code checking for ``from __future__ import absolute_import`` shouldn't
    assume that all imports have non-``None`` namespaces.
    """
    src = "from __future__ import absolute_import\nimport xyzzy"
    assert parse(src)._has_explicit_absolute_import()
