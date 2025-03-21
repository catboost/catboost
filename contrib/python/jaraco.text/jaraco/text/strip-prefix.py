import sys

from jaraco.text import Stripper


def strip_prefix():
    r"""
    Strip any common prefix from stdin.

    >>> import io, pytest
    >>> getfixture('monkeypatch').setattr('sys.stdin', io.StringIO('abcdef\nabc123'))
    >>> strip_prefix()
    def
    123
    """
    sys.stdout.writelines(Stripper.strip_prefix(sys.stdin).lines)
