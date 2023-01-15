import gc
import platform

import pytest

from markupsafe import escape


@pytest.mark.skipif(
    escape.__module__ == "markupsafe._native",
    reason="only test memory leak with speedups",
)
def test_markup_leaks():
    counts = set()

    for _i in range(20):
        for _j in range(1000):
            escape("foo")
            escape("<foo>")
            escape("foo")
            escape("<foo>")

        if platform.python_implementation() == "PyPy":
            gc.collect()

        counts.add(len(gc.get_objects()))

    assert len(counts) == 1
