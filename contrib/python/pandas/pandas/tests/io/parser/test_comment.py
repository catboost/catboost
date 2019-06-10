# -*- coding: utf-8 -*-

"""
Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py
"""

import numpy as np
import pytest

from pandas.compat import StringIO

from pandas import DataFrame
import pandas.util.testing as tm


@pytest.mark.parametrize("na_values", [None, ["NaN"]])
def test_comment(all_parsers, na_values):
    parser = all_parsers
    data = """A,B,C
1,2.,4.#hello world
5.,NaN,10.0
"""
    expected = DataFrame([[1., 2., 4.], [5., np.nan, 10.]],
                         columns=["A", "B", "C"])
    result = parser.read_csv(StringIO(data), comment="#",
                             na_values=na_values)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("read_kwargs", [
    dict(),
    dict(lineterminator="*"),
    dict(delim_whitespace=True),
])
def test_line_comment(all_parsers, read_kwargs):
    parser = all_parsers
    data = """# empty
A,B,C
1,2.,4.#hello world
#ignore this line
5.,NaN,10.0
"""
    if read_kwargs.get("delim_whitespace"):
        data = data.replace(",", " ")
    elif read_kwargs.get("lineterminator"):
        if parser.engine != "c":
            pytest.skip("Custom terminator not supported with Python engine")

        data = data.replace("\n", read_kwargs.get("lineterminator"))

    read_kwargs["comment"] = "#"
    result = parser.read_csv(StringIO(data), **read_kwargs)

    expected = DataFrame([[1., 2., 4.], [5., np.nan, 10.]],
                         columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows(all_parsers):
    parser = all_parsers
    data = """# empty
random line
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # This should ignore the first four lines (including comments).
    expected = DataFrame([[1., 2., 4.], [5., np.nan, 10.]],
                         columns=["A", "B", "C"])
    result = parser.read_csv(StringIO(data), comment="#", skiprows=4)
    tm.assert_frame_equal(result, expected)


def test_comment_header(all_parsers):
    parser = all_parsers
    data = """# empty
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Header should begin at the second non-comment line.
    expected = DataFrame([[1., 2., 4.], [5., np.nan, 10.]],
                         columns=["A", "B", "C"])
    result = parser.read_csv(StringIO(data), comment="#", header=1)
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows_header(all_parsers):
    parser = all_parsers
    data = """# empty
# second empty line
# third empty line
X,Y,Z
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Skiprows should skip the first 4 lines (including comments),
    # while header should start from the second non-commented line,
    # starting with line 5.
    expected = DataFrame([[1., 2., 4.], [5., np.nan, 10.]],
                         columns=["A", "B", "C"])
    result = parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("comment_char", ["#", "~", "&", "^", "*", "@"])
def test_custom_comment_char(all_parsers, comment_char):
    parser = all_parsers
    data = "a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo"
    result = parser.read_csv(StringIO(data.replace("#", comment_char)),
                             comment=comment_char)

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", ["infer", None])
def test_comment_first_line(all_parsers, header):
    # see gh-4623
    parser = all_parsers
    data = "# notes\na,b,c\n# more notes\n1,2,3"

    if header is None:
        expected = DataFrame({0: ["a", "1"], 1: ["b", "2"], 2: ["c", "3"]})
    else:
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    result = parser.read_csv(StringIO(data), comment="#", header=header)
    tm.assert_frame_equal(result, expected)
