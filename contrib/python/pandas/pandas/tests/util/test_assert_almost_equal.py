# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pandas import DataFrame, Index, Series, Timestamp
from pandas.util.testing import assert_almost_equal


def _assert_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    kwargs : dict
        The arguments passed to `assert_almost_equal`.
    """
    assert_almost_equal(a, b, **kwargs)
    assert_almost_equal(b, a, **kwargs)


def _assert_not_almost_equal(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    kwargs : dict
        The arguments passed to `assert_almost_equal`.
    """
    try:
        assert_almost_equal(a, b, **kwargs)
        msg = ("{a} and {b} were approximately equal "
               "when they shouldn't have been").format(a=a, b=b)
        pytest.fail(msg=msg)
    except AssertionError:
        pass


def _assert_not_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    kwargs : dict
        The arguments passed to `tm.assert_almost_equal`.
    """
    _assert_not_almost_equal(a, b, **kwargs)
    _assert_not_almost_equal(b, a, **kwargs)


@pytest.mark.parametrize("a,b", [
    (1.1, 1.1), (1.1, 1.100001), (np.int16(1), 1.000001),
    (np.float64(1.1), 1.1), (np.uint32(5), 5),
])
def test_assert_almost_equal_numbers(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [
    (1.1, 1), (1.1, True), (1, 2), (1.0001, np.int16(1)),
])
def test_assert_not_almost_equal_numbers(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [
    (0, 0), (0, 0.0), (0, np.float64(0)), (0.000001, 0),
])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [
    (0.001, 0), (1, 0),
])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [
    (1, "abc"), (1, [1, ]), (1, object()),
])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "left_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"])
@pytest.mark.parametrize(
    "right_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"])
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    # Empty compare.
    _assert_almost_equal_both(np.array([], dtype=left_dtype),
                              np.array([], dtype=right_dtype),
                              check_dtype=False)


def test_assert_almost_equal_dicts():
    _assert_almost_equal_both({"a": 1, "b": 2}, {"a": 1, "b": 2})


@pytest.mark.parametrize("a,b", [
    ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
    ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}),
    ({"a": 1}, 1), ({"a": 1}, "abc"), ({"a": 1}, [1, ]),
])
def test_assert_not_almost_equal_dicts(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("val", [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    dict_val = 1
    real_dict = dict(a=val)

    class DictLikeObj(object):
        def keys(self):
            return "a",

        def __getitem__(self, item):
            if item == "a":
                return dict_val

    func = (_assert_almost_equal_both if val == dict_val
            else _assert_not_almost_equal_both)
    func(real_dict, DictLikeObj(), check_dtype=False)


def test_assert_almost_equal_strings():
    _assert_almost_equal_both("abc", "abc")


@pytest.mark.parametrize("a,b", [
    ("abc", "abcd"), ("abc", "abd"), ("abc", 1), ("abc", [1, ]),
])
def test_assert_not_almost_equal_strings(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [
    ([1, 2, 3], [1, 2, 3]), (np.array([1, 2, 3]), np.array([1, 2, 3])),
])
def test_assert_almost_equal_iterables(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [
    # Class is different.
    (np.array([1, 2, 3]), [1, 2, 3]),

    # Dtype is different.
    (np.array([1, 2, 3]), np.array([1., 2., 3.])),

    # Can't compare generators.
    (iter([1, 2, 3]), [1, 2, 3]), ([1, 2, 3], [1, 2, 4]),
    ([1, 2, 3], [1, 2, 3, 4]), ([1, 2, 3], 1),
])
def test_assert_not_almost_equal_iterables(a, b):
    _assert_not_almost_equal(a, b)


def test_assert_almost_equal_null():
    _assert_almost_equal_both(None, None)


@pytest.mark.parametrize("a,b", [
    (None, np.NaN), (None, 0), (np.NaN, 0),
])
def test_assert_not_almost_equal_null(a, b):
    _assert_not_almost_equal(a, b)


@pytest.mark.parametrize("a,b", [
    (np.inf, np.inf), (np.inf, float("inf")),
    (np.array([np.inf, np.nan, -np.inf]),
     np.array([np.inf, np.nan, -np.inf])),
    (np.array([np.inf, None, -np.inf], dtype=np.object_),
     np.array([np.inf, np.nan, -np.inf], dtype=np.object_)),
])
def test_assert_almost_equal_inf(a, b):
    _assert_almost_equal_both(a, b)


def test_assert_not_almost_equal_inf():
    _assert_not_almost_equal_both(np.inf, 0)


@pytest.mark.parametrize("a,b", [
    (Index([1., 1.1]), Index([1., 1.100001])),
    (Series([1., 1.1]), Series([1., 1.100001])),
    (np.array([1.1, 2.000001]), np.array([1.1, 2.0])),
    (DataFrame({"a": [1., 1.1]}), DataFrame({"a": [1., 1.100001]}))
])
def test_assert_almost_equal_pandas(a, b):
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_object():
    a = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    b = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_value_mismatch():
    msg = "expected 2\\.00000 but got 1\\.00000, with decimal 5"

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(1, 2)


@pytest.mark.parametrize("a,b,klass1,klass2", [
    (np.array([1]), 1, "ndarray", "int"),
    (1, np.array([1]), "int", "ndarray"),
])
def test_assert_almost_equal_class_mismatch(a, b, klass1, klass2):
    msg = """numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}""".format(klass1=klass1, klass2=klass2)

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(a, b)


def test_assert_almost_equal_value_mismatch1():
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(np.array([np.nan, 2, 3]),
                            np.array([1, np.nan, 3]))


def test_assert_almost_equal_value_mismatch2():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_almost_equal_value_mismatch3():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(np.array([[1, 2], [3, 4], [5, 6]]),
                            np.array([[1, 3], [3, 4], [5, 6]]))


def test_assert_almost_equal_value_mismatch4():
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(np.array([[1, 2], [3, 4]]),
                            np.array([[1, 3], [3, 4]]))


def test_assert_almost_equal_shape_mismatch_override():
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""
    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(np.array([1, 2]),
                            np.array([3, 4, 5]),
                            obj="Index")


def test_assert_almost_equal_unicode():
    # see gh-20503
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(np.array([u"á", u"à", u"ä"]),
                            np.array([u"á", u"à", u"å"]))


def test_assert_almost_equal_timestamp():
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal(a, b)


def test_assert_almost_equal_iterable_length_mismatch():
    msg = """Iterable are different

Iterable length are different
\\[left\\]:  2
\\[right\\]: 3"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal([1, 2], [3, 4, 5])


def test_assert_almost_equal_iterable_values_mismatch():
    msg = """Iterable are different

Iterable values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        assert_almost_equal([1, 2], [1, 3])
