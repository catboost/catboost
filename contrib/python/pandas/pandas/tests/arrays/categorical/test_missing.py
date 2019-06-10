# -*- coding: utf-8 -*-
import collections

import numpy as np
import pytest

from pandas.compat import lrange

from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas import Categorical, Index, isna
import pandas.util.testing as tm


class TestCategoricalMissing(object):

    def test_na_flags_int_categories(self):
        # #1457

        categories = lrange(10)
        labels = np.random.randint(0, 10, 20)
        labels[::5] = -1

        cat = Categorical(labels, categories, fastpath=True)
        repr(cat)

        tm.assert_numpy_array_equal(isna(cat), labels == -1)

    def test_nan_handling(self):

        # Nans are represented as -1 in codes
        c = Categorical(["a", "b", np.nan, "a"])
        tm.assert_index_equal(c.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0],
                                                       dtype=np.int8))
        c[1] = np.nan
        tm.assert_index_equal(c.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(c._codes, np.array([0, -1, -1, 0],
                                                       dtype=np.int8))

        # Adding nan to categories should make assigned nan point to the
        # category!
        c = Categorical(["a", "b", np.nan, "a"])
        tm.assert_index_equal(c.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0],
                                                       dtype=np.int8))

    def test_set_dtype_nans(self):
        c = Categorical(['a', 'b', np.nan])
        result = c._set_dtype(CategoricalDtype(['a', 'c']))
        tm.assert_numpy_array_equal(result.codes, np.array([0, -1, -1],
                                                           dtype='int8'))

    def test_set_item_nan(self):
        cat = Categorical([1, 2, 3])
        cat[1] = np.nan

        exp = Categorical([1, np.nan, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(cat, exp)

    @pytest.mark.parametrize('fillna_kwargs, msg', [
        (dict(value=1, method='ffill'),
         "Cannot specify both 'value' and 'method'."),
        (dict(),
         "Must specify a fill 'value' or 'method'."),
        (dict(method='bad'),
         "Invalid fill method. Expecting .* bad"),
    ])
    def test_fillna_raises(self, fillna_kwargs, msg):
        # https://github.com/pandas-dev/pandas/issues/19682
        cat = Categorical([1, 2, 3])

        with pytest.raises(ValueError, match=msg):
            cat.fillna(**fillna_kwargs)

    @pytest.mark.parametrize("named", [True, False])
    def test_fillna_iterable_category(self, named):
        # https://github.com/pandas-dev/pandas/issues/21097
        if named:
            Point = collections.namedtuple("Point", "x y")
        else:
            Point = lambda *args: args  # tuple
        cat = Categorical([Point(0, 0), Point(0, 1), None])
        result = cat.fillna(Point(0, 0))
        expected = Categorical([Point(0, 0), Point(0, 1), Point(0, 0)])

        tm.assert_categorical_equal(result, expected)
