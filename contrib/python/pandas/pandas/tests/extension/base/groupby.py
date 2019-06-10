import pytest

import pandas as pd
import pandas.util.testing as tm

from .base import BaseExtensionTests


class BaseGroupbyTests(BaseExtensionTests):
    """Groupby-specific tests."""

    def test_grouping_grouper(self, data_for_grouping):
        df = pd.DataFrame({
            "A": ["B", "B", None, None, "A", "A", "B", "C"],
            "B": data_for_grouping
        })
        gr1 = df.groupby("A").grouper.groupings[0]
        gr2 = df.groupby("B").grouper.groupings[0]

        tm.assert_numpy_array_equal(gr1.grouper, df.A.values)
        tm.assert_extension_array_equal(gr2.grouper, data_for_grouping)

    @pytest.mark.parametrize('as_index', [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4],
                           "B": data_for_grouping})
        result = df.groupby("B", as_index=as_index).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=True)

        index = pd.Index(index, name="B")
        expected = pd.Series([3, 1, 4], index=index, name="A")
        if as_index:
            self.assert_series_equal(result, expected)
        else:
            expected = expected.reset_index()
            self.assert_frame_equal(result, expected)

    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4],
                           "B": data_for_grouping})
        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index(index, name="B")
        expected = pd.Series([1, 3, 4], index=index, name="A")
        self.assert_series_equal(result, expected)

    def test_groupby_extension_transform(self, data_for_grouping):
        valid = data_for_grouping[~data_for_grouping.isna()]
        df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4],
                           "B": valid})

        result = df.groupby("B").A.transform(len)
        expected = pd.Series([3, 3, 2, 2, 3, 1], name="A")

        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize('op', [
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ], ids=['scalar', 'list', 'series', 'object'])
    def test_groupby_extension_apply(self, data_for_grouping, op):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4],
                           "B": data_for_grouping})
        df.groupby("B").apply(op)
        df.groupby("B").A.apply(op)
        df.groupby("A").apply(op)
        df.groupby("A").B.apply(op)

    def test_in_numeric_groupby(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4],
                           "B": data_for_grouping,
                           "C": [1, 1, 1, 1, 1, 1, 1, 1]})
        result = df.groupby("A").sum().columns

        if data_for_grouping.dtype._is_numeric:
            expected = pd.Index(['B', 'C'])
        else:
            expected = pd.Index(['C'])

        tm.assert_index_equal(result, expected)
