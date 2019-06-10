""" generic datetimelike tests """
import numpy as np
import pytest

import pandas as pd
import pandas.util.testing as tm

from .common import Base


class DatetimeLike(Base):

    def test_argmax_axis_invalid(self):
        # GH#23081
        rng = self.create_index()
        with pytest.raises(ValueError):
            rng.argmax(axis=1)
        with pytest.raises(ValueError):
            rng.argmin(axis=2)
        with pytest.raises(ValueError):
            rng.min(axis=-2)
        with pytest.raises(ValueError):
            rng.max(axis=-3)

    def test_can_hold_identifiers(self):
        idx = self.create_index()
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_shift_identity(self):

        idx = self.create_index()
        tm.assert_index_equal(idx, idx.shift(0))

    def test_str(self):

        # test the string repr
        idx = self.create_index()
        idx.name = 'foo'
        assert not "length=%s" % len(idx) in str(idx)
        assert "'foo'" in str(idx)
        assert idx.__class__.__name__ in str(idx)

        if hasattr(idx, 'tz'):
            if idx.tz is not None:
                assert idx.tz in str(idx)
        if hasattr(idx, 'freq'):
            assert "freq='%s'" % idx.freqstr in str(idx)

    def test_view(self):
        i = self.create_index()

        i_view = i.view('i8')
        result = self._holder(i)
        tm.assert_index_equal(result, i)

        i_view = i.view(self._holder)
        result = self._holder(i)
        tm.assert_index_equal(result, i_view)

    def test_map_callable(self):
        expected = self.index + self.index.freq
        result = self.index.map(lambda x: x + x.freq)
        tm.assert_index_equal(result, expected)

        # map to NaT
        result = self.index.map(lambda x: pd.NaT if x == self.index[0] else x)
        expected = pd.Index([pd.NaT] + self.index[1:].tolist())
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: pd.Series(values, index)])
    def test_map_dictlike(self, mapper):
        expected = self.index + self.index.freq

        # don't compare the freqs
        if isinstance(expected, pd.DatetimeIndex):
            expected.freq = None

        result = self.index.map(mapper(expected, self.index))
        tm.assert_index_equal(result, expected)

        expected = pd.Index([pd.NaT] + self.index[1:].tolist())
        result = self.index.map(mapper(expected, self.index))
        tm.assert_index_equal(result, expected)

        # empty map; these map to np.nan because we cannot know
        # to re-infer things
        expected = pd.Index([np.nan] * len(self.index))
        result = self.index.map(mapper([], []))
        tm.assert_index_equal(result, expected)

    def test_asobject_deprecated(self):
        # GH18572
        d = self.create_index()
        with tm.assert_produces_warning(FutureWarning):
            i = d.asobject
        assert isinstance(i, pd.Index)
