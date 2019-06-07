# -*- coding: utf-8 -*-
from datetime import datetime, timedelta

import numpy as np
import pytest

from pandas.compat import long

from pandas import Timedelta, Timestamp
import pandas.util.testing as tm

from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TestTimestampArithmetic(object):
    def test_overflow_offset(self):
        # no overflow expected

        stamp = Timestamp("2000/1/1")
        offset_no_overflow = to_offset("D") * 100

        expected = Timestamp("2000/04/10")
        assert stamp + offset_no_overflow == expected

        assert offset_no_overflow + stamp == expected

        expected = Timestamp("1999/09/23")
        assert stamp - offset_no_overflow == expected

    def test_overflow_offset_raises(self):
        # xref https://github.com/statsmodels/statsmodels/issues/3374
        # ends up multiplying really large numbers which overflow

        stamp = Timestamp('2017-01-13 00:00:00', freq='D')
        offset_overflow = 20169940 * offsets.Day(1)
        msg = ("the add operation between "
               r"\<-?\d+ \* Days\> and \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
               "will overflow")

        with pytest.raises(OverflowError, match=msg):
            stamp + offset_overflow

        with pytest.raises(OverflowError, match=msg):
            offset_overflow + stamp

        with pytest.raises(OverflowError, match=msg):
            stamp - offset_overflow

        # xref https://github.com/pandas-dev/pandas/issues/14080
        # used to crash, so check for proper overflow exception

        stamp = Timestamp("2000/1/1")
        offset_overflow = to_offset("D") * 100 ** 25

        with pytest.raises(OverflowError, match=msg):
            stamp + offset_overflow

        with pytest.raises(OverflowError, match=msg):
            offset_overflow + stamp

        with pytest.raises(OverflowError, match=msg):
            stamp - offset_overflow

    def test_delta_preserve_nanos(self):
        val = Timestamp(long(1337299200000000123))
        result = val + timedelta(1)
        assert result.nanosecond == val.nanosecond

    def test_timestamp_sub_datetime(self):
        dt = datetime(2013, 10, 12)
        ts = Timestamp(datetime(2013, 10, 13))
        assert (ts - dt).days == 1
        assert (dt - ts).days == -1

    def test_addition_subtraction_types(self):
        # Assert on the types resulting from Timestamp +/- various date/time
        # objects
        dt = datetime(2014, 3, 4)
        td = timedelta(seconds=1)
        # build a timestamp with a frequency, since then it supports
        # addition/subtraction of integers
        ts = Timestamp(dt, freq='D')

        with tm.assert_produces_warning(FutureWarning):
            # GH#22535 add/sub with integers is deprecated
            assert type(ts + 1) == Timestamp
            assert type(ts - 1) == Timestamp

        # Timestamp + datetime not supported, though subtraction is supported
        # and yields timedelta more tests in tseries/base/tests/test_base.py
        assert type(ts - dt) == Timedelta
        assert type(ts + td) == Timestamp
        assert type(ts - td) == Timestamp

        # Timestamp +/- datetime64 not supported, so not tested (could possibly
        # assert error raised?)
        td64 = np.timedelta64(1, 'D')
        assert type(ts + td64) == Timestamp
        assert type(ts - td64) == Timestamp

    def test_addition_subtraction_preserve_frequency(self):
        ts = Timestamp('2014-03-05', freq='D')
        td = timedelta(days=1)
        original_freq = ts.freq

        with tm.assert_produces_warning(FutureWarning):
            # GH#22535 add/sub with integers is deprecated
            assert (ts + 1).freq == original_freq
            assert (ts - 1).freq == original_freq

        assert (ts + td).freq == original_freq
        assert (ts - td).freq == original_freq

        td64 = np.timedelta64(1, 'D')
        assert (ts + td64).freq == original_freq
        assert (ts - td64).freq == original_freq
