from __future__ import division

import numpy as np
import pytest

from pandas import Interval, Timedelta, Timestamp
import pandas.core.common as com


@pytest.fixture
def interval():
    return Interval(0, 1)


class TestInterval(object):

    def test_properties(self, interval):
        assert interval.closed == 'right'
        assert interval.left == 0
        assert interval.right == 1
        assert interval.mid == 0.5

    def test_repr(self, interval):
        assert repr(interval) == "Interval(0, 1, closed='right')"
        assert str(interval) == "(0, 1]"

        interval_left = Interval(0, 1, closed='left')
        assert repr(interval_left) == "Interval(0, 1, closed='left')"
        assert str(interval_left) == "[0, 1)"

    def test_contains(self, interval):
        assert 0.5 in interval
        assert 1 in interval
        assert 0 not in interval

        msg = "__contains__ not defined for two intervals"
        with pytest.raises(TypeError, match=msg):
            interval in interval

        interval_both = Interval(0, 1, closed='both')
        assert 0 in interval_both
        assert 1 in interval_both

        interval_neither = Interval(0, 1, closed='neither')
        assert 0 not in interval_neither
        assert 0.5 in interval_neither
        assert 1 not in interval_neither

    def test_equal(self):
        assert Interval(0, 1) == Interval(0, 1, closed='right')
        assert Interval(0, 1) != Interval(0, 1, closed='left')
        assert Interval(0, 1) != 0

    def test_comparison(self):
        with pytest.raises(TypeError, match='unorderable types'):
            Interval(0, 1) < 2

        assert Interval(0, 1) < Interval(1, 2)
        assert Interval(0, 1) < Interval(0, 2)
        assert Interval(0, 1) < Interval(0.5, 1.5)
        assert Interval(0, 1) <= Interval(0, 1)
        assert Interval(0, 1) > Interval(-1, 2)
        assert Interval(0, 1) >= Interval(0, 1)

    def test_hash(self, interval):
        # should not raise
        hash(interval)

    @pytest.mark.parametrize('left, right, expected', [
        (0, 5, 5),
        (-2, 5.5, 7.5),
        (10, 10, 0),
        (10, np.inf, np.inf),
        (-np.inf, -5, np.inf),
        (-np.inf, np.inf, np.inf),
        (Timedelta('0 days'), Timedelta('5 days'), Timedelta('5 days')),
        (Timedelta('10 days'), Timedelta('10 days'), Timedelta('0 days')),
        (Timedelta('1H10M'), Timedelta('5H5M'), Timedelta('3H55M')),
        (Timedelta('5S'), Timedelta('1H'), Timedelta('59M55S'))])
    def test_length(self, left, right, expected):
        # GH 18789
        iv = Interval(left, right)
        result = iv.length
        assert result == expected

    @pytest.mark.parametrize('left, right, expected', [
        ('2017-01-01', '2017-01-06', '5 days'),
        ('2017-01-01', '2017-01-01 12:00:00', '12 hours'),
        ('2017-01-01 12:00', '2017-01-01 12:00:00', '0 days'),
        ('2017-01-01 12:01', '2017-01-05 17:31:00', '4 days 5 hours 30 min')])
    @pytest.mark.parametrize('tz', (None, 'UTC', 'CET', 'US/Eastern'))
    def test_length_timestamp(self, tz, left, right, expected):
        # GH 18789
        iv = Interval(Timestamp(left, tz=tz), Timestamp(right, tz=tz))
        result = iv.length
        expected = Timedelta(expected)
        assert result == expected

    @pytest.mark.parametrize('left, right', [
        ('a', 'z'),
        (('a', 'b'), ('c', 'd')),
        (list('AB'), list('ab')),
        (Interval(0, 1), Interval(1, 2))])
    def test_length_errors(self, left, right):
        # GH 18789
        iv = Interval(left, right)
        msg = 'cannot compute length between .* and .*'
        with pytest.raises(TypeError, match=msg):
            iv.length

    def test_math_add(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(1, 2, closed=closed)

        result = interval + 1
        assert result == expected

        result = 1 + interval
        assert result == expected

        result = interval
        result += 1
        assert result == expected

        msg = r"unsupported operand type\(s\) for \+"
        with pytest.raises(TypeError, match=msg):
            interval + interval

        with pytest.raises(TypeError, match=msg):
            interval + 'foo'

    def test_math_sub(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(-1, 0, closed=closed)

        result = interval - 1
        assert result == expected

        result = interval
        result -= 1
        assert result == expected

        msg = r"unsupported operand type\(s\) for -"
        with pytest.raises(TypeError, match=msg):
            interval - interval

        with pytest.raises(TypeError, match=msg):
            interval - 'foo'

    def test_math_mult(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 2, closed=closed)

        result = interval * 2
        assert result == expected

        result = 2 * interval
        assert result == expected

        result = interval
        result *= 2
        assert result == expected

        msg = r"unsupported operand type\(s\) for \*"
        with pytest.raises(TypeError, match=msg):
            interval * interval

        msg = r"can\'t multiply sequence by non-int"
        with pytest.raises(TypeError, match=msg):
            interval * 'foo'

    def test_math_div(self, closed):
        interval = Interval(0, 1, closed=closed)
        expected = Interval(0, 0.5, closed=closed)

        result = interval / 2.0
        assert result == expected

        result = interval
        result /= 2.0
        assert result == expected

        msg = r"unsupported operand type\(s\) for /"
        with pytest.raises(TypeError, match=msg):
            interval / interval

        with pytest.raises(TypeError, match=msg):
            interval / 'foo'

    def test_math_floordiv(self, closed):
        interval = Interval(1, 2, closed=closed)
        expected = Interval(0, 1, closed=closed)

        result = interval // 2
        assert result == expected

        result = interval
        result //= 2
        assert result == expected

        msg = r"unsupported operand type\(s\) for //"
        with pytest.raises(TypeError, match=msg):
            interval // interval

        with pytest.raises(TypeError, match=msg):
            interval // 'foo'

    def test_constructor_errors(self):
        msg = "invalid option for 'closed': foo"
        with pytest.raises(ValueError, match=msg):
            Interval(0, 1, closed='foo')

        msg = 'left side of interval must be <= right side'
        with pytest.raises(ValueError, match=msg):
            Interval(1, 0)

    @pytest.mark.parametrize('tz_left, tz_right', [
        (None, 'UTC'), ('UTC', None), ('UTC', 'US/Eastern')])
    def test_constructor_errors_tz(self, tz_left, tz_right):
        # GH 18538
        left = Timestamp('2017-01-01', tz=tz_left)
        right = Timestamp('2017-01-02', tz=tz_right)
        error = TypeError if com._any_none(tz_left, tz_right) else ValueError
        with pytest.raises(error):
            Interval(left, right)
