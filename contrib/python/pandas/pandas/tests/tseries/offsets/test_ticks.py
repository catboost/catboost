# -*- coding: utf-8 -*-
"""
Tests for offsets.Tick and subclasses
"""
from __future__ import division

from datetime import datetime, timedelta

from hypothesis import assume, example, given, settings, strategies as st
import numpy as np
import pytest

from pandas import Timedelta, Timestamp

from pandas.tseries import offsets
from pandas.tseries.offsets import Hour, Micro, Milli, Minute, Nano, Second

from .common import assert_offset_equal

# ---------------------------------------------------------------------
# Test Helpers

tick_classes = [Hour, Minute, Second, Milli, Micro, Nano]


# ---------------------------------------------------------------------


def test_apply_ticks():
    result = offsets.Hour(3).apply(offsets.Hour(4))
    exp = offsets.Hour(7)
    assert (result == exp)


def test_delta_to_tick():
    delta = timedelta(3)

    tick = offsets._delta_to_tick(delta)
    assert (tick == offsets.Day(3))

    td = Timedelta(nanoseconds=5)
    tick = offsets._delta_to_tick(td)
    assert tick == Nano(5)


@pytest.mark.parametrize('cls', tick_classes)
@settings(deadline=None)  # GH 24641
@example(n=2, m=3)
@example(n=800, m=300)
@example(n=1000, m=5)
@given(n=st.integers(-999, 999), m=st.integers(-999, 999))
def test_tick_add_sub(cls, n, m):
    # For all Tick subclasses and all integers n, m, we should have
    # tick(n) + tick(m) == tick(n+m)
    # tick(n) - tick(m) == tick(n-m)
    left = cls(n)
    right = cls(m)
    expected = cls(n + m)

    assert left + right == expected
    assert left.apply(right) == expected

    expected = cls(n - m)
    assert left - right == expected


@pytest.mark.parametrize('cls', tick_classes)
@settings(deadline=None)
@example(n=2, m=3)
@given(n=st.integers(-999, 999), m=st.integers(-999, 999))
def test_tick_equality(cls, n, m):
    assume(m != n)
    # tick == tock iff tick.n == tock.n
    left = cls(n)
    right = cls(m)
    assert left != right
    assert not (left == right)

    right = cls(n)
    assert left == right
    assert not (left != right)

    if n != 0:
        assert cls(n) != cls(-n)


# ---------------------------------------------------------------------


def test_Hour():
    assert_offset_equal(Hour(),
                        datetime(2010, 1, 1), datetime(2010, 1, 1, 1))
    assert_offset_equal(Hour(-1),
                        datetime(2010, 1, 1, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Hour(),
                        datetime(2010, 1, 1), datetime(2010, 1, 1, 2))
    assert_offset_equal(-1 * Hour(),
                        datetime(2010, 1, 1, 1), datetime(2010, 1, 1))

    assert Hour(3) + Hour(2) == Hour(5)
    assert Hour(3) - Hour(2) == Hour()

    assert Hour(4) != Hour(1)


def test_Minute():
    assert_offset_equal(Minute(),
                        datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 1))
    assert_offset_equal(Minute(-1),
                        datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))
    assert_offset_equal(2 * Minute(),
                        datetime(2010, 1, 1), datetime(2010, 1, 1, 0, 2))
    assert_offset_equal(-1 * Minute(),
                        datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 1))

    assert Minute(3) + Minute(2) == Minute(5)
    assert Minute(3) - Minute(2) == Minute()
    assert Minute(5) != Minute()


def test_Second():
    assert_offset_equal(Second(),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 1))
    assert_offset_equal(Second(-1),
                        datetime(2010, 1, 1, 0, 0, 1),
                        datetime(2010, 1, 1))
    assert_offset_equal(2 * Second(),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 2))
    assert_offset_equal(-1 * Second(),
                        datetime(2010, 1, 1, 0, 0, 1),
                        datetime(2010, 1, 1))

    assert Second(3) + Second(2) == Second(5)
    assert Second(3) - Second(2) == Second()


def test_Millisecond():
    assert_offset_equal(Milli(),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 0, 1000))
    assert_offset_equal(Milli(-1),
                        datetime(2010, 1, 1, 0, 0, 0, 1000),
                        datetime(2010, 1, 1))
    assert_offset_equal(Milli(2),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 0, 2000))
    assert_offset_equal(2 * Milli(),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 0, 2000))
    assert_offset_equal(-1 * Milli(),
                        datetime(2010, 1, 1, 0, 0, 0, 1000),
                        datetime(2010, 1, 1))

    assert Milli(3) + Milli(2) == Milli(5)
    assert Milli(3) - Milli(2) == Milli()


def test_MillisecondTimestampArithmetic():
    assert_offset_equal(Milli(),
                        Timestamp('2010-01-01'),
                        Timestamp('2010-01-01 00:00:00.001'))
    assert_offset_equal(Milli(-1),
                        Timestamp('2010-01-01 00:00:00.001'),
                        Timestamp('2010-01-01'))


def test_Microsecond():
    assert_offset_equal(Micro(),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 0, 1))
    assert_offset_equal(Micro(-1),
                        datetime(2010, 1, 1, 0, 0, 0, 1),
                        datetime(2010, 1, 1))

    assert_offset_equal(2 * Micro(),
                        datetime(2010, 1, 1),
                        datetime(2010, 1, 1, 0, 0, 0, 2))
    assert_offset_equal(-1 * Micro(),
                        datetime(2010, 1, 1, 0, 0, 0, 1),
                        datetime(2010, 1, 1))

    assert Micro(3) + Micro(2) == Micro(5)
    assert Micro(3) - Micro(2) == Micro()


def test_NanosecondGeneric():
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert timestamp.nanosecond == 0

    result = timestamp + Nano(10)
    assert result.nanosecond == 10

    reverse_result = Nano(10) + timestamp
    assert reverse_result.nanosecond == 10


def test_Nanosecond():
    timestamp = Timestamp(datetime(2010, 1, 1))
    assert_offset_equal(Nano(),
                        timestamp,
                        timestamp + np.timedelta64(1, 'ns'))
    assert_offset_equal(Nano(-1),
                        timestamp + np.timedelta64(1, 'ns'),
                        timestamp)
    assert_offset_equal(2 * Nano(),
                        timestamp,
                        timestamp + np.timedelta64(2, 'ns'))
    assert_offset_equal(-1 * Nano(),
                        timestamp + np.timedelta64(1, 'ns'),
                        timestamp)

    assert Nano(3) + Nano(2) == Nano(5)
    assert Nano(3) - Nano(2) == Nano()

    # GH9284
    assert Nano(1) + Nano(10) == Nano(11)
    assert Nano(5) + Micro(1) == Nano(1005)
    assert Micro(5) + Nano(1) == Nano(5001)


@pytest.mark.parametrize('kls, expected',
                         [(Hour, Timedelta(hours=5)),
                          (Minute, Timedelta(hours=2, minutes=3)),
                          (Second, Timedelta(hours=2, seconds=3)),
                          (Milli, Timedelta(hours=2, milliseconds=3)),
                          (Micro, Timedelta(hours=2, microseconds=3)),
                          (Nano, Timedelta(hours=2, nanoseconds=3))])
def test_tick_addition(kls, expected):
    offset = kls(3)
    result = offset + Timedelta(hours=2)
    assert isinstance(result, Timedelta)
    assert result == expected


@pytest.mark.parametrize('cls', tick_classes)
def test_tick_division(cls):
    off = cls(10)

    assert off / cls(5) == 2
    assert off / 2 == cls(5)
    assert off / 2.0 == cls(5)

    assert off / off.delta == 1
    assert off / off.delta.to_timedelta64() == 1

    assert off / Nano(1) == off.delta / Nano(1).delta

    if cls is not Nano:
        # A case where we end up with a smaller class
        result = off / 1000
        assert isinstance(result, offsets.Tick)
        assert not isinstance(result, cls)
        assert result.delta == off.delta / 1000

    if cls._inc < Timedelta(seconds=1):
        # Case where we end up with a bigger class
        result = off / .001
        assert isinstance(result, offsets.Tick)
        assert not isinstance(result, cls)
        assert result.delta == off.delta / .001


@pytest.mark.parametrize('cls1', tick_classes)
@pytest.mark.parametrize('cls2', tick_classes)
def test_tick_zero(cls1, cls2):
    assert cls1(0) == cls2(0)
    assert cls1(0) + cls2(0) == cls1(0)

    if cls1 is not Nano:
        assert cls1(2) + cls2(0) == cls1(2)

    if cls1 is Nano:
        assert cls1(2) + Nano(0) == cls1(2)


@pytest.mark.parametrize('cls', tick_classes)
def test_tick_equalities(cls):
    assert cls() == cls(1)


@pytest.mark.parametrize('cls', tick_classes)
def test_tick_offset(cls):
    assert not cls().isAnchored()


@pytest.mark.parametrize('cls', tick_classes)
def test_compare_ticks(cls):
    three = cls(3)
    four = cls(4)

    assert three < cls(4)
    assert cls(3) < four
    assert four > cls(3)
    assert cls(4) > three
    assert cls(3) == cls(3)
    assert cls(3) != cls(4)


@pytest.mark.parametrize('cls', tick_classes)
def test_compare_ticks_to_strs(cls):
    # GH#23524
    off = cls(19)

    # These tests should work with any strings, but we particularly are
    #  interested in "infer" as that comparison is convenient to make in
    #  Datetime/Timedelta Array/Index constructors
    assert not off == "infer"
    assert not "foo" == off

    for left, right in [("infer", off), (off, "infer")]:
        with pytest.raises(TypeError):
            left < right
        with pytest.raises(TypeError):
            left <= right
        with pytest.raises(TypeError):
            left > right
        with pytest.raises(TypeError):
            left >= right
