# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import timedelta, datetime

from dateutil import tz
from dateutil import utils
from dateutil.tz import UTC
from dateutil.utils import within_delta

from freezegun import freeze_time

NYC = tz.gettz("America/New_York")


@freeze_time(datetime(2014, 12, 15, 1, 21, 33, 4003))
def test_utils_today():
    assert utils.today() == datetime(2014, 12, 15, 0, 0, 0)


@freeze_time(datetime(2014, 12, 15, 12), tz_offset=5)
def test_utils_today_tz_info():
    assert utils.today(NYC) == datetime(2014, 12, 15, 0, 0, 0, tzinfo=NYC)


@freeze_time(datetime(2014, 12, 15, 23), tz_offset=5)
def test_utils_today_tz_info_different_day():
    assert utils.today(UTC) == datetime(2014, 12, 16, 0, 0, 0, tzinfo=UTC)


def test_utils_default_tz_info_naive():
    dt = datetime(2014, 9, 14, 9, 30)
    assert utils.default_tzinfo(dt, NYC).tzinfo is NYC


def test_utils_default_tz_info_aware():
    dt = datetime(2014, 9, 14, 9, 30, tzinfo=UTC)
    assert utils.default_tzinfo(dt, NYC).tzinfo is UTC


def test_utils_within_delta():
    d1 = datetime(2016, 1, 1, 12, 14, 1, 9)
    d2 = d1.replace(microsecond=15)

    assert within_delta(d1, d2, timedelta(seconds=1))
    assert not within_delta(d1, d2, timedelta(microseconds=1))


def test_utils_within_delta_with_negative_delta():
    d1 = datetime(2016, 1, 1)
    d2 = datetime(2015, 12, 31)

    assert within_delta(d2, d1, timedelta(days=-1))
