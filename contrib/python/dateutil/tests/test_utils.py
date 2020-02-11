# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import timedelta, datetime

import unittest

from dateutil import tz
from dateutil import utils
from dateutil.tz import UTC
from dateutil.utils import within_delta

from freezegun import freeze_time

NYC = tz.gettz("America/New_York")


class UtilsTest(unittest.TestCase):
    @freeze_time(datetime(2014, 12, 15, 1, 21, 33, 4003))
    def testToday(self):
        self.assertEqual(utils.today(), datetime(2014, 12, 15, 0, 0, 0))

    @freeze_time(datetime(2014, 12, 15, 12), tz_offset=5)
    def testTodayTzInfo(self):
        self.assertEqual(utils.today(NYC),
                         datetime(2014, 12, 15, 0, 0, 0, tzinfo=NYC))

    @freeze_time(datetime(2014, 12, 15, 23), tz_offset=5)
    def testTodayTzInfoDifferentDay(self):
        self.assertEqual(utils.today(UTC),
                         datetime(2014, 12, 16, 0, 0, 0, tzinfo=UTC))

    def testDefaultTZInfoNaive(self):
        dt = datetime(2014, 9, 14, 9, 30)
        self.assertIs(utils.default_tzinfo(dt, NYC).tzinfo,
                      NYC)

    def testDefaultTZInfoAware(self):
        dt = datetime(2014, 9, 14, 9, 30, tzinfo=UTC)
        self.assertIs(utils.default_tzinfo(dt, NYC).tzinfo,
                      UTC)

    def testWithinDelta(self):
        d1 = datetime(2016, 1, 1, 12, 14, 1, 9)
        d2 = d1.replace(microsecond=15)

        self.assertTrue(within_delta(d1, d2, timedelta(seconds=1)))
        self.assertFalse(within_delta(d1, d2, timedelta(microseconds=1)))

    def testWithinDeltaWithNegativeDelta(self):
        d1 = datetime(2016, 1, 1)
        d2 = datetime(2015, 12, 31)

        self.assertTrue(within_delta(d2, d1, timedelta(days=-1)))
