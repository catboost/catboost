import sys
import unittest

class ImportVersionTest(unittest.TestCase):
    """ Test that dateutil.__version__ can be imported"""

    def testImportVersionStr(self):
        from dateutil import __version__

    def testImportRoot(self):
        import dateutil

        self.assertTrue(hasattr(dateutil, '__version__'))


class ImportEasterTest(unittest.TestCase):
    """ Test that dateutil.easter-related imports work properly """

    def testEasterDirect(self):
        import dateutil.easter

    def testEasterFrom(self):
        from dateutil import easter

    def testEasterStar(self):
        from dateutil.easter import easter


class ImportParserTest(unittest.TestCase):
    """ Test that dateutil.parser-related imports work properly """
    def testParserDirect(self):
        import dateutil.parser

    def testParserFrom(self):
        from dateutil import parser

    def testParserAll(self):
        # All interface
        from dateutil.parser import parse
        from dateutil.parser import parserinfo

        # Other public classes
        from dateutil.parser import parser

        for var in (parse, parserinfo, parser):
            self.assertIsNot(var, None)


class ImportRelativeDeltaTest(unittest.TestCase):
    """ Test that dateutil.relativedelta-related imports work properly """
    def testRelativeDeltaDirect(self):
        import dateutil.relativedelta

    def testRelativeDeltaFrom(self):
        from dateutil import relativedelta

    def testRelativeDeltaAll(self):
        from dateutil.relativedelta import relativedelta
        from dateutil.relativedelta import MO, TU, WE, TH, FR, SA, SU

        for var in (relativedelta, MO, TU, WE, TH, FR, SA, SU):
            self.assertIsNot(var, None)

        # In the public interface but not in all
        from dateutil.relativedelta import weekday
        self.assertIsNot(weekday, None)


class ImportRRuleTest(unittest.TestCase):
    """ Test that dateutil.rrule related imports work properly """
    def testRRuleDirect(self):
        import dateutil.rrule

    def testRRuleFrom(self):
        from dateutil import rrule

    def testRRuleAll(self):
        from dateutil.rrule import rrule
        from dateutil.rrule import rruleset
        from dateutil.rrule import rrulestr
        from dateutil.rrule import YEARLY, MONTHLY, WEEKLY, DAILY
        from dateutil.rrule import HOURLY, MINUTELY, SECONDLY
        from dateutil.rrule import MO, TU, WE, TH, FR, SA, SU

        rr_all = (rrule, rruleset, rrulestr,
                  YEARLY, MONTHLY, WEEKLY, DAILY,
                  HOURLY, MINUTELY, SECONDLY,
                  MO, TU, WE, TH, FR, SA, SU)

        for var in rr_all:
            self.assertIsNot(var, None)

        # In the public interface but not in all
        from dateutil.rrule import weekday
        self.assertIsNot(weekday, None)


class ImportTZTest(unittest.TestCase):
    """ Test that dateutil.tz related imports work properly """
    def testTzDirect(self):
        import dateutil.tz

    def testTzFrom(self):
        from dateutil import tz

    def testTzAll(self):
        from dateutil.tz import tzutc
        from dateutil.tz import tzoffset
        from dateutil.tz import tzlocal
        from dateutil.tz import tzfile
        from dateutil.tz import tzrange
        from dateutil.tz import tzstr
        from dateutil.tz import tzical
        from dateutil.tz import gettz
        from dateutil.tz import tzwin
        from dateutil.tz import tzwinlocal
        from dateutil.tz import UTC
        from dateutil.tz import datetime_ambiguous
        from dateutil.tz import datetime_exists
        from dateutil.tz import resolve_imaginary

        tz_all = ["tzutc", "tzoffset", "tzlocal", "tzfile", "tzrange",
                  "tzstr", "tzical", "gettz", "datetime_ambiguous",
                  "datetime_exists", "resolve_imaginary", "UTC"]

        tz_all += ["tzwin", "tzwinlocal"] if sys.platform.startswith("win") else []
        lvars = locals()

        for var in tz_all:
            self.assertIsNot(lvars[var], None)

@unittest.skipUnless(sys.platform.startswith('win'), "Requires Windows")
class ImportTZWinTest(unittest.TestCase):
    """ Test that dateutil.tzwin related imports work properly """
    def testTzwinDirect(self):
        import dateutil.tzwin

    def testTzwinFrom(self):
        from dateutil import tzwin

    def testTzwinStar(self):
        from dateutil.tzwin import tzwin
        from dateutil.tzwin import tzwinlocal

        tzwin_all = [tzwin, tzwinlocal]

        for var in tzwin_all:
            self.assertIsNot(var, None)


class ImportZoneInfoTest(unittest.TestCase):
    def testZoneinfoDirect(self):
        import dateutil.zoneinfo

    def testZoneinfoFrom(self):
        from dateutil import zoneinfo

    def testZoneinfoStar(self):
        from dateutil.zoneinfo import gettz
        from dateutil.zoneinfo import gettz_db_metadata
        from dateutil.zoneinfo import rebuild

        zi_all = (gettz, gettz_db_metadata, rebuild)

        for var in zi_all:
            self.assertIsNot(var, None)
