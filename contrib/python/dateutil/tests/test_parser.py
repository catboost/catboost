# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import itertools
from datetime import datetime, timedelta
import unittest
import sys

from dateutil import tz
from dateutil.tz import tzoffset
from dateutil.parser import parse, parserinfo
from dateutil.parser import UnknownTimezoneWarning

from ._common import TZEnvContext

from six import assertRaisesRegex, PY3
from io import StringIO

import pytest

# Platform info
IS_WIN = sys.platform.startswith('win')

try:
    datetime.now().strftime('%-d')
    PLATFORM_HAS_DASH_D = True
except ValueError:
    PLATFORM_HAS_DASH_D = False

# Parser test cases using no keyword arguments. Format: (parsable_text, expected_datetime, assertion_message)
PARSER_TEST_CASES = [
    ("Thu Sep 25 10:36:28 2003", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    ("Thu Sep 25 2003", datetime(2003, 9, 25), "date command format strip"),
    ("2003-09-25T10:49:41", datetime(2003, 9, 25, 10, 49, 41), "iso format strip"),
    ("2003-09-25T10:49", datetime(2003, 9, 25, 10, 49), "iso format strip"),
    ("2003-09-25T10", datetime(2003, 9, 25, 10), "iso format strip"),
    ("2003-09-25", datetime(2003, 9, 25), "iso format strip"),
    ("20030925T104941", datetime(2003, 9, 25, 10, 49, 41), "iso stripped format strip"),
    ("20030925T1049", datetime(2003, 9, 25, 10, 49, 0), "iso stripped format strip"),
    ("20030925T10", datetime(2003, 9, 25, 10), "iso stripped format strip"),
    ("20030925", datetime(2003, 9, 25), "iso stripped format strip"),
    ("2003-09-25 10:49:41,502", datetime(2003, 9, 25, 10, 49, 41, 502000), "python logger format"),
    ("199709020908", datetime(1997, 9, 2, 9, 8), "no separator"),
    ("19970902090807", datetime(1997, 9, 2, 9, 8, 7), "no separator"),
    ("2003-09-25", datetime(2003, 9, 25), "date with dash"),
    ("09-25-2003", datetime(2003, 9, 25), "date with dash"),
    ("25-09-2003", datetime(2003, 9, 25), "date with dash"),
    ("10-09-2003", datetime(2003, 10, 9), "date with dash"),
    ("10-09-03", datetime(2003, 10, 9), "date with dash"),
    ("2003.09.25", datetime(2003, 9, 25), "date with dot"),
    ("09.25.2003", datetime(2003, 9, 25), "date with dot"),
    ("25.09.2003", datetime(2003, 9, 25), "date with dot"),
    ("10.09.2003", datetime(2003, 10, 9), "date with dot"),
    ("10.09.03", datetime(2003, 10, 9), "date with dot"),
    ("2003/09/25", datetime(2003, 9, 25), "date with slash"),
    ("09/25/2003", datetime(2003, 9, 25), "date with slash"),
    ("25/09/2003", datetime(2003, 9, 25), "date with slash"),
    ("10/09/2003", datetime(2003, 10, 9), "date with slash"),
    ("10/09/03", datetime(2003, 10, 9), "date with slash"),
    ("2003 09 25", datetime(2003, 9, 25), "date with space"),
    ("09 25 2003", datetime(2003, 9, 25), "date with space"),
    ("25 09 2003", datetime(2003, 9, 25), "date with space"),
    ("10 09 2003", datetime(2003, 10, 9), "date with space"),
    ("10 09 03", datetime(2003, 10, 9), "date with space"),
    ("25 09 03", datetime(2003, 9, 25), "date with space"),
    ("03 25 Sep", datetime(2003, 9, 25), "strangely ordered date"),
    ("25 03 Sep", datetime(2025, 9, 3), "strangely ordered date"),
    ("  July   4 ,  1976   12:01:02   am  ", datetime(1976, 7, 4, 0, 1, 2), "extra space"),
    ("Wed, July 10, '96", datetime(1996, 7, 10, 0, 0), "random format"),
    ("1996.July.10 AD 12:08 PM", datetime(1996, 7, 10, 12, 8), "random format"),
    ("July 4, 1976", datetime(1976, 7, 4), "random format"),
    ("7 4 1976", datetime(1976, 7, 4), "random format"),
    ("4 jul 1976", datetime(1976, 7, 4), "random format"),
    ("7-4-76", datetime(1976, 7, 4), "random format"),
    ("19760704", datetime(1976, 7, 4), "random format"),
    ("0:01:02 on July 4, 1976", datetime(1976, 7, 4, 0, 1, 2), "random format"),
    ("0:01:02 on July 4, 1976", datetime(1976, 7, 4, 0, 1, 2), "random format"),
    ("July 4, 1976 12:01:02 am", datetime(1976, 7, 4, 0, 1, 2), "random format"),
    ("Mon Jan  2 04:24:27 1995", datetime(1995, 1, 2, 4, 24, 27), "random format"),
    ("04.04.95 00:22", datetime(1995, 4, 4, 0, 22), "random format"),
    ("Jan 1 1999 11:23:34.578", datetime(1999, 1, 1, 11, 23, 34, 578000), "random format"),
    ("950404 122212", datetime(1995, 4, 4, 12, 22, 12), "random format"),
    ("3rd of May 2001", datetime(2001, 5, 3), "random format"),
    ("5th of March 2001", datetime(2001, 3, 5), "random format"),
    ("1st of May 2003", datetime(2003, 5, 1), "random format"),
    ('0099-01-01T00:00:00', datetime(99, 1, 1, 0, 0), "99 ad"),
    ('0031-01-01T00:00:00', datetime(31, 1, 1, 0, 0), "31 ad"),
    ("20080227T21:26:01.123456789", datetime(2008, 2, 27, 21, 26, 1, 123456), "high precision seconds"),
    ('13NOV2017', datetime(2017, 11, 13), "dBY (See GH360)"),
    ('0003-03-04', datetime(3, 3, 4), "pre 12 year same month (See GH PR #293)"),
    ('December.0031.30', datetime(31, 12, 30), "BYd corner case (GH#687)")
]


@pytest.mark.parametrize("parsable_text,expected_datetime,assertion_message", PARSER_TEST_CASES)
def test_parser(parsable_text, expected_datetime, assertion_message):
    assert parse(parsable_text) == expected_datetime, assertion_message


# Parser test cases using datetime(2003, 9, 25) as a default.
# Format: (parsable_text, expected_datetime, assertion_message)
PARSER_DEFAULT_TEST_CASES = [
    ("Thu Sep 25 10:36:28", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    ("Thu Sep 10:36:28", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    ("Thu 10:36:28", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    ("Sep 10:36:28", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    ("10:36:28", datetime(2003, 9, 25, 10, 36, 28), "date command format strip"),
    ("10:36", datetime(2003, 9, 25, 10, 36), "date command format strip"),
    ("Sep 2003", datetime(2003, 9, 25), "date command format strip"),
    ("Sep", datetime(2003, 9, 25), "date command format strip"),
    ("2003", datetime(2003, 9, 25), "date command format strip"),
    ("10h36m28.5s", datetime(2003, 9, 25, 10, 36, 28, 500000), "hour with letters"),
    ("10h36m28s", datetime(2003, 9, 25, 10, 36, 28), "hour with letters strip"),
    ("10h36m", datetime(2003, 9, 25, 10, 36), "hour with letters strip"),
    ("10h", datetime(2003, 9, 25, 10), "hour with letters strip"),
    ("10 h 36", datetime(2003, 9, 25, 10, 36), "hour with letters strip"),
    ("10 h 36.5", datetime(2003, 9, 25, 10, 36, 30), "hour with letter strip"),
    ("36 m 5", datetime(2003, 9, 25, 0, 36, 5), "hour with letters spaces"),
    ("36 m 5 s", datetime(2003, 9, 25, 0, 36, 5), "minute with letters spaces"),
    ("36 m 05", datetime(2003, 9, 25, 0, 36, 5), "minute with letters spaces"),
    ("36 m 05 s", datetime(2003, 9, 25, 0, 36, 5), "minutes with letters spaces"),
    ("10h am", datetime(2003, 9, 25, 10), "hour am pm"),
    ("10h pm", datetime(2003, 9, 25, 22), "hour am pm"),
    ("10am", datetime(2003, 9, 25, 10), "hour am pm"),
    ("10pm", datetime(2003, 9, 25, 22), "hour am pm"),
    ("10:00 am", datetime(2003, 9, 25, 10), "hour am pm"),
    ("10:00 pm", datetime(2003, 9, 25, 22), "hour am pm"),
    ("10:00am", datetime(2003, 9, 25, 10), "hour am pm"),
    ("10:00pm", datetime(2003, 9, 25, 22), "hour am pm"),
    ("10:00a.m", datetime(2003, 9, 25, 10), "hour am pm"),
    ("10:00p.m", datetime(2003, 9, 25, 22), "hour am pm"),
    ("10:00a.m.", datetime(2003, 9, 25, 10), "hour am pm"),
    ("10:00p.m.", datetime(2003, 9, 25, 22), "hour am pm"),
    ("Wed", datetime(2003, 10, 1), "weekday alone"),
    ("Wednesday", datetime(2003, 10, 1), "long weekday"),
    ("October", datetime(2003, 10, 25), "long month"),
    ("31-Dec-00", datetime(2000, 12, 31), "zero year"),
    ("0:01:02", datetime(2003, 9, 25, 0, 1, 2), "random format"),
    ("12h 01m02s am", datetime(2003, 9, 25, 0, 1, 2), "random format"),
    ("12:08 PM", datetime(2003, 9, 25, 12, 8), "random format"),
    ("01h02m03", datetime(2003, 9, 25, 1, 2, 3), "random format"),
    ("01h02", datetime(2003, 9, 25, 1, 2), "random format"),
    ("01h02s", datetime(2003, 9, 25, 1, 0, 2), "random format"),
    ("01m02", datetime(2003, 9, 25, 0, 1, 2), "random format"),
    ("01m02h", datetime(2003, 9, 25, 2, 1), "random format"),
    ("2004 10 Apr 11h30m", datetime(2004, 4, 10, 11, 30), "random format")
]


@pytest.mark.parametrize("parsable_text,expected_datetime,assertion_message", PARSER_DEFAULT_TEST_CASES)
def test_parser_default(parsable_text, expected_datetime, assertion_message):
    assert parse(parsable_text, default=datetime(2003, 9, 25)) == expected_datetime, assertion_message


class TestFormat(unittest.TestCase):

    def test_ybd(self):
        # If we have a 4-digit year, a non-numeric month (abbreviated or not),
        # and a day (1 or 2 digits), then there is no ambiguity as to which
        # token is a year/month/day.  This holds regardless of what order the
        # terms are in and for each of the separators below.

        seps = ['-', ' ', '/', '.']

        year_tokens = ['%Y']
        month_tokens = ['%b', '%B']
        day_tokens = ['%d']
        if PLATFORM_HAS_DASH_D:
            day_tokens.append('%-d')

        prods = itertools.product(year_tokens, month_tokens, day_tokens)
        perms = [y for x in prods for y in itertools.permutations(x)]
        unambig_fmts = [sep.join(perm) for sep in seps for perm in perms]

        actual = datetime(2003, 9, 25)

        for fmt in unambig_fmts:
            dstr = actual.strftime(fmt)
            res = parse(dstr)
            self.assertEqual(res, actual)


class TestInputFormats(object):
    def test_empty_string_invalid(self):
        with pytest.raises(ValueError):
            parse('')

    def test_none_invalid(self):
        with pytest.raises(TypeError):
            parse(None)

    def test_int_invalid(self):
        with pytest.raises(TypeError):
            parse(13)

    def test_duck_typing(self):
        # We want to support arbitrary classes that implement the stream
        # interface.

        class StringPassThrough(object):
            def __init__(self, stream):
                self.stream = stream

            def read(self, *args, **kwargs):
                return self.stream.read(*args, **kwargs)

        dstr = StringPassThrough(StringIO('2014 January 19'))

        res = parse(dstr)
        expected = datetime(2014, 1, 19)
        assert res == expected

    def test_parse_stream(self):
        dstr = StringIO('2014 January 19')

        res = parse(dstr)
        expected = datetime(2014, 1, 19)
        assert res == expected

    def test_parse_str(self):
        # Parser should be able to handle bytestring and unicode
        uni_str = '2014-05-01 08:00:00'
        bytes_str = uni_str.encode()

        res = parse(bytes_str)
        expected = parse(uni_str)
        assert res == expected

    def test_parse_bytes(self):
        res = parse(b'2014 January 19')
        expected = datetime(2014, 1, 19)
        assert res == expected

    def test_parse_bytearray(self):
        # GH#417
        res = parse(bytearray(b'2014 January 19'))
        expected = datetime(2014, 1, 19)
        assert res == expected


class ParserTest(unittest.TestCase):

    def setUp(self):
        self.tzinfos = {"BRST": -10800}
        self.brsttz = tzoffset("BRST", -10800)
        self.default = datetime(2003, 9, 25)

        # Parser should be able to handle bytestring and unicode
        self.uni_str = '2014-05-01 08:00:00'
        self.str_str = self.uni_str.encode()

    def testParserParseStr(self):
        from dateutil.parser import parser

        self.assertEqual(parser().parse(self.str_str),
                         parser().parse(self.uni_str))

    def testParseUnicodeWords(self):

        class rus_parserinfo(parserinfo):
            MONTHS = [("янв", "Январь"),
                      ("фев", "Февраль"),
                      ("мар", "Март"),
                      ("апр", "Апрель"),
                      ("май", "Май"),
                      ("июн", "Июнь"),
                      ("июл", "Июль"),
                      ("авг", "Август"),
                      ("сен", "Сентябрь"),
                      ("окт", "Октябрь"),
                      ("ноя", "Ноябрь"),
                      ("дек", "Декабрь")]

        self.assertEqual(parse('10 Сентябрь 2015 10:20',
                               parserinfo=rus_parserinfo()),
                         datetime(2015, 9, 10, 10, 20))

    def testParseWithNulls(self):
        # This relies on the from __future__ import unicode_literals, because
        # explicitly specifying a unicode literal is a syntax error in Py 3.2
        # May want to switch to u'...' if we ever drop Python 3.2 support.
        pstring = '\x00\x00August 29, 1924'

        self.assertEqual(parse(pstring),
                         datetime(1924, 8, 29))

    def testDateCommandFormat(self):
        self.assertEqual(parse("Thu Sep 25 10:36:28 BRST 2003",
                               tzinfos=self.tzinfos),
                         datetime(2003, 9, 25, 10, 36, 28,
                                  tzinfo=self.brsttz))

    def testDateCommandFormatUnicode(self):
        self.assertEqual(parse("Thu Sep 25 10:36:28 BRST 2003",
                               tzinfos=self.tzinfos),
                         datetime(2003, 9, 25, 10, 36, 28,
                                  tzinfo=self.brsttz))

    def testDateCommandFormatReversed(self):
        self.assertEqual(parse("2003 10:36:28 BRST 25 Sep Thu",
                               tzinfos=self.tzinfos),
                         datetime(2003, 9, 25, 10, 36, 28,
                                  tzinfo=self.brsttz))

    def testDateCommandFormatWithLong(self):
        if not PY3:
            self.assertEqual(parse("Thu Sep 25 10:36:28 BRST 2003",
                                   tzinfos={"BRST": long(-10800)}),
                             datetime(2003, 9, 25, 10, 36, 28,
                                      tzinfo=self.brsttz))

    def testDateCommandFormatIgnoreTz(self):
        self.assertEqual(parse("Thu Sep 25 10:36:28 BRST 2003",
                               ignoretz=True),
                         datetime(2003, 9, 25, 10, 36, 28))

    def testDateRCommandFormat(self):
        self.assertEqual(parse("Thu, 25 Sep 2003 10:49:41 -0300"),
                         datetime(2003, 9, 25, 10, 49, 41,
                                  tzinfo=self.brsttz))

    def testISOFormat(self):
        self.assertEqual(parse("2003-09-25T10:49:41.5-03:00"),
                         datetime(2003, 9, 25, 10, 49, 41, 500000,
                                  tzinfo=self.brsttz))

    def testISOFormatStrip1(self):
        self.assertEqual(parse("2003-09-25T10:49:41-03:00"),
                         datetime(2003, 9, 25, 10, 49, 41,
                                  tzinfo=self.brsttz))

    def testISOFormatStrip2(self):
        self.assertEqual(parse("2003-09-25T10:49:41+03:00"),
                         datetime(2003, 9, 25, 10, 49, 41,
                                  tzinfo=tzoffset(None, 10800)))

    def testISOStrippedFormat(self):
        self.assertEqual(parse("20030925T104941.5-0300"),
                         datetime(2003, 9, 25, 10, 49, 41, 500000,
                                  tzinfo=self.brsttz))

    def testISOStrippedFormatStrip1(self):
        self.assertEqual(parse("20030925T104941-0300"),
                         datetime(2003, 9, 25, 10, 49, 41,
                                  tzinfo=self.brsttz))

    def testISOStrippedFormatStrip2(self):
        self.assertEqual(parse("20030925T104941+0300"),
                         datetime(2003, 9, 25, 10, 49, 41,
                                  tzinfo=tzoffset(None, 10800)))

    def testDateWithDash8(self):
        self.assertEqual(parse("10-09-2003", dayfirst=True),
                         datetime(2003, 9, 10))

    def testDateWithDash11(self):
        self.assertEqual(parse("10-09-03", yearfirst=True),
                         datetime(2010, 9, 3))

    def testDateWithDot8(self):
        self.assertEqual(parse("10.09.2003", dayfirst=True),
                         datetime(2003, 9, 10))

    def testDateWithDot11(self):
        self.assertEqual(parse("10.09.03", yearfirst=True),
                         datetime(2010, 9, 3))

    def testDateWithSlash8(self):
        self.assertEqual(parse("10/09/2003", dayfirst=True),
                         datetime(2003, 9, 10))

    def testDateWithSlash11(self):
        self.assertEqual(parse("10/09/03", yearfirst=True),
                         datetime(2010, 9, 3))

    def testDateWithSpace8(self):
        self.assertEqual(parse("10 09 2003", dayfirst=True),
                         datetime(2003, 9, 10))

    def testDateWithSpace11(self):
        self.assertEqual(parse("10 09 03", yearfirst=True),
                         datetime(2010, 9, 3))

    def testAMPMNoHour(self):
        with self.assertRaises(ValueError):
            parse("AM")

        with self.assertRaises(ValueError):
            parse("Jan 20, 2015 PM")

    def testAMPMRange(self):
        with self.assertRaises(ValueError):
            parse("13:44 AM")

        with self.assertRaises(ValueError):
            parse("January 25, 1921 23:13 PM")

    def testPertain(self):
        self.assertEqual(parse("Sep 03", default=self.default),
                         datetime(2003, 9, 3))
        self.assertEqual(parse("Sep of 03", default=self.default),
                         datetime(2003, 9, 25))

    def testFuzzy(self):
        s = "Today is 25 of September of 2003, exactly " \
            "at 10:49:41 with timezone -03:00."
        self.assertEqual(parse(s, fuzzy=True),
                         datetime(2003, 9, 25, 10, 49, 41,
                                  tzinfo=self.brsttz))

    def testFuzzyWithTokens(self):
        s1 = "Today is 25 of September of 2003, exactly " \
            "at 10:49:41 with timezone -03:00."
        self.assertEqual(parse(s1, fuzzy_with_tokens=True),
                         (datetime(2003, 9, 25, 10, 49, 41,
                                   tzinfo=self.brsttz),
                         ('Today is ', 'of ', ', exactly at ',
                          ' with timezone ', '.')))

        s2 = "http://biz.yahoo.com/ipo/p/600221.html"
        self.assertEqual(parse(s2, fuzzy_with_tokens=True),
                         (datetime(2060, 2, 21, 0, 0, 0),
                         ('http://biz.yahoo.com/ipo/p/', '.html')))

    def testFuzzyAMPMProblem(self):
        # Sometimes fuzzy parsing results in AM/PM flag being set without
        # hours - if it's fuzzy it should ignore that.
        s1 = "I have a meeting on March 1, 1974."
        s2 = "On June 8th, 2020, I am going to be the first man on Mars"

        # Also don't want any erroneous AM or PMs changing the parsed time
        s3 = "Meet me at the AM/PM on Sunset at 3:00 AM on December 3rd, 2003"
        s4 = "Meet me at 3:00AM on December 3rd, 2003 at the AM/PM on Sunset"

        self.assertEqual(parse(s1, fuzzy=True), datetime(1974, 3, 1))
        self.assertEqual(parse(s2, fuzzy=True), datetime(2020, 6, 8))
        self.assertEqual(parse(s3, fuzzy=True), datetime(2003, 12, 3, 3))
        self.assertEqual(parse(s4, fuzzy=True), datetime(2003, 12, 3, 3))

    def testFuzzyIgnoreAMPM(self):
        s1 = "Jan 29, 1945 14:45 AM I going to see you there?"
        with pytest.warns(UnknownTimezoneWarning):
            res = parse(s1, fuzzy=True)
        self.assertEqual(res, datetime(1945, 1, 29, 14, 45))

    def testRandomFormat2(self):
        self.assertEqual(parse("1996.07.10 AD at 15:08:56 PDT",
                               ignoretz=True),
                         datetime(1996, 7, 10, 15, 8, 56))

    def testRandomFormat4(self):
        self.assertEqual(parse("Tuesday, April 12, 1952 AD 3:30:42pm PST",
                               ignoretz=True),
                         datetime(1952, 4, 12, 15, 30, 42))

    def testRandomFormat5(self):
        self.assertEqual(parse("November 5, 1994, 8:15:30 am EST",
                               ignoretz=True),
                         datetime(1994, 11, 5, 8, 15, 30))

    def testRandomFormat6(self):
        self.assertEqual(parse("1994-11-05T08:15:30-05:00",
                               ignoretz=True),
                         datetime(1994, 11, 5, 8, 15, 30))

    def testRandomFormat7(self):
        self.assertEqual(parse("1994-11-05T08:15:30Z",
                               ignoretz=True),
                         datetime(1994, 11, 5, 8, 15, 30))

    def testRandomFormat17(self):
        self.assertEqual(parse("1976-07-04T00:01:02Z", ignoretz=True),
                         datetime(1976, 7, 4, 0, 1, 2))

    def testRandomFormat18(self):
        self.assertEqual(parse("1986-07-05T08:15:30z",
                               ignoretz=True),
                         datetime(1986, 7, 5, 8, 15, 30))

    def testRandomFormat20(self):
        self.assertEqual(parse("Tue Apr 4 00:22:12 PDT 1995", ignoretz=True),
                         datetime(1995, 4, 4, 0, 22, 12))

    def testRandomFormat24(self):
        self.assertEqual(parse("0:00 PM, PST", default=self.default,
                               ignoretz=True),
                         datetime(2003, 9, 25, 12, 0))

    def testRandomFormat26(self):
        with pytest.warns(UnknownTimezoneWarning):
            res = parse("5:50 A.M. on June 13, 1990")

        self.assertEqual(res, datetime(1990, 6, 13, 5, 50))

    def testInvalidDay(self):
        with self.assertRaises(ValueError):
            parse("Feb 30, 2007")

    def testUnspecifiedDayFallback(self):
        # Test that for an unspecified day, the fallback behavior is correct.
        self.assertEqual(parse("April 2009", default=datetime(2010, 1, 31)),
                         datetime(2009, 4, 30))

    def testUnspecifiedDayFallbackFebNoLeapYear(self):
        self.assertEqual(parse("Feb 2007", default=datetime(2010, 1, 31)),
                         datetime(2007, 2, 28))

    def testUnspecifiedDayFallbackFebLeapYear(self):
        self.assertEqual(parse("Feb 2008", default=datetime(2010, 1, 31)),
                         datetime(2008, 2, 29))

    def testTzinfoDictionaryCouldReturnNone(self):
        self.assertEqual(parse('2017-02-03 12:40 BRST', tzinfos={"BRST": None}),
                        datetime(2017, 2, 3, 12, 40))

    def testTzinfosCallableCouldReturnNone(self):
        self.assertEqual(parse('2017-02-03 12:40 BRST', tzinfos=lambda *args: None),
                                    datetime(2017, 2, 3, 12, 40))

    def testErrorType01(self):
        self.assertRaises(ValueError,
                          parse, 'shouldfail')

    def testCorrectErrorOnFuzzyWithTokens(self):
        assertRaisesRegex(self, ValueError, 'Unknown string format',
                          parse, '04/04/32/423', fuzzy_with_tokens=True)
        assertRaisesRegex(self, ValueError, 'Unknown string format',
                          parse, '04/04/04 +32423', fuzzy_with_tokens=True)
        assertRaisesRegex(self, ValueError, 'Unknown string format',
                          parse, '04/04/0d4', fuzzy_with_tokens=True)

    def testIncreasingCTime(self):
        # This test will check 200 different years, every month, every day,
        # every hour, every minute, every second, and every weekday, using
        # a delta of more or less 1 year, 1 month, 1 day, 1 minute and
        # 1 second.
        delta = timedelta(days=365+31+1, seconds=1+60+60*60)
        dt = datetime(1900, 1, 1, 0, 0, 0, 0)
        for i in range(200):
            self.assertEqual(parse(dt.ctime()), dt)
            dt += delta

    def testIncreasingISOFormat(self):
        delta = timedelta(days=365+31+1, seconds=1+60+60*60)
        dt = datetime(1900, 1, 1, 0, 0, 0, 0)
        for i in range(200):
            self.assertEqual(parse(dt.isoformat()), dt)
            dt += delta

    def testMicrosecondsPrecisionError(self):
        # Skip found out that sad precision problem. :-(
        dt1 = parse("00:11:25.01")
        dt2 = parse("00:12:10.01")
        self.assertEqual(dt1.microsecond, 10000)
        self.assertEqual(dt2.microsecond, 10000)

    def testMicrosecondPrecisionErrorReturns(self):
        # One more precision issue, discovered by Eric Brown.  This should
        # be the last one, as we're no longer using floating points.
        for ms in [100001, 100000, 99999, 99998,
                    10001,  10000,  9999,  9998,
                     1001,   1000,   999,   998,
                      101,    100,    99,    98]:
            dt = datetime(2008, 2, 27, 21, 26, 1, ms)
            self.assertEqual(parse(dt.isoformat()), dt)

    def testCustomParserInfo(self):
        # Custom parser info wasn't working, as Michael Elsdörfer discovered.
        from dateutil.parser import parserinfo, parser

        class myparserinfo(parserinfo):
            MONTHS = parserinfo.MONTHS[:]
            MONTHS[0] = ("Foo", "Foo")
        myparser = parser(myparserinfo())
        dt = myparser.parse("01/Foo/2007")
        self.assertEqual(dt, datetime(2007, 1, 1))

    def testCustomParserShortDaynames(self):
        # Horacio Hoyos discovered that day names shorter than 3 characters,
        # for example two letter German day name abbreviations, don't work:
        # https://github.com/dateutil/dateutil/issues/343
        from dateutil.parser import parserinfo, parser

        class GermanParserInfo(parserinfo):
            WEEKDAYS = [("Mo", "Montag"),
                        ("Di", "Dienstag"),
                        ("Mi", "Mittwoch"),
                        ("Do", "Donnerstag"),
                        ("Fr", "Freitag"),
                        ("Sa", "Samstag"),
                        ("So", "Sonntag")]

        myparser = parser(GermanParserInfo())
        dt = myparser.parse("Sa 21. Jan 2017")
        self.assertEqual(dt, datetime(2017, 1, 21))

    def testNoYearFirstNoDayFirst(self):
        dtstr = '090107'

        # Should be MMDDYY
        self.assertEqual(parse(dtstr),
                         datetime(2007, 9, 1))

        self.assertEqual(parse(dtstr, yearfirst=False, dayfirst=False),
                         datetime(2007, 9, 1))

    def testYearFirst(self):
        dtstr = '090107'

        # Should be MMDDYY
        self.assertEqual(parse(dtstr, yearfirst=True),
                         datetime(2009, 1, 7))

        self.assertEqual(parse(dtstr, yearfirst=True, dayfirst=False),
                         datetime(2009, 1, 7))

    def testDayFirst(self):
        dtstr = '090107'

        # Should be DDMMYY
        self.assertEqual(parse(dtstr, dayfirst=True),
                         datetime(2007, 1, 9))

        self.assertEqual(parse(dtstr, yearfirst=False, dayfirst=True),
                         datetime(2007, 1, 9))

    def testDayFirstYearFirst(self):
        dtstr = '090107'
        # Should be YYDDMM
        self.assertEqual(parse(dtstr, yearfirst=True, dayfirst=True),
                         datetime(2009, 7, 1))

    def testUnambiguousYearFirst(self):
        dtstr = '2015 09 25'
        self.assertEqual(parse(dtstr, yearfirst=True),
                         datetime(2015, 9, 25))

    def testUnambiguousDayFirst(self):
        dtstr = '2015 09 25'
        self.assertEqual(parse(dtstr, dayfirst=True),
                         datetime(2015, 9, 25))

    def testUnambiguousDayFirstYearFirst(self):
        dtstr = '2015 09 25'
        self.assertEqual(parse(dtstr, dayfirst=True, yearfirst=True),
                         datetime(2015, 9, 25))

    def test_mstridx(self):
        # See GH408
        dtstr = '2015-15-May'
        self.assertEqual(parse(dtstr),
                         datetime(2015, 5, 15))

    def test_idx_check(self):
        dtstr = '2017-07-17 06:15:'
        # Pre-PR, the trailing colon will cause an IndexError at 824-825
        # when checking `i < len_l` and then accessing `l[i+1]`
        res = parse(dtstr, fuzzy=True)
        self.assertEqual(res, datetime(2017, 7, 17, 6, 15))

    def test_hmBY(self):
        # See GH#483
        dtstr = '02:17NOV2017'
        res = parse(dtstr, default=self.default)
        self.assertEqual(res, datetime(2017, 11, self.default.day, 2, 17))

    def test_validate_hour(self):
        # See GH353
        invalid = "201A-01-01T23:58:39.239769+03:00"
        with self.assertRaises(ValueError):
            parse(invalid)

    def test_era_trailing_year(self):
        dstr = 'AD2001'
        res = parse(dstr)
        assert res.year == 2001, res


class TestParseUnimplementedCases(object):
    @pytest.mark.xfail
    def test_somewhat_ambiguous_string(self):
        # Ref: github issue #487
        # The parser is choosing the wrong part for hour
        # causing datetime to raise an exception.
        dtstr = '1237 PM BRST Mon Oct 30 2017'
        res = parse(dtstr, tzinfo=self.tzinfos)
        assert res == datetime(2017, 10, 30, 12, 37, tzinfo=self.tzinfos)

    @pytest.mark.xfail
    def test_YmdH_M_S(self):
        # found in nasdaq's ftp data
        dstr = '1991041310:19:24'
        expected = datetime(1991, 4, 13, 10, 19, 24)
        res = parse(dstr)
        assert res == expected, (res, expected)

    @pytest.mark.xfail
    def test_first_century(self):
        dstr = '0031 Nov 03'
        expected = datetime(31, 11, 3)
        res = parse(dstr)
        assert res == expected, res

    @pytest.mark.xfail
    def test_era_trailing_year_with_dots(self):
        dstr = 'A.D.2001'
        res = parse(dstr)
        assert res.year == 2001, res

    @pytest.mark.xfail
    def test_ad_nospace(self):
        expected = datetime(6, 5, 19)
        for dstr in [' 6AD May 19', ' 06AD May 19',
                     ' 006AD May 19', ' 0006AD May 19']:
            res = parse(dstr)
            assert res == expected, (dstr, res)

    @pytest.mark.xfail
    def test_four_letter_day(self):
        dstr = 'Frid Dec 30, 2016'
        expected = datetime(2016, 12, 30)
        res = parse(dstr)
        assert res == expected

    @pytest.mark.xfail
    def test_non_date_number(self):
        dstr = '1,700'
        with pytest.raises(ValueError):
            parse(dstr)

    @pytest.mark.xfail
    def test_on_era(self):
        # This could be classified as an "eras" test, but the relevant part
        # about this is the ` on `
        dstr = '2:15 PM on January 2nd 1973 A.D.'
        expected = datetime(1973, 1, 2, 14, 15)
        res = parse(dstr)
        assert res == expected

    @pytest.mark.xfail
    def test_extraneous_year(self):
        # This was found in the wild at insidertrading.org
        dstr = "2011 MARTIN CHILDREN'S IRREVOCABLE TRUST u/a/d NOVEMBER 7, 2012"
        res = parse(dstr, fuzzy_with_tokens=True)
        expected = datetime(2012, 11, 7)
        assert res == expected

    @pytest.mark.xfail
    def test_extraneous_year_tokens(self):
        # This was found in the wild at insidertrading.org
        # Unlike in the case above, identifying the first "2012" as the year
        # would not be a problem, but infering that the latter 2012 is hhmm
        # is a problem.
        dstr = "2012 MARTIN CHILDREN'S IRREVOCABLE TRUST u/a/d NOVEMBER 7, 2012"
        expected = datetime(2012, 11, 7)
        (res, tokens) = parse(dstr, fuzzy_with_tokens=True)
        assert res == expected
        assert tokens == ("2012 MARTIN CHILDREN'S IRREVOCABLE TRUST u/a/d ",)

    @pytest.mark.xfail
    def test_extraneous_year2(self):
        # This was found in the wild at insidertrading.org
        dstr = ("Berylson Amy Smith 1998 Grantor Retained Annuity Trust "
                "u/d/t November 2, 1998 f/b/o Jennifer L Berylson")
        res = parse(dstr, fuzzy_with_tokens=True)
        expected = datetime(1998, 11, 2)
        assert res == expected

    @pytest.mark.xfail
    def test_extraneous_year3(self):
        # This was found in the wild at insidertrading.org
        dstr = "SMITH R &  WEISS D 94 CHILD TR FBO M W SMITH UDT 12/1/1994"
        res = parse(dstr, fuzzy_with_tokens=True)
        expected = datetime(1994, 12, 1)
        assert res == expected

    @pytest.mark.xfail
    def test_unambiguous_YYYYMM(self):
        # 171206 can be parsed as YYMMDD. However, 201712 cannot be parsed
        # as instance of YYMMDD and parser could fallback to YYYYMM format.
        dstr = "201712"
        res = parse(dstr)
        expected = datetime(2017, 12, 1)
        assert res == expected


@pytest.mark.skipif(IS_WIN, reason='Windows does not use TZ var')
@pytest.mark.skipif(
    not TZEnvContext.tz_change_allowed(),
    reason=TZEnvContext.tz_change_disallowed_message()
)
def test_parse_unambiguous_nonexistent_local():
    # When dates are specified "EST" even when they should be "EDT" in the
    # local time zone, we should still assign the local time zone
    with TZEnvContext('EST+5EDT,M3.2.0/2,M11.1.0/2'):
        dt_exp = datetime(2011, 8, 1, 12, 30, tzinfo=tz.tzlocal())
        dt = parse('2011-08-01T12:30 EST')

        assert dt.tzname() == 'EDT'
        assert dt == dt_exp


@pytest.mark.skipif(IS_WIN, reason='Windows does not use TZ var')
@pytest.mark.skipif(
    not TZEnvContext.tz_change_allowed(),
    reason=TZEnvContext.tz_change_disallowed_message()
)
def test_tzlocal_in_gmt():
    # GH #318
    with TZEnvContext('GMT0BST,M3.5.0,M10.5.0'):
        # This is an imaginary datetime in tz.tzlocal() but should still
        # parse using the GMT-as-alias-for-UTC rule
        dt = parse('2004-05-01T12:00 GMT')
        dt_exp = datetime(2004, 5, 1, 12, tzinfo=tz.tzutc())

        assert dt == dt_exp


@pytest.mark.skipif(IS_WIN, reason='Windows does not use TZ var')
@pytest.mark.skipif(
    not TZEnvContext.tz_change_allowed(),
    reason=TZEnvContext.tz_change_disallowed_message()
)
def test_tzlocal_parse_fold():
    # One manifestion of GH #318
    with TZEnvContext('EST+5EDT,M3.2.0/2,M11.1.0/2'):
        dt_exp = datetime(2011, 11, 6, 1, 30, tzinfo=tz.tzlocal())
        dt_exp = tz.enfold(dt_exp, fold=1)
        dt = parse('2011-11-06T01:30 EST')

        # Because this is ambiguous, kuntil `tz.tzlocal() is tz.tzlocal()`
        # we'll just check the attributes we care about rather than
        # dt == dt_exp
        assert dt.tzname() == dt_exp.tzname()
        assert dt.replace(tzinfo=None) == dt_exp.replace(tzinfo=None)
        assert getattr(dt, 'fold') == getattr(dt_exp, 'fold')
        assert dt.astimezone(tz.tzutc()) == dt_exp.astimezone(tz.tzutc())


def test_parse_tzinfos_fold():
    NYC = tz.gettz('America/New_York')
    tzinfos = {'EST': NYC, 'EDT': NYC}

    dt_exp = tz.enfold(datetime(2011, 11, 6, 1, 30, tzinfo=NYC), fold=1)
    dt = parse('2011-11-06T01:30 EST', tzinfos=tzinfos)

    assert dt == dt_exp
    assert dt.tzinfo is dt_exp.tzinfo
    assert getattr(dt, 'fold') == getattr(dt_exp, 'fold')
    assert dt.astimezone(tz.tzutc()) == dt_exp.astimezone(tz.tzutc())


@pytest.mark.parametrize('dtstr,dt', [
    ('5.6h', datetime(2003, 9, 25, 5, 36)),
    ('5.6m', datetime(2003, 9, 25, 0, 5, 36)),
    # '5.6s' never had a rounding problem, test added for completeness
    ('5.6s', datetime(2003, 9, 25, 0, 0, 5, 600000))
])
def test_rounding_floatlike_strings(dtstr, dt):
    assert parse(dtstr, default=datetime(2003, 9, 25)) == dt


@pytest.mark.parametrize('value', ['1: test', 'Nan'])
def test_decimal_error(value):
    # GH 632, GH 662 - decimal.Decimal raises some non-ValueError exception when
    # constructed with an invalid value
    with pytest.raises(ValueError):
        parse(value)
