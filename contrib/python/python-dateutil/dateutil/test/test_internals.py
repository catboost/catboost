# -*- coding: utf-8 -*-
"""
Tests for implementation details, not necessarily part of the user-facing
API.

The motivating case for these tests is #483, where we want to smoke-test
code that may be difficult to reach through the standard API calls.
"""

import sys
import pytest

from dateutil.parser._parser import _ymd
from dateutil import tz

IS_PY32 = sys.version_info[0:2] == (3, 2)


@pytest.mark.smoke
def test_YMD_could_be_day():
    ymd = _ymd('foo bar 124 baz')

    ymd.append(2, 'M')
    assert ymd.has_month
    assert not ymd.has_year
    assert ymd.could_be_day(4)
    assert not ymd.could_be_day(-6)
    assert not ymd.could_be_day(32)

    # Assumes leap year
    assert ymd.could_be_day(29)

    ymd.append(1999)
    assert ymd.has_year
    assert not ymd.could_be_day(29)

    ymd.append(16, 'D')
    assert ymd.has_day
    assert not ymd.could_be_day(1)

    ymd = _ymd('foo bar 124 baz')
    ymd.append(1999)
    assert ymd.could_be_day(31)


###
# Test that private interfaces in _parser are deprecated properly
@pytest.mark.skipif(IS_PY32, reason='pytest.warns not supported on Python 3.2')
def test_parser_private_warns():
    from dateutil.parser import _timelex, _tzparser
    from dateutil.parser import _parsetz

    with pytest.warns(DeprecationWarning):
        _tzparser()

    with pytest.warns(DeprecationWarning):
        _timelex('2014-03-03')

    with pytest.warns(DeprecationWarning):
        _parsetz('+05:00')


@pytest.mark.skipif(IS_PY32, reason='pytest.warns not supported on Python 3.2')
def test_parser_parser_private_not_warns():
    from dateutil.parser._parser import _timelex, _tzparser
    from dateutil.parser._parser import _parsetz

    with pytest.warns(None) as recorder:
        _tzparser()
        assert len(recorder) == 0

    with pytest.warns(None) as recorder:
        _timelex('2014-03-03')

        assert len(recorder) == 0

    with pytest.warns(None) as recorder:
        _parsetz('+05:00')
        assert len(recorder) == 0


@pytest.mark.tzstr
def test_tzstr_internal_timedeltas():
    with pytest.warns(tz.DeprecatedTzFormatWarning):
        tz1 = tz.tzstr("EST5EDT,5,4,0,7200,11,-3,0,7200")

    with pytest.warns(tz.DeprecatedTzFormatWarning):
        tz2 = tz.tzstr("EST5EDT,4,1,0,7200,10,-1,0,7200")

    assert tz1._start_delta != tz2._start_delta
    assert tz1._end_delta != tz2._end_delta
