# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from datetime import datetime, timedelta, date, time
import itertools as it

from dateutil import tz
from dateutil.tz import UTC
from dateutil.parser import isoparser, isoparse

import pytest
import six


def _generate_tzoffsets(limited):
    def _mkoffset(hmtuple, fmt):
        h, m = hmtuple
        m_td = (-1 if h < 0 else 1) * m

        tzo = tz.tzoffset(None, timedelta(hours=h, minutes=m_td))
        return tzo, fmt.format(h, m)

    out = []
    if not limited:
        # The subset that's just hours
        hm_out_h = [(h, 0) for h in (-23, -5, 0, 5, 23)]
        out.extend([_mkoffset(hm, '{:+03d}') for hm in hm_out_h])

        # Ones that have hours and minutes
        hm_out = [] + hm_out_h
        hm_out += [(-12, 15), (11, 30), (10, 2), (5, 15), (-5, 30)]
    else:
        hm_out = [(-5, -0)]

    fmts = ['{:+03d}:{:02d}', '{:+03d}{:02d}']
    out += [_mkoffset(hm, fmt) for hm in hm_out for fmt in fmts]

    # Also add in UTC and naive
    out.append((UTC, 'Z'))
    out.append((None, ''))

    return out

FULL_TZOFFSETS = _generate_tzoffsets(False)
FULL_TZOFFSETS_AWARE = [x for x in FULL_TZOFFSETS if x[1]]
TZOFFSETS = _generate_tzoffsets(True)

DATES = [datetime(1996, 1, 1), datetime(2017, 1, 1)]
@pytest.mark.parametrize('dt', tuple(DATES))
def test_year_only(dt):
    dtstr = dt.strftime('%Y')

    assert isoparse(dtstr) == dt

DATES += [datetime(2000, 2, 1), datetime(2017, 4, 1)]
@pytest.mark.parametrize('dt', tuple(DATES))
def test_year_month(dt):
    fmt   = '%Y-%m'
    dtstr = dt.strftime(fmt)

    assert isoparse(dtstr) == dt

DATES += [datetime(2016, 2, 29), datetime(2018, 3, 15)]
YMD_FMTS = ('%Y%m%d', '%Y-%m-%d')
@pytest.mark.parametrize('dt', tuple(DATES))
@pytest.mark.parametrize('fmt', YMD_FMTS)
def test_year_month_day(dt, fmt):
    dtstr = dt.strftime(fmt)

    assert isoparse(dtstr) == dt

def _isoparse_date_and_time(dt, date_fmt, time_fmt, tzoffset,
                            microsecond_precision=None):
    tzi, offset_str = tzoffset
    fmt = date_fmt + 'T' + time_fmt
    dt = dt.replace(tzinfo=tzi)
    dtstr = dt.strftime(fmt)

    if microsecond_precision is not None:
        if not fmt.endswith('%f'):  # pragma: nocover
            raise ValueError('Time format has no microseconds!')

        if microsecond_precision != 6: 
            dtstr = dtstr[:-(6 - microsecond_precision)]
        elif microsecond_precision > 6: # pragma: nocover
            raise ValueError('Precision must be 1-6') 

    dtstr += offset_str

    assert isoparse(dtstr) == dt

DATETIMES = [datetime(1998, 4, 16, 12),
             datetime(2019, 11, 18, 23),
             datetime(2014, 12, 16, 4)]
@pytest.mark.parametrize('dt', tuple(DATETIMES))
@pytest.mark.parametrize('date_fmt', YMD_FMTS)
@pytest.mark.parametrize('tzoffset', TZOFFSETS)
def test_ymd_h(dt, date_fmt, tzoffset):
    _isoparse_date_and_time(dt, date_fmt, '%H', tzoffset)

DATETIMES = [datetime(2012, 1, 6, 9, 37)]
@pytest.mark.parametrize('dt', tuple(DATETIMES))
@pytest.mark.parametrize('date_fmt', YMD_FMTS)
@pytest.mark.parametrize('time_fmt', ('%H%M', '%H:%M'))
@pytest.mark.parametrize('tzoffset', TZOFFSETS)
def test_ymd_hm(dt, date_fmt, time_fmt, tzoffset):
    _isoparse_date_and_time(dt, date_fmt, time_fmt, tzoffset)

DATETIMES = [datetime(2003, 9, 2, 22, 14, 2),
             datetime(2003, 8, 8, 14, 9, 14),
             datetime(2003, 4, 7, 6, 14, 59)]
HMS_FMTS = ('%H%M%S', '%H:%M:%S')
@pytest.mark.parametrize('dt', tuple(DATETIMES))
@pytest.mark.parametrize('date_fmt', YMD_FMTS)
@pytest.mark.parametrize('time_fmt', HMS_FMTS)
@pytest.mark.parametrize('tzoffset', TZOFFSETS)
def test_ymd_hms(dt, date_fmt, time_fmt, tzoffset):
    _isoparse_date_and_time(dt, date_fmt, time_fmt, tzoffset)

DATETIMES = [datetime(2017, 11, 27, 6, 14, 30, 123456)]
@pytest.mark.parametrize('dt', tuple(DATETIMES))
@pytest.mark.parametrize('date_fmt', YMD_FMTS)
@pytest.mark.parametrize('time_fmt', (x + sep + '%f' for x in HMS_FMTS
                                      for sep in '.,'))
@pytest.mark.parametrize('tzoffset', TZOFFSETS)
@pytest.mark.parametrize('precision', list(range(3, 7)))
def test_ymd_hms_micro(dt, date_fmt, time_fmt, tzoffset, precision):
    # Truncate the microseconds to the desired precision for the representation
    dt = dt.replace(microsecond=int(round(dt.microsecond, precision-6)))

    _isoparse_date_and_time(dt, date_fmt, time_fmt, tzoffset, precision)

###
# Truncation of extra digits beyond microsecond precision
@pytest.mark.parametrize('dt_str', [
    '2018-07-03T14:07:00.123456000001',
    '2018-07-03T14:07:00.123456999999',
])
def test_extra_subsecond_digits(dt_str):
    assert isoparse(dt_str) == datetime(2018, 7, 3, 14, 7, 0, 123456)

@pytest.mark.parametrize('tzoffset', FULL_TZOFFSETS)
def test_full_tzoffsets(tzoffset):
    dt = datetime(2017, 11, 27, 6, 14, 30, 123456)
    date_fmt = '%Y-%m-%d'
    time_fmt = '%H:%M:%S.%f'

    _isoparse_date_and_time(dt, date_fmt, time_fmt, tzoffset)

@pytest.mark.parametrize('dt_str', [
    '2014-04-11T00',
    '2014-04-10T24',
    '2014-04-11T00:00',
    '2014-04-10T24:00',
    '2014-04-11T00:00:00',
    '2014-04-10T24:00:00',
    '2014-04-11T00:00:00.000',
    '2014-04-10T24:00:00.000',
    '2014-04-11T00:00:00.000000',
    '2014-04-10T24:00:00.000000']
)
def test_datetime_midnight(dt_str):
    assert isoparse(dt_str) == datetime(2014, 4, 11, 0, 0, 0, 0)

@pytest.mark.parametrize('datestr', [
    '2014-01-01',
    '20140101',
])
@pytest.mark.parametrize('sep', [' ', 'a', 'T', '_', '-'])
def test_isoparse_sep_none(datestr, sep):
    isostr = datestr + sep + '14:33:09'
    assert isoparse(isostr) == datetime(2014, 1, 1, 14, 33, 9)

##
# Uncommon date formats
TIME_ARGS = ('time_args',
    ((None, time(0), None), ) + tuple(('%H:%M:%S.%f', _t, _tz)
        for _t, _tz in it.product([time(0), time(9, 30), time(14, 47)],
                                  TZOFFSETS)))

@pytest.mark.parametrize('isocal,dt_expected',[
    ((2017, 10), datetime(2017, 3, 6)),
    ((2020, 1), datetime(2019, 12, 30)),    # ISO year != Cal year
    ((2004, 53), datetime(2004, 12, 27)),   # Only half the week is in 2014
])
def test_isoweek(isocal, dt_expected):
    # TODO: Figure out how to parametrize this on formats, too
    for fmt in ('{:04d}-W{:02d}', '{:04d}W{:02d}'):
        dtstr = fmt.format(*isocal)
        assert isoparse(dtstr) == dt_expected

@pytest.mark.parametrize('isocal,dt_expected',[
    ((2016, 13, 7), datetime(2016, 4, 3)),
    ((2004, 53, 7), datetime(2005, 1, 2)),      # ISO year != Cal year
    ((2009, 1, 2), datetime(2008, 12, 30)),     # ISO year < Cal year
    ((2009, 53, 6), datetime(2010, 1, 2))       # ISO year > Cal year
])
def test_isoweek_day(isocal, dt_expected):
    # TODO: Figure out how to parametrize this on formats, too
    for fmt in ('{:04d}-W{:02d}-{:d}', '{:04d}W{:02d}{:d}'):
        dtstr = fmt.format(*isocal)
        assert isoparse(dtstr) == dt_expected

@pytest.mark.parametrize('isoord,dt_expected', [
    ((2004, 1), datetime(2004, 1, 1)),
    ((2016, 60), datetime(2016, 2, 29)),
    ((2017, 60), datetime(2017, 3, 1)),
    ((2016, 366), datetime(2016, 12, 31)),
    ((2017, 365), datetime(2017, 12, 31))
])
def test_iso_ordinal(isoord, dt_expected):
    for fmt in ('{:04d}-{:03d}', '{:04d}{:03d}'):
        dtstr = fmt.format(*isoord)

        assert isoparse(dtstr) == dt_expected


###
# Acceptance of bytes
@pytest.mark.parametrize('isostr,dt', [
    (b'2014', datetime(2014, 1, 1)),
    (b'20140204', datetime(2014, 2, 4)),
    (b'2014-02-04', datetime(2014, 2, 4)),
    (b'2014-02-04T12', datetime(2014, 2, 4, 12)),
    (b'2014-02-04T12:30', datetime(2014, 2, 4, 12, 30)),
    (b'2014-02-04T12:30:15', datetime(2014, 2, 4, 12, 30, 15)),
    (b'2014-02-04T12:30:15.224', datetime(2014, 2, 4, 12, 30, 15, 224000)),
    (b'20140204T123015.224', datetime(2014, 2, 4, 12, 30, 15, 224000)),
    (b'2014-02-04T12:30:15.224Z', datetime(2014, 2, 4, 12, 30, 15, 224000,
                                           UTC)),
    (b'2014-02-04T12:30:15.224z', datetime(2014, 2, 4, 12, 30, 15, 224000,
                                           UTC)),
    (b'2014-02-04T12:30:15.224+05:00',
        datetime(2014, 2, 4, 12, 30, 15, 224000,
                 tzinfo=tz.tzoffset(None, timedelta(hours=5))))])
def test_bytes(isostr, dt):
    assert isoparse(isostr) == dt


###
# Invalid ISO strings
@pytest.mark.parametrize('isostr,exception', [
    ('201', ValueError),                        # ISO string too short
    ('2012-0425', ValueError),                  # Inconsistent date separators
    ('201204-25', ValueError),                  # Inconsistent date separators
    ('20120425T0120:00', ValueError),           # Inconsistent time separators
    ('20120425T01:2000', ValueError),           # Inconsistent time separators
    ('14:3015', ValueError),                    # Inconsistent time separator
    ('20120425T012500-334', ValueError),        # Wrong microsecond separator
    ('2001-1', ValueError),                     # YYYY-M not valid
    ('2012-04-9', ValueError),                  # YYYY-MM-D not valid
    ('201204', ValueError),                     # YYYYMM not valid
    ('20120411T03:30+', ValueError),            # Time zone too short
    ('20120411T03:30+1234567', ValueError),     # Time zone too long
    ('20120411T03:30-25:40', ValueError),       # Time zone invalid
    ('2012-1a', ValueError),                    # Invalid month
    ('20120411T03:30+00:60', ValueError),       # Time zone invalid minutes
    ('20120411T03:30+00:61', ValueError),       # Time zone invalid minutes
    ('20120411T033030.123456012:00',            # No sign in time zone
        ValueError),
    ('2012-W00', ValueError),                   # Invalid ISO week
    ('2012-W55', ValueError),                   # Invalid ISO week
    ('2012-W01-0', ValueError),                 # Invalid ISO week day
    ('2012-W01-8', ValueError),                 # Invalid ISO week day
    ('2013-000', ValueError),                   # Invalid ordinal day
    ('2013-366', ValueError),                   # Invalid ordinal day
    ('2013366', ValueError),                    # Invalid ordinal day
    ('2014-03-12Т12:30:14', ValueError),        # Cyrillic T
    ('2014-04-21T24:00:01', ValueError),        # Invalid use of 24 for midnight
    ('2014_W01-1', ValueError),                 # Invalid separator
    ('2014W01-1', ValueError),                  # Inconsistent use of dashes
    ('2014-W011', ValueError),                  # Inconsistent use of dashes

])
def test_iso_raises(isostr, exception):
    with pytest.raises(exception):
        isoparse(isostr)


@pytest.mark.parametrize('sep_act, valid_sep, exception', [
    ('T', 'C', ValueError),
    ('C', 'T', ValueError),
])
def test_iso_with_sep_raises(sep_act, valid_sep, exception):
    parser = isoparser(sep=valid_sep)
    isostr = '2012-04-25' + sep_act + '01:25:00'
    with pytest.raises(exception):
        parser.isoparse(isostr)


###
# Test ISOParser constructor
@pytest.mark.parametrize('sep', ['  ', '9', '🍛'])
def test_isoparser_invalid_sep(sep):
    with pytest.raises(ValueError):
        isoparser(sep=sep)


# This only fails on Python 3
@pytest.mark.xfail(not six.PY2, reason="Fails on Python 3 only")
def test_isoparser_byte_sep():
    dt = datetime(2017, 12, 6, 12, 30, 45)
    dt_str = dt.isoformat(sep=str('T'))

    dt_rt = isoparser(sep=b'T').isoparse(dt_str)

    assert dt == dt_rt


###
# Test parse_tzstr
@pytest.mark.parametrize('tzoffset', FULL_TZOFFSETS)
def test_parse_tzstr(tzoffset):
    dt = datetime(2017, 11, 27, 6, 14, 30, 123456)
    date_fmt = '%Y-%m-%d'
    time_fmt = '%H:%M:%S.%f'

    _isoparse_date_and_time(dt, date_fmt, time_fmt, tzoffset)


@pytest.mark.parametrize('tzstr', [
    '-00:00', '+00:00', '+00', '-00', '+0000', '-0000'
])
@pytest.mark.parametrize('zero_as_utc', [True, False])
def test_parse_tzstr_zero_as_utc(tzstr, zero_as_utc):
    tzi = isoparser().parse_tzstr(tzstr, zero_as_utc=zero_as_utc)
    assert tzi == UTC
    assert (type(tzi) == tz.tzutc) == zero_as_utc


@pytest.mark.parametrize('tzstr,exception', [
    ('00:00', ValueError),     # No sign
    ('05:00', ValueError),     # No sign
    ('_00:00', ValueError),    # Invalid sign
    ('+25:00', ValueError),    # Offset too large
    ('00:0000', ValueError),   # String too long
])
def test_parse_tzstr_fails(tzstr, exception):
    with pytest.raises(exception):
        isoparser().parse_tzstr(tzstr)

###
# Test parse_isodate
def __make_date_examples():
    dates_no_day = [
        date(1999, 12, 1),
        date(2016, 2, 1)
    ]

    if not six.PY2:
        # strftime does not support dates before 1900 in Python 2
        dates_no_day.append(date(1000, 11, 1))

    # Only one supported format for dates with no day
    o = zip(dates_no_day, it.repeat('%Y-%m'))

    dates_w_day = [
        date(1969, 12, 31),
        date(1900, 1, 1),
        date(2016, 2, 29),
        date(2017, 11, 14)
    ]

    dates_w_day_fmts = ('%Y%m%d', '%Y-%m-%d')
    o = it.chain(o, it.product(dates_w_day, dates_w_day_fmts))

    return list(o)


@pytest.mark.parametrize('d,dt_fmt', __make_date_examples())
@pytest.mark.parametrize('as_bytes', [True, False])
def test_parse_isodate(d, dt_fmt, as_bytes):
    d_str = d.strftime(dt_fmt)
    if isinstance(d_str, six.text_type) and as_bytes:
        d_str = d_str.encode('ascii')
    elif isinstance(d_str, bytes) and not as_bytes:
        d_str = d_str.decode('ascii')

    iparser = isoparser()
    assert iparser.parse_isodate(d_str) == d


@pytest.mark.parametrize('isostr,exception', [
    ('243', ValueError),                        # ISO string too short
    ('2014-0423', ValueError),                  # Inconsistent date separators
    ('201404-23', ValueError),                  # Inconsistent date separators
    ('2014日03月14', ValueError),                # Not ASCII
    ('2013-02-29', ValueError),                 # Not a leap year
    ('2014/12/03', ValueError),                 # Wrong separators
    ('2014-04-19T', ValueError),                # Unknown components
    ('201202', ValueError),                     # Invalid format
])
def test_isodate_raises(isostr, exception):
    with pytest.raises(exception):
        isoparser().parse_isodate(isostr)


def test_parse_isodate_error_text():
    with pytest.raises(ValueError) as excinfo:
        isoparser().parse_isodate('2014-0423')

    # ensure the error message does not contain b' prefixes
    if six.PY2:
        expected_error = "String contains unknown ISO components: u'2014-0423'"
    else:
        expected_error = "String contains unknown ISO components: '2014-0423'"
    assert expected_error == str(excinfo.value)


###
# Test parse_isotime
def __make_time_examples():
    outputs = []

    # HH
    time_h = [time(0), time(8), time(22)]
    time_h_fmts = ['%H']

    outputs.append(it.product(time_h, time_h_fmts))

    # HHMM / HH:MM
    time_hm = [time(0, 0), time(0, 30), time(8, 47), time(16, 1)]
    time_hm_fmts = ['%H%M', '%H:%M']

    outputs.append(it.product(time_hm, time_hm_fmts))

    # HHMMSS / HH:MM:SS
    time_hms = [time(0, 0, 0), time(0, 15, 30),
                time(8, 2, 16), time(12, 0), time(16, 2), time(20, 45)]

    time_hms_fmts = ['%H%M%S', '%H:%M:%S']

    outputs.append(it.product(time_hms, time_hms_fmts))

    # HHMMSS.ffffff / HH:MM:SS.ffffff
    time_hmsu = [time(0, 0, 0, 0), time(4, 15, 3, 247993),
                 time(14, 21, 59, 948730),
                 time(23, 59, 59, 999999)]

    time_hmsu_fmts = ['%H%M%S.%f', '%H:%M:%S.%f']

    outputs.append(it.product(time_hmsu, time_hmsu_fmts))

    outputs = list(map(list, outputs))

    # Time zones
    ex_naive = list(it.chain.from_iterable(x[0:2] for x in outputs))
    o = it.product(ex_naive, TZOFFSETS)    # ((time, fmt), (tzinfo, offsetstr))
    o = ((t.replace(tzinfo=tzi), fmt + off_str)
         for (t, fmt), (tzi, off_str) in o)

    outputs.append(o)

    return list(it.chain.from_iterable(outputs))


@pytest.mark.parametrize('time_val,time_fmt', __make_time_examples())
@pytest.mark.parametrize('as_bytes', [True, False])
def test_isotime(time_val, time_fmt, as_bytes):
    tstr = time_val.strftime(time_fmt)
    if isinstance(tstr, six.text_type) and as_bytes:
        tstr = tstr.encode('ascii')
    elif isinstance(tstr, bytes) and not as_bytes:
        tstr = tstr.decode('ascii')

    iparser = isoparser()

    assert iparser.parse_isotime(tstr) == time_val


@pytest.mark.parametrize('isostr', [
    '24:00',
    '2400',
    '24:00:00',
    '240000',
    '24:00:00.000',
    '24:00:00,000',
    '24:00:00.000000',
    '24:00:00,000000',
])
def test_isotime_midnight(isostr):
    iparser = isoparser()
    assert iparser.parse_isotime(isostr) == time(0, 0, 0, 0)


@pytest.mark.parametrize('isostr,exception', [
    ('3', ValueError),                          # ISO string too short
    ('14時30分15秒', ValueError),                # Not ASCII
    ('14_30_15', ValueError),                   # Invalid separators
    ('1430:15', ValueError),                    # Inconsistent separator use
    ('25', ValueError),                         # Invalid hours
    ('25:15', ValueError),                      # Invalid hours
    ('14:60', ValueError),                      # Invalid minutes
    ('14:59:61', ValueError),                   # Invalid seconds
    ('14:30:15.34468305:00', ValueError),       # No sign in time zone
    ('14:30:15+', ValueError),                  # Time zone too short
    ('14:30:15+1234567', ValueError),           # Time zone invalid
    ('14:59:59+25:00', ValueError),             # Invalid tz hours
    ('14:59:59+12:62', ValueError),             # Invalid tz minutes
    ('14:59:30_344583', ValueError),            # Invalid microsecond separator
    ('24:01', ValueError),                      # 24 used for non-midnight time
    ('24:00:01', ValueError),                   # 24 used for non-midnight time
    ('24:00:00.001', ValueError),               # 24 used for non-midnight time
    ('24:00:00.000001', ValueError),            # 24 used for non-midnight time
])
def test_isotime_raises(isostr, exception):
    iparser = isoparser()
    with pytest.raises(exception):
        iparser.parse_isotime(isostr)
