# coding: utf-8
"""Test suite for our JSON utilities."""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import datetime
from datetime import timedelta
import json

try:
    from unittest import mock
except ImportError:
    # py2
    import mock

from dateutil.tz import tzlocal, tzoffset
from jupyter_client import jsonutil
from jupyter_client.session import utcnow


def test_extract_dates():
    timestamps = [
        '2013-07-03T16:34:52.249482',
        '2013-07-03T16:34:52.249482Z',
        '2013-07-03T16:34:52.249482-0800',
        '2013-07-03T16:34:52.249482+0800',
        '2013-07-03T16:34:52.249482-08:00',
        '2013-07-03T16:34:52.249482+08:00',
    ]
    extracted = jsonutil.extract_dates(timestamps)
    ref = extracted[0]
    for dt in extracted:
        assert isinstance(dt, datetime.datetime)
        assert dt.tzinfo != None

    assert extracted[0].tzinfo.utcoffset(ref) == tzlocal().utcoffset(ref)
    assert extracted[1].tzinfo.utcoffset(ref) == timedelta(0)
    assert extracted[2].tzinfo.utcoffset(ref) == timedelta(hours=-8)
    assert extracted[3].tzinfo.utcoffset(ref) == timedelta(hours=8)
    assert extracted[4].tzinfo.utcoffset(ref) == timedelta(hours=-8)
    assert extracted[5].tzinfo.utcoffset(ref) == timedelta(hours=8)

def test_parse_ms_precision():
    base = '2013-07-03T16:34:52'
    digits = '1234567890'
    
    parsed = jsonutil.parse_date(base)
    assert isinstance(parsed, datetime.datetime)
    for i in range(len(digits)):
        ts = base + '.' + digits[:i]
        parsed = jsonutil.parse_date(ts)
        if i >= 1 and i <= 6:
            assert isinstance(parsed, datetime.datetime)
        else:
            assert isinstance(parsed, str)



def test_date_default():
    naive = datetime.datetime.now()
    local = tzoffset('Local', -8 * 3600)
    other = tzoffset('Other', 2 * 3600)
    data = dict(naive=naive, utc=utcnow(), withtz=naive.replace(tzinfo=other))
    with mock.patch.object(jsonutil, 'tzlocal', lambda : local):
        jsondata = json.dumps(data, default=jsonutil.date_default)
    assert "Z" in jsondata
    assert jsondata.count("Z") == 1
    extracted = jsonutil.extract_dates(json.loads(jsondata))
    for dt in extracted.values():
        assert isinstance(dt, datetime.datetime)
        assert dt.tzinfo != None

