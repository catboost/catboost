# -*- coding: utf-8 -*-
from datetime import datetime

import dateutil.tz
import pytest
import pytz

from pandas._libs.tslibs import conversion, timezones

from pandas import Timestamp


@pytest.mark.parametrize("tz_name", list(pytz.common_timezones))
def test_cache_keys_are_distinct_for_pytz_vs_dateutil(tz_name):
    if tz_name == "UTC":
        pytest.skip("UTC: special case in dateutil")

    tz_p = timezones.maybe_get_tz(tz_name)
    tz_d = timezones.maybe_get_tz("dateutil/" + tz_name)

    if tz_d is None:
        pytest.skip(tz_name + ": dateutil does not know about this one")

    assert timezones._p_tz_cache_key(tz_p) != timezones._p_tz_cache_key(tz_d)


def test_tzlocal_repr():
    # see gh-13583
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())
    assert ts.tz == dateutil.tz.tzlocal()
    assert "tz='tzlocal()')" in repr(ts)


def test_tzlocal_maybe_get_tz():
    # see gh-13583
    tz = timezones.maybe_get_tz('tzlocal()')
    assert tz == dateutil.tz.tzlocal()


def test_tzlocal_offset():
    # see gh-13583
    #
    # Get offset using normal datetime for test.
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())

    offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
    offset = offset.total_seconds() * 1000000000

    assert ts.value + offset == Timestamp("2011-01-01").value


@pytest.fixture(params=[
    (pytz.timezone("US/Eastern"), lambda tz, x: tz.localize(x)),
    (dateutil.tz.gettz("US/Eastern"), lambda tz, x: x.replace(tzinfo=tz))
])
def infer_setup(request):
    eastern, localize = request.param

    start_naive = datetime(2001, 1, 1)
    end_naive = datetime(2009, 1, 1)

    start = localize(eastern, start_naive)
    end = localize(eastern, end_naive)

    return eastern, localize, start, end, start_naive, end_naive


def test_infer_tz_compat(infer_setup):
    eastern, _, start, end, start_naive, end_naive = infer_setup

    assert (timezones.infer_tzinfo(start, end) is
            conversion.localize_pydatetime(start_naive, eastern).tzinfo)
    assert (timezones.infer_tzinfo(start, None) is
            conversion.localize_pydatetime(start_naive, eastern).tzinfo)
    assert (timezones.infer_tzinfo(None, end) is
            conversion.localize_pydatetime(end_naive, eastern).tzinfo)


def test_infer_tz_utc_localize(infer_setup):
    _, _, start, end, start_naive, end_naive = infer_setup
    utc = pytz.utc

    start = utc.localize(start_naive)
    end = utc.localize(end_naive)

    assert timezones.infer_tzinfo(start, end) is utc


@pytest.mark.parametrize("ordered", [True, False])
def test_infer_tz_mismatch(infer_setup, ordered):
    eastern, _, _, _, start_naive, end_naive = infer_setup
    msg = "Inputs must both have the same timezone"

    utc = pytz.utc
    start = utc.localize(start_naive)
    end = conversion.localize_pydatetime(end_naive, eastern)

    args = (start, end) if ordered else (end, start)

    with pytest.raises(AssertionError, match=msg):
        timezones.infer_tzinfo(*args)
