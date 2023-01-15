from hypothesis import given, assume
from hypothesis import strategies as st

from dateutil import tz
from dateutil.parser import isoparse

import pytest

# Strategies
TIME_ZONE_STRATEGY = st.sampled_from([None, tz.UTC] +
    [tz.gettz(zname) for zname in ('US/Eastern', 'US/Pacific',
                                   'Australia/Sydney', 'Europe/London')])
ASCII_STRATEGY = st.characters(max_codepoint=127)


@pytest.mark.isoparser
@given(dt=st.datetimes(timezones=TIME_ZONE_STRATEGY), sep=ASCII_STRATEGY)
def test_timespec_auto(dt, sep):
    if dt.tzinfo is not None:
        # Assume offset has no sub-second components
        assume(dt.utcoffset().total_seconds() % 60 == 0)

    sep = str(sep)          # Python 2.7 requires bytes
    dtstr = dt.isoformat(sep=sep)
    dt_rt = isoparse(dtstr)

    assert dt_rt == dt
