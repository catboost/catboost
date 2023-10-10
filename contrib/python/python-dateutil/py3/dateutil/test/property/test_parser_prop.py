from hypothesis.strategies import integers
from hypothesis import given

import pytest

from dateutil.parser import parserinfo


@pytest.mark.parserinfo
@given(integers(min_value=100, max_value=9999))
def test_convertyear(n):
    assert n == parserinfo().convertyear(n)


@pytest.mark.parserinfo
@given(integers(min_value=-50,
                max_value=49))
def test_convertyear_no_specified_century(n):
    p = parserinfo()
    new_year = p._year + n
    result = p.convertyear(new_year % 100, century_specified=False)
    assert result == new_year
