# -*- coding: ascii -*-

import doctest

import pytz
import pytz.tzinfo

def test_doctest_pytz():
    nfailures, ntests = doctest.testmod(pytz)
    assert not nfailures


def test_doctest_pytz_tzinfo():
    nfailures, ntests = doctest.testmod(pytz.tzinfo)
    assert not nfailures
