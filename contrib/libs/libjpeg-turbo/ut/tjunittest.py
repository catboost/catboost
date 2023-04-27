#!/usr/bin/python

import yatest.common

_TJUNITTEST = "contrib/libs/libjpeg-turbo/tjunittest/tjunittest"

def run_unit(args=None):
    binary = yatest.common.binary_path(_TJUNITTEST)
    return yatest.common.canonical_execute(binary, args)

def test_tjunittest():
    return run_unit()

def test_tjunittest_alloc():
    return run_unit(["-alloc"])

def test_tjunittest_yuv():
    return run_unit(["-yuv"])

def test_tjunittest_yuv_alloc():
    return run_unit(["-yuv", "-alloc"])

def test_tjunittest_yuv_noyuvpad():
    return run_unit(["-yuv", "-noyuvpad"])
