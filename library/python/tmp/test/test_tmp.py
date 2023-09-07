# coding: utf-8

import os

import library.python.tmp as tmp


def test_tmp_env():
    os.environ["TEST_VAL1"] = "1"
    try:
        with tmp.environment({"TEST_VAL2": "2"}):
            assert os.environ.get("TEST_VAL1") is None
            assert os.environ.get("TEST_VAL2") == "2"
            raise AssertionError()
    except AssertionError:
        assert os.environ.get("TEST_VAL1") == "1"


def test_temp_dir():
    with tmp.temp_dir() as t_dir:
        assert os.path.isdir(t_dir)
    assert not os.path.exists(t_dir)


def test_temp_file():
    with tmp.temp_file() as t_file:
        assert os.path.isfile(t_file)
    assert not os.path.exists(t_file)
