# -*- coding: utf-8 -*-
import os

import pytest

from jinja2 import Environment
from jinja2 import loaders
from jinja2.utils import have_async_gen


def pytest_ignore_collect(path):
    if "async" in path.basename and not have_async_gen:
        return True
    return False


@pytest.fixture
def env():
    """returns a new environment."""
    return Environment()


@pytest.fixture
def dict_loader():
    """returns DictLoader"""
    return loaders.DictLoader({"justdict.html": "FOO"})


@pytest.fixture
def package_loader():
    """returns PackageLoader initialized from templates"""
    return loaders.PackageLoader("res", "templates")


@pytest.fixture
def filesystem_loader():
    """returns FileSystemLoader initialized to res/templates directory"""
    try:
        import yatest.common
        here =  yatest.common.test_source_path()
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
    return loaders.FileSystemLoader(here + "/res/templates")


@pytest.fixture
def function_loader():
    """returns a FunctionLoader"""
    return loaders.FunctionLoader({"justfunction.html": "FOO"}.get)


@pytest.fixture
def choice_loader(dict_loader, package_loader):
    """returns a ChoiceLoader"""
    return loaders.ChoiceLoader([dict_loader, package_loader])


@pytest.fixture
def prefix_loader(filesystem_loader, dict_loader):
    """returns a PrefixLoader"""
    return loaders.PrefixLoader({"a": filesystem_loader, "b": dict_loader})
