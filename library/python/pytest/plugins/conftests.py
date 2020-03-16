import os
import importlib
import sys
import inspect

from pytest import hookimpl

from .fixtures import metrics, links  # noqa

orig_getfile = inspect.getfile


def getfile(object):
    res = orig_getfile(object)
    if inspect.ismodule(object):
        if not res and getattr(object, '__orig_file__'):
            res = object.__orig_file__
    return res

inspect.getfile = getfile
conftest_modules = []


@hookimpl(trylast=True)
def pytest_load_initial_conftests(early_config, parser, args):
    conftests = filter(lambda name: name.endswith(".conftest"), sys.extra_modules)

    def conftest_key(name):
        if not name.startswith("__tests__."):
            # Make __tests__ come last
            return "_." + name
        return name

    for name in sorted(conftests, key=conftest_key):
        mod = importlib.import_module(name)
        if os.getenv("CONFTEST_LOAD_POLICY") != "LOCAL":
            mod.__orig_file__ = mod.__file__
            mod.__file__ = ""
        conftest_modules.append(mod)
        early_config.pluginmanager.consider_conftest(mod)


def getconftestmodules(*args, **kwargs):
    return conftest_modules


def pytest_sessionstart(session):
    # Override filesystem based relevant conftest discovery on the call path
    assert session.config.pluginmanager
    session.config.pluginmanager._getconftestmodules = getconftestmodules
