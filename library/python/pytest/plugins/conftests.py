import os
import importlib
import sys
import inspect

import yatest.common as yc

from pytest import hookimpl
from yatest_lib.ya import Ya

from library.python.pytest.plugins.fixtures import metrics, links  # noqa

orig_getfile = inspect.getfile


def getfile(object):
    if inspect.ismodule(object) and getattr(object, '__orig_file__', None):
        res = object.__orig_file__
    else:
        res = orig_getfile(object)
    return res


inspect.getfile = getfile
conftest_modules = []


@hookimpl(trylast=True)
def pytest_load_initial_conftests(early_config, parser, args):
    yc.runtime._set_ya_config(ya=Ya())

    if hasattr(sys, 'extra_modules'):
        conftests = filter(lambda name: name.endswith(".conftest"), sys.extra_modules)
    else:
        conftests = []

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
