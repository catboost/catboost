import sys

import py

import pytest  # noqa
import _pytest.python
import _pytest.doctest


class LoadedModule(_pytest.python.Module):

    def __init__(self, name, session, namespace=True):
        self.name = name + ".py"  # pytest requires names to be ended with .py
        self.session = session
        self.config = session.config
        self.fspath = py.path.local()
        self.parent = session
        self.keywords = {}
        self.own_markers = []

        self.namespace = namespace

    @property
    def nodeid(self):
        return self.name

    def _getobj(self):
        module_name = self.name[:-(len(".py"))]
        if self.namespace:
            module_name = "__tests__." + module_name
        __import__(module_name)
        return sys.modules[module_name]


class DoctestModule(LoadedModule):

    def collect(self):
        import doctest

        module = self._getobj()
        # uses internal doctest module parsing mechanism
        finder = doctest.DocTestFinder()
        optionflags = _pytest.doctest.get_optionflags(self)
        runner = doctest.DebugRunner(verbose=0, optionflags=optionflags)

        for test in finder.find(module, self.name[:-(len(".py"))]):
            if test.examples:  # skip empty doctests
                yield _pytest.doctest.DoctestItem(test.name, self, runner, test)


class CollectionPlugin(object):
    def __init__(self, test_modules, doctest_modules):
        self._test_modules = test_modules
        self._doctest_modules = doctest_modules

    def pytest_sessionstart(self, session):

        def collect(*args, **kwargs):
            for test_module in self._test_modules:
                yield LoadedModule(test_module, session=session)
                yield DoctestModule(test_module, session=session)

            for doctest_module in self._doctest_modules:
                yield DoctestModule(doctest_module, session=session, namespace=False)

        session.collect = collect
