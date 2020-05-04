import os
import sys

import py

import pytest  # noqa
import _pytest.python
import _pytest.doctest
import json
import library.python.testing.filter.filter as test_filter


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
        if os.getenv("CONFTEST_LOAD_POLICY") == "LOCAL":
            return self._getobj().__file__
        else:
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


# NOTE: Since we are overriding collect method of pytest session, pytest hooks are not invoked during collection.
def pytest_ignore_collect(module, session, filenames_from_full_filters, accept_filename_predicate):
    if session.config.option.mode == 'list':
        return not accept_filename_predicate(module.name)

    if filenames_from_full_filters is not None and module.name not in filenames_from_full_filters:
        return True

    test_file_filter = getattr(session.config.option, 'test_file_filter', None)
    if test_file_filter is None:
        return False
    for filename in test_file_filter:  # in test_file_filter may be only one filename
        if module.name != filename.replace("/", "."):
            return True
    return False


class CollectionPlugin(object):
    def __init__(self, test_modules, doctest_modules):
        self._test_modules = test_modules
        self._doctest_modules = doctest_modules

    def pytest_sessionstart(self, session):

        def collect(*args, **kwargs):
            accept_filename_predicate = test_filter.make_py_file_filter(session.config.option.test_filter)
            full_test_names_file_path = session.config.option.test_list_path
            filenames_filter = None

            if full_test_names_file_path and os.path.exists(full_test_names_file_path):
                with open(full_test_names_file_path, 'r') as afile:
                    # in afile stored 2 dimensional array such that array[modulo_index] contains tests which should be run in this test suite
                    full_names_filter = set(json.load(afile)[int(session.config.option.modulo_index)])
                    filenames_filter = set(map(lambda x: x.split('::')[0], full_names_filter))

            for test_module in self._test_modules:
                module = LoadedModule(test_module, session=session)
                if not pytest_ignore_collect(module, session, filenames_filter, accept_filename_predicate):
                    yield module

                module = DoctestModule(test_module, session=session)
                if not pytest_ignore_collect(module, session, filenames_filter, accept_filename_predicate):
                    yield module

            for doctest_module in self._doctest_modules:
                yield DoctestModule(doctest_module, session=session, namespace=False)

        session.collect = collect
