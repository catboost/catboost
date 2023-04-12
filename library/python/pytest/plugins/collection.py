import os
import sys
from six import reraise
import logging

import py

import pytest  # noqa
import _pytest.python
import _pytest.doctest
import json
import library.python.testing.filter.filter as test_filter


class LoadedModule(_pytest.python.Module):
    def __init__(self, parent, name, **kwargs):
        self.name = name + '.py'
        self.session = parent
        self.parent = parent
        self.config = parent.config
        self.keywords = {}
        self.own_markers = []
        self.extra_keyword_matches = set()
        self.fspath = py.path.local()

    @classmethod
    def from_parent(cls, **kwargs):
        namespace = kwargs.pop('namespace', True)
        kwargs.setdefault('fspath', py.path.local())

        loaded_module = getattr(super(LoadedModule, cls), 'from_parent', cls)(**kwargs)
        loaded_module.namespace = namespace

        return loaded_module

    @property
    def _nodeid(self):
        if os.getenv('CONFTEST_LOAD_POLICY') == 'LOCAL':
            return self._getobj().__file__
        else:
            return self.name

    @property
    def nodeid(self):
        return self._nodeid

    def _getobj(self):
        module_name = self.name[:-len('.py')]
        if self.namespace:
            module_name = '__tests__.' + module_name
        try:
            __import__(module_name)
        except Exception as e:
            msg = 'Failed to load module "{}" and obtain list of tests due to an error'.format(module_name)
            logging.exception('%s: %s', msg, e)
            etype, exc, tb = sys.exc_info()
            reraise(etype, type(exc)('{}\n{}'.format(exc, msg)), tb)

        return sys.modules[module_name]


class DoctestModule(LoadedModule):

    def collect(self):
        import doctest

        module = self._getobj()
        # uses internal doctest module parsing mechanism
        finder = doctest.DocTestFinder()
        optionflags = _pytest.doctest.get_optionflags(self)
        runner = doctest.DebugRunner(verbose=0, optionflags=optionflags)

        try:
            for test in finder.find(module, self.name[:-len('.py')]):
                if test.examples:  # skip empty doctests
                    yield getattr(_pytest.doctest.DoctestItem, 'from_parent', _pytest.doctest.DoctestItem)(
                        name=test.name,
                        parent=self,
                        runner=runner,
                        dtest=test)
        except Exception:
            logging.exception('DoctestModule failed, probably you can add NO_DOCTESTS() macro to ya.make')
            etype, exc, tb = sys.exc_info()
            msg = 'DoctestModule failed, probably you can add NO_DOCTESTS() macro to ya.make'
            reraise(etype, type(exc)('{}\n{}'.format(exc, msg)), tb)


# NOTE: Since we are overriding collect method of pytest session, pytest hooks are not invoked during collection.
def pytest_ignore_collect(module, session, filenames_from_full_filters, accept_filename_predicate):
    if session.config.option.mode == 'list':
        return not accept_filename_predicate(module.name)

    if filenames_from_full_filters is not None and module.name not in filenames_from_full_filters:
        return True

    test_file_filter = getattr(session.config.option, 'test_file_filter', None)
    if test_file_filter is None:
        return False
    if module.name != test_file_filter.replace('/', '.'):
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
                module = LoadedModule.from_parent(name=test_module, parent=session)
                if not pytest_ignore_collect(module, session, filenames_filter, accept_filename_predicate):
                    yield module

                if os.environ.get('YA_PYTEST_DISABLE_DOCTEST', 'no') == 'no':
                    module = DoctestModule.from_parent(name=test_module, parent=session)
                    if not pytest_ignore_collect(module, session, filenames_filter, accept_filename_predicate):
                        yield module

            if os.environ.get('YA_PYTEST_DISABLE_DOCTEST', 'no') == 'no':
                for doctest_module in self._doctest_modules:
                    yield DoctestModule.from_parent(name=doctest_module, parent=session, namespace=False)

        session.collect = collect
