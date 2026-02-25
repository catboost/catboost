import json
import os
import sys
import types
import logging

import py
import pytest
import _pytest.doctest
import six
from six import reraise

if six.PY3:
    import pathlib

from library.python.pytest import module_utils
import library.python.testing.filter.filter as test_filter


def _make_module_with_single_skipped_test(module_name, file_name, exc):
    if exc.msg is not None:
        mark = pytest.mark.skip(reason=exc.msg)
    else:
        mark = pytest.mark.skip()

    module = types.ModuleType(module_name)
    module.__file__ = file_name

    @mark
    def module_level_skip():
        pass

    module_level_skip.__test__ = True
    setattr(module, 'pytest.skip', module_level_skip)
    return module


class LoadedModule(pytest.Module):
    def __init__(self, parent, name, namespace=True, **kwargs):
        self.module_name = name
        if namespace:
            assert name.startswith('__tests__.')
            self.display_module_name = name[len('__tests__.'):]
        else:
            self.display_module_name = name
        self.is_fake_module = False
        # Always passed to `super().__init__` explicitly.
        kwargs.pop('nodeid', None)

        if os.getenv('CONFTEST_LOAD_POLICY') == 'LOCAL':
            nodeid = module_utils.get_proper_module_path(self.module_name)
            if nodeid is None:
                self.is_fake_module = True
                return

            name = nodeid
            if six.PY3:
                kwargs['path'] = kwargs.get('path') or pathlib.Path(nodeid)
            else:
                raw_module_path = module_utils.get_module_file_path(self.module_name)
                kwargs['fspath'] = kwargs.get('fspath') or py.path.local(raw_module_path)
        else:
            nodeid = self.display_module_name + '.py'
            name = nodeid
            if six.PY3:
                kwargs['path'] = kwargs.get('path') or pathlib.Path(py.path.local())
            else:
                kwargs['fspath'] = kwargs.get('fspath') or py.path.local()

        if six.PY3:
            super().__init__(parent=parent, name=name, nodeid=nodeid, **kwargs)
        else:
            super(LoadedModule, self).__init__(parent=parent, nodeid=nodeid, **kwargs)
            self.name = name

    @classmethod
    def from_parent(cls, **kwargs):
        return getattr(super(LoadedModule, cls), 'from_parent', cls)(**kwargs)

    def _getobj(self):
        # A simplified version of pytest `importtestmodule` that works with resfs.
        try:
            __import__(self.module_name)
        except pytest.skip.Exception as e:
            if not e.allow_module_level:
                raise RuntimeError("Using pytest.skip outside of a test will skip the entire module. If that's your intention, pass `allow_module_level=True`.")
            # DEVTOOLSSUPPORT-79470: reraising pytest.skip.Exception from here seems to break reporting in ya plugin.
            # Instead, pretend to be a module with a single skipped test.
            return _make_module_with_single_skipped_test(self.module_name, self.nodeid, e)
        except Exception as e:
            msg = 'Failed to load module "{}" and obtain list of tests due to an error'.format(self.display_module_name)
            logging.exception('%s: %s', msg, e)
            etype, exc, tb = sys.exc_info()
            reraise(etype, type(exc)('{}\n{}'.format(exc, msg)), tb)

        return sys.modules[self.module_name]


class DoctestModule(LoadedModule):

    def collect(self):
        import doctest

        module = self._getobj()
        # uses internal doctest module parsing mechanism
        finder = doctest.DocTestFinder()
        if six.PY3:
            optionflags = _pytest.doctest.get_optionflags(self.config)
        else:
            optionflags = _pytest.doctest.get_optionflags(self)
        runner = doctest.DebugRunner(verbose=0, optionflags=optionflags)

        try:
            for test in finder.find(module, self.display_module_name):
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
# This function is only used in CollectionPlugin below, this is not an implementation of a pytest hook.
def _pytest_ignore_collect(module, config, filenames_from_full_filters, accept_filename_predicate):
    if module.is_fake_module:
        return True

    # TODO refactor to use meaningful attributes like `path` instead?
    #  test_file_filter would need to be relative to repository root.
    legacy_name = module.display_module_name + '.py'

    if config.option.mode == 'list':
        return not accept_filename_predicate(legacy_name)

    if filenames_from_full_filters is not None and legacy_name not in filenames_from_full_filters:
        return True

    test_file_filter = getattr(config.option, 'test_file_filter', None)
    if test_file_filter and legacy_name != test_file_filter.replace('/', '.'):
        return True

    return False


def _patch_set_initial_conftests(pluginmanager):
    """
    Avoids attempts to access filesystem directly in built-in `pytest_load_initial_conftests` hook impl.
    """
    if six.PY3:
        def _set_initial_conftests_py3(pyargs, noconftest, *args, **kwargs):
            pluginmanager._noconftest = noconftest
            pluginmanager._using_pyargs = pyargs

        pluginmanager._set_initial_conftests = _set_initial_conftests_py3
    else:
        def _set_initial_conftests_py2(namespace):
            pluginmanager._noconftest = namespace.noconftest
            pluginmanager._using_pyargs = namespace.pyargs

        pluginmanager._set_initial_conftests = _set_initial_conftests_py2


class CollectionPlugin(object):
    def __init__(self, test_modules, doctest_modules):
        self._test_modules = test_modules
        self._doctest_modules = doctest_modules

    @pytest.hookimpl(tryfirst=True)
    def pytest_load_initial_conftests(self, early_config):
        _patch_set_initial_conftests(early_config.pluginmanager)

    def pytest_sessionstart(self, session):

        def collect(*args, **kwargs):
            config = session.config

            if os.getenv('CONFTEST_LOAD_POLICY') == 'LOCAL':
                # A custom function that is set in conftests.py.
                config._ya_register_non_initial_conftests()

            accept_filename_predicate = test_filter.make_py_file_filter(config.option.test_filter)
            full_test_names_file_path = config.option.test_list_path
            filenames_filter = None

            if full_test_names_file_path and os.path.exists(full_test_names_file_path):
                with open(full_test_names_file_path, 'r') as afile:
                    # in afile stored 2 dimensional array such that array[modulo_index] contains tests which should be run in this test suite
                    full_names_filter = set(json.load(afile)[int(config.option.modulo_index)])
                    filenames_filter = set(map(lambda x: x.split('::')[0], full_names_filter))

            for test_module in self._test_modules:
                module = LoadedModule.from_parent(name=test_module, parent=session)
                if not _pytest_ignore_collect(module, config, filenames_filter, accept_filename_predicate):
                    yield module

                if os.environ.get('YA_PYTEST_DISABLE_DOCTEST', 'no') == 'no':
                    module = DoctestModule.from_parent(name=test_module, parent=session)
                    if not _pytest_ignore_collect(module, config, filenames_filter, accept_filename_predicate):
                        yield module

            if os.environ.get('YA_PYTEST_DISABLE_DOCTEST', 'no') == 'no':
                for doctest_module in self._doctest_modules:
                    yield DoctestModule.from_parent(name=doctest_module, parent=session, namespace=False)

        session.collect = collect
