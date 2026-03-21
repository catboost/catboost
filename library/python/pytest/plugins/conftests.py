import collections
import functools
import importlib
import inspect
import os
import six
import sys

import pytest

from library.python.pytest import module_utils
import yatest.common

ConftestInfo = collections.namedtuple('ConftestInfo', ['module_name', 'proper_path'])
LoadedConftestInfo = collections.namedtuple('LoadedConftestInfo', ['conftest_dir', 'module'])


def _patch_inspect_getfile():
    orig_getfile = inspect.getfile

    def getfile(object):
        if inspect.ismodule(object) and getattr(object, '__orig_file__', None):
            res = object.__orig_file__
        else:
            res = orig_getfile(object)
        return res

    inspect.getfile = getfile


def _consider_conftest(pluginmanager, module, registration_name):
    if six.PY3:
        pluginmanager.consider_conftest(module, registration_name=registration_name)
    else:
        pluginmanager.consider_conftest(module)


def _import_conftest(pluginmanager, module_name, proper_path):
    """Essentially PytestPluginManager._importconftest, adapted to resource-based conftest modules."""
    module = importlib.import_module(module_name)

    if six.PY3:
        pluginmanager._check_non_top_pytest_plugins(module, proper_path)

    _consider_conftest(pluginmanager, module, proper_path)
    conftest_info = LoadedConftestInfo(conftest_dir=os.path.dirname(proper_path), module=module)
    pluginmanager._ya_loaded_conftest_infos.append(conftest_info)


def _import_conftest_force_initial(pluginmanager, module_name):
    """Sets conftest filename to '', which makes pytest consider it an initial conftest."""
    module = importlib.import_module(module_name)

    if six.PY2:
        setattr(module, '__orig_file__', module.__file__)
        module.__file__ = ''

    _consider_conftest(pluginmanager, module, '')
    conftest_info = LoadedConftestInfo(conftest_dir='', module=module)
    pluginmanager._ya_loaded_conftest_infos.append(conftest_info)


def _getconftestmodules(self, path):
    """
    _getconftestmodules returns a cached list of conftest modules that relate to the given test path.
    This list is then used by pytest to temporarily mute other non-initial conftests when processing those tests.
    We patch pytest's implementation to avoid fs access when discovering conftests and in is_file() checks.
    """
    path = str(path)
    if path.endswith('.py'):
        path = os.path.dirname(path)

    cache = self._ya_getconftestmodules_cache
    if path not in cache:
        cache[path] = [
            conftest_info.module
            for conftest_info in self._ya_loaded_conftest_infos
            if module_utils.is_relative_to(path, conftest_info.conftest_dir)
        ]

    return cache[path]


def _non_local_conftest_key(name):
    if not name.startswith('__tests__.'):
        # Make __tests__ come last
        return '_.' + name
    return name


@pytest.hookimpl(trylast=True)
def pytest_load_initial_conftests(early_config):
    pluginmanager = early_config.pluginmanager

    pluginmanager._ya_loaded_conftest_infos = []
    pluginmanager._ya_getconftestmodules_cache = {}
    pluginmanager._getconftestmodules = functools.partial(_getconftestmodules, pluginmanager)

    extra_modules = getattr(sys, 'extra_modules', [])
    conftests = filter(lambda name: name == 'conftest' or name.endswith('.conftest'), extra_modules)

    # CONFTEST_LOAD_POLICY is set by ya.make macros:
    # * CONFTEST_LOAD_POLICY_LOCAL()
    # * CONFTEST_LOAD_POLICY_LEGACY_GLOBAL()
    # Current default is LEGACY_GLOBAL.
    if os.getenv('CONFTEST_LOAD_POLICY') == 'LOCAL':
        test_dir = str(yatest.common.context.project_path)

        initial_conftests = []
        non_initial_conftests = []

        for module_name in conftests:
            proper_path = module_utils.get_proper_module_path(module_name)
            assert proper_path is not None
            conftest_dir = os.path.dirname(proper_path)
            conftest_info = ConftestInfo(module_name=module_name, proper_path=proper_path)

            # There are two kinds of relevant conftests:
            # - Initial: conftest is in the directory of the test suite or in a parent directory (applies to all tests)
            # - Non-initial: conftest is strictly within the test directory
            if module_utils.is_relative_to(test_dir, conftest_dir):
                initial_conftests.append(conftest_info)
            elif module_utils.is_relative_to(conftest_dir, test_dir):
                non_initial_conftests.append(conftest_info)

        # Ensure parent conftests are loaded before child conftests
        initial_conftests.sort(key=lambda x: len(x.proper_path))
        non_initial_conftests.sort(key=lambda x: len(x.proper_path))

        for conftest_info in initial_conftests:
            _import_conftest(pluginmanager, conftest_info.module_name, conftest_info.proper_path)

        def _ya_register_non_initial_conftests():
            for conftest_info in non_initial_conftests:
                _import_conftest(pluginmanager, conftest_info.module_name, conftest_info.proper_path)

    else:
        if six.PY2:
            _patch_inspect_getfile()

        for name in sorted(conftests, key=_non_local_conftest_key):
            _import_conftest_force_initial(pluginmanager, name)

        def _ya_register_non_initial_conftests():
            pass

    # See collection.py, non-initial conftests are registered at the start of collection phase, like in vanilla.
    # Alas, stash is not available in py2.
    early_config._ya_register_non_initial_conftests = _ya_register_non_initial_conftests
