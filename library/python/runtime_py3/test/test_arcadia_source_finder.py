import unittest
from unittest.mock import patch
from parameterized import parameterized

import __res as res


NAMESPACE_PREFIX = b'py/namespace/'
TEST_SOURCE_ROOT = '/home/arcadia'
TEST_FS = {
    'home': {
        'arcadia': {
            'project': {
                'normal_lib': {
                    'mod1.py': '',
                    'package1': {
                        'mod2.py': '',
                    },
                },
                'lib_with_namespace': {
                    'ns_mod1.py': '',
                    'ns_package1': {
                        'ns_mod2.py': '',
                    },
                },
                'top_level_lib': {
                    'tl_mod1.py': '',
                    'tl_package1': {
                        'tl_mod2.py': '',
                    },
                },
                'normal_lib_extension': {
                    'mod3.py': '',
                    'package1': {
                        'mod4.py': '',
                    },
                },
            },
            'contrib': {
                'python': {
                    'pylib': {
                        'libmod.py': '',
                        'tests': {
                            'conftest.py': '',
                            'ya.make': '',
                        },
                    },
                },
            },
        },
    },
}
TEST_RESOURCE = {
    b'py/namespace/unique_prefix1/project/normal_lib': b'project.normal_lib.',
    # 'normal_lib_extension' extend normal_lib by additional modules
    b'py/namespace/unique_prefix1/project/normal_lib_extension': b'project.normal_lib.',
    b'py/namespace/unique_prefix2/project/lib_with_namespace': b'virtual.namespace.',
    b'py/namespace/unique_prefix3/project/top_level_lib': b'.',
    # Contrib: the library is in the top level namespace but 'tests' project is not
    b'py/namespace/unique_prefix4/contrib/python/pylib': b'.',
    b'py/namespace/unique_prefix4/contrib/python/pylib/tests': b'contrib.python.pylib.tests.',
}
MODULES = {
    'project.normal_lib.mod1': b'project/normal_lib/mod1.py',
    'project.normal_lib.mod3': b'project/normal_lib_extension/mod3.py',
    'project.normal_lib.package1.mod2': b'project/normal_lib/package1/mod2.py',
    'project.normal_lib.package1.mod4': b'project/normal_lib_extension/package1/mod4.py',
    'virtual.namespace.ns_mod1': b'project/lib_with_namespace/ns_mod1.py',
    'virtual.namespace.ns_package1.ns_mod2': b'project/lib_with_namespace/ns_package1/ns_mod2.py',
    'tl_mod1': b'project/top_level_lib/tl_mod1.py',
    'tl_package1.tl_mod2': b'project/top_level_lib/tl_package1/tl_mod2.py',
    'libmod': b'contrib/python/pylib/libmod.py',
    'contrib.python.pylib.tests.conftest': b'contrib/python/pylib/tests/conftest.py',
}
PACKAGES = [
    'project',
    'project.normal_lib',
    'project.normal_lib.package1',
    'virtual',
    'virtual.namespace',
    'virtual.namespace.ns_package1',
    'tl_package1',
    'contrib',
    'contrib.python',
    'contrib.python.pylib',
    'contrib.python.pylib.tests',
]
UNKNOWN_MODULES = [
    'project.normal_lib.unknown_module',
    'virtual.namespace.unknown_module',
    'unknown_module',
    # contribr/python/pylib directory is not a regular package and cannot be used for a usual module lookup
    'contrib.python.pylib.libmod',
    # Parent project contrib/python/pylib with top level namespace should not affect nested 'tests' project
    'tests.conftest',
]


def iter_keys_mock(prefix):
    assert prefix == NAMESPACE_PREFIX
    l = len(prefix)
    for k in TEST_RESOURCE.keys():
        yield k, k[l:]


def resource_find_mock(key):
    return TEST_RESOURCE.get(key)


def find_fake_fs(filename):
    path = filename.lstrip('/').split('/')
    curdir = TEST_FS
    for item in path:
        if item in curdir:
            curdir = curdir[item]
        else:
            return None
    return curdir


def path_isfile_mock(filename):
    f = find_fake_fs(filename)
    return isinstance(f, str)


def path_isdir_mock(filename):
    f = find_fake_fs(filename)
    return isinstance(f, dict)


def os_listdir_mock(dirname):
    f = find_fake_fs(dirname)
    if isinstance(f, dict):
        return f.keys()
    else:
        return []


class TestArcadiaSourceFinder(unittest.TestCase):
    def setUp(self):
        self.patchers = [
            patch('__res.iter_keys', wraps=iter_keys_mock),
            patch('__res.__resource.find', wraps=resource_find_mock),
            patch('__res._path_isdir', wraps=path_isdir_mock),
            patch('__res._path_isfile', wraps=path_isfile_mock),
            patch('__res._os.listdir', wraps=os_listdir_mock),
        ]
        for patcher in self.patchers:
            patcher.start()
        self.arcadia_source_finder = res.ArcadiaSourceFinder(TEST_SOURCE_ROOT)

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    @parameterized.expand(MODULES.items())
    def test_get_module_path_for_modules(self, module, path):
        assert path == self.arcadia_source_finder.get_module_path(module)

    @parameterized.expand(PACKAGES)
    def test_get_module_path_for_packages(self, package):
        assert self.arcadia_source_finder.get_module_path(package) is None

    @parameterized.expand(UNKNOWN_MODULES)
    def test_get_module_path_for_unknown_modules(self, unknown_module):
        assert self.arcadia_source_finder.get_module_path(unknown_module) is None

    @parameterized.expand(MODULES.keys())
    def test_is_package_for_modules(self, module):
        assert self.arcadia_source_finder.is_package(module) is False

    @parameterized.expand(PACKAGES)
    def test_is_package_for_packages(self, package):
        assert self.arcadia_source_finder.is_package(package) is True

    @parameterized.expand(UNKNOWN_MODULES)
    def test_is_package_for_unknown_modules(self, unknown_module):
        self.assertRaises(ImportError, lambda: self.arcadia_source_finder.is_package(unknown_module))

    @parameterized.expand([
        ('project.', {
            ('PFX.normal_lib', True),
        }),
        ('project.normal_lib.', {
            ('PFX.mod1', False),
            ('PFX.mod3', False),
            ('PFX.package1', True),
        }),
        ('project.normal_lib.package1.', {
            ('PFX.mod2', False),
            ('PFX.mod4', False),
        }),
        ('virtual.', {
            ('PFX.namespace', True),
        }),
        ('virtual.namespace.', {
            ('PFX.ns_mod1', False),
            ('PFX.ns_package1', True),
        }),
        ('virtual.namespace.ns_package1.', {
            ('PFX.ns_mod2', False),
        }),
        ('', {
            ('PFX.project', True),
            ('PFX.virtual', True),
            ('PFX.tl_mod1', False),
            ('PFX.tl_package1', True),
            ('PFX.contrib', True),
            ('PFX.libmod', False),
        }),
        ('tl_package1.', {
            ('PFX.tl_mod2', False),
        }),
        ('contrib.python.pylib.', {
            ('PFX.tests', True),
        }),
        ('contrib.python.pylib.tests.', {
            ('PFX.conftest', False),
        }),
    ])
    def test_iter_modules(self, package_prefix, expected):
        got = self.arcadia_source_finder.iter_modules(package_prefix, 'PFX.')
        assert expected == set(got)

    # Check iter_modules() don't crash and return correct result after not existing module was requested
    def test_iter_modules_after_unknown_module_import(self):
        self.arcadia_source_finder.get_module_path('project.unknown_module')
        assert {('normal_lib', True)} == set(self.arcadia_source_finder.iter_modules('project.', ''))


class TestArcadiaSourceFinderForEmptyResources(unittest.TestCase):
    @staticmethod
    def _unreachable():
        raise Exception()

    def setUp(self):
        self.patchers = [
            patch('__res.iter_keys', wraps=lambda x: []),
            patch('__res.__resource.find', wraps=self._unreachable),
            patch('__res._path_isdir', wraps=self._unreachable),
            patch('__res._path_isfile', wraps=self._unreachable),
            patch('__res._os.listdir', wraps=self._unreachable),
        ]
        for patcher in self.patchers:
            patcher.start()
        self.arcadia_source_finder = res.ArcadiaSourceFinder(TEST_SOURCE_ROOT)

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    def test_get_module_path(self):
        assert self.arcadia_source_finder.get_module_path('project.normal_lib.mod1') is None

    def test_is_package(self):
        self.assertRaises(ImportError, lambda: self.arcadia_source_finder.is_package('project'))
        self.assertRaises(ImportError, lambda: self.arcadia_source_finder.is_package('project.normal_lib.mod1'))

    def test_iter_modules(self):
        assert [] == list(self.arcadia_source_finder.iter_modules('', 'PFX.'))
