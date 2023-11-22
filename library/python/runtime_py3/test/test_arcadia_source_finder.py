import stat
import unittest
import yaml
from collections import namedtuple
from unittest.mock import patch
from parameterized import parameterized

import __res as res


NAMESPACE_PREFIX = b"py/namespace/"
TEST_SOURCE_ROOT = "/home/arcadia"


class ImporterMocks:
    def __init__(self, mock_fs, mock_resources):
        self._mock_fs = mock_fs
        self._mock_resources = mock_resources
        self._patchers = [
            patch("__res.iter_keys", wraps=self._iter_keys),
            patch("__res.__resource.find", wraps=self._resource_find),
            patch("__res._path_isfile", wraps=self._path_isfile),
            patch("__res._os.listdir", wraps=self._os_listdir),
            patch("__res._os.lstat", wraps=self._os_lstat),
        ]
        for patcher in self._patchers:
            patcher.start()

    def stop(self):
        for patcher in self._patchers:
            patcher.stop()

    def _iter_keys(self, prefix):
        assert prefix == NAMESPACE_PREFIX
        for k in self._mock_resources.keys():
            yield k, k.removeprefix(prefix)

    def _resource_find(self, key):
        return self._mock_resources.get(key)

    def _lookup_mock_fs(self, filename):
        path = filename.lstrip("/").split("/")
        curdir = self._mock_fs
        for item in path:
            if item in curdir:
                curdir = curdir[item]
            else:
                return None
        return curdir

    def _path_isfile(self, filename):
        f = self._lookup_mock_fs(filename)
        return isinstance(f, str)

    def _os_lstat(self, filename):
        f = self._lookup_mock_fs(filename)
        mode = stat.S_IFDIR if isinstance(f, dict) else stat.S_IFREG
        return namedtuple("fake_stat_type", "st_mode")(st_mode=mode)

    def _os_listdir(self, dirname):
        f = self._lookup_mock_fs(dirname)
        if isinstance(f, dict):
            return f.keys()
        else:
            return []


class ArcadiaSourceFinderTestCase(unittest.TestCase):
    def setUp(self):
        self.import_mock = ImporterMocks(yaml.safe_load(self._get_mock_fs()), self._get_mock_resources())
        self.arcadia_source_finder = res.ArcadiaSourceFinder(TEST_SOURCE_ROOT)

    def tearDown(self):
        self.import_mock.stop()

    def _get_mock_fs(self):
        raise NotImplementedError()

    def _get_mock_resources(self):
        raise NotImplementedError()


class TestLibraryWithoutNamespace(ArcadiaSourceFinderTestCase):
    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               project:
                 lib:
                   mod1.py: ""
                   package1:
                     mod2.py: ""
        """

    def _get_mock_resources(self):
        return {
            b"py/namespace/unique_prefix1/project/lib": b"project.lib.",
        }

    @parameterized.expand(
        [
            ("project.lib.mod1", b"project/lib/mod1.py"),
            ("project.lib.package1.mod2", b"project/lib/package1/mod2.py"),
            ("project.lib.unknown_module", None),
            ("project.lib", None),  # package
        ]
    )
    def test_get_module_path(self, module, path):
        assert path == self.arcadia_source_finder.get_module_path(module)

    @parameterized.expand(
        [
            ("project.lib.mod1", False),
            ("project.lib.package1.mod2", False),
            ("project", True),
            ("project.lib", True),
            ("project.lib.package1", True),
        ]
    )
    def test_is_packages(self, module, is_package):
        assert is_package == self.arcadia_source_finder.is_package(module)

    def test_is_package_for_unknown_module(self):
        self.assertRaises(
            ImportError,
            lambda: self.arcadia_source_finder.is_package("project.lib.package2"),
        )

    @parameterized.expand(
        [
            (
                "",
                {
                    ("PFX.project", True),
                },
            ),
            (
                "project.",
                {
                    ("PFX.lib", True),
                },
            ),
            (
                "project.lib.",
                {
                    ("PFX.mod1", False),
                    ("PFX.package1", True),
                },
            ),
            (
                "project.lib.package1.",
                {
                    ("PFX.mod2", False),
                },
            ),
        ]
    )
    def test_iter_modules(self, package_prefix, expected):
        got = self.arcadia_source_finder.iter_modules(package_prefix, "PFX.")
        assert expected == set(got)

    # Check iter_modules() don't crash and return correct result after not existing module was requested
    def test_iter_modules_after_unknown_module_import(self):
        self.arcadia_source_finder.get_module_path("project.unknown_module")
        assert {("lib", True)} == set(self.arcadia_source_finder.iter_modules("project.", ""))


class TestLibraryExtendedFromAnotherLibrary(ArcadiaSourceFinderTestCase):
    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               project:
                 lib:
                   mod1.py: ''
                 lib_extension:
                   mod2.py: ''
        """

    def _get_mock_resources(self):
        return {
            b"py/namespace/unique_prefix1/project/lib": b"project.lib.",
            b"py/namespace/unique_prefix2/project/lib_extension": b"project.lib.",
        }

    @parameterized.expand(
        [
            ("project.lib.mod1", b"project/lib/mod1.py"),
            ("project.lib.mod2", b"project/lib_extension/mod2.py"),
        ]
    )
    def test_get_module_path(self, module, path):
        assert path == self.arcadia_source_finder.get_module_path(module)

    @parameterized.expand(
        [
            (
                "project.lib.",
                {
                    ("PFX.mod1", False),
                    ("PFX.mod2", False),
                },
            ),
        ]
    )
    def test_iter_modules(self, package_prefix, expected):
        got = self.arcadia_source_finder.iter_modules(package_prefix, "PFX.")
        assert expected == set(got)


class TestNamespaceAndTopLevelLibraries(ArcadiaSourceFinderTestCase):
    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               project:
                 ns_lib:
                   mod1.py: ''
                 top_level_lib:
                   mod2.py: ''
        """

    def _get_mock_resources(self):
        return {
            b"py/namespace/unique_prefix1/project/ns_lib": b"ns.",
            b"py/namespace/unique_prefix2/project/top_level_lib": b".",
        }

    @parameterized.expand(
        [
            ("ns.mod1", b"project/ns_lib/mod1.py"),
            ("mod2", b"project/top_level_lib/mod2.py"),
        ]
    )
    def test_get_module_path(self, module, path):
        assert path == self.arcadia_source_finder.get_module_path(module)

    @parameterized.expand(
        [
            ("ns", True),
            ("ns.mod1", False),
            ("mod2", False),
        ]
    )
    def test_is_packages(self, module, is_package):
        assert is_package == self.arcadia_source_finder.is_package(module)

    @parameterized.expand(
        [
            "project",
            "project.ns_lib",
            "project.top_level_lib",
        ]
    )
    def test_is_package_for_unknown_modules(self, module):
        self.assertRaises(ImportError, lambda: self.arcadia_source_finder.is_package(module))

    @parameterized.expand(
        [
            (
                "",
                {
                    ("PFX.ns", True),
                    ("PFX.mod2", False),
                },
            ),
            (
                "ns.",
                {
                    ("PFX.mod1", False),
                },
            ),
        ]
    )
    def test_iter_modules(self, package_prefix, expected):
        got = self.arcadia_source_finder.iter_modules(package_prefix, "PFX.")
        assert expected == set(got)


class TestIgnoreDirectoriesWithYaMakeFile(ArcadiaSourceFinderTestCase):
    """Packages and modules from tests should not be part of pylib namespace"""

    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               contrib:
                 python:
                   pylib:
                     mod1.py: ""
                     tests:
                       conftest.py: ""
                       ya.make: ""
        """

    def _get_mock_resources(self):
        return {
            b"py/namespace/unique_prefix1/contrib/python/pylib": b"pylib.",
        }

    def test_get_module_path_for_lib(self):
        assert b"contrib/python/pylib/mod1.py" == self.arcadia_source_finder.get_module_path("pylib.mod1")

    def test_get_module_for_tests(self):
        assert self.arcadia_source_finder.get_module_path("pylib.tests.conftest") is None

    def test_is_package_for_tests(self):
        self.assertRaises(ImportError, lambda: self.arcadia_source_finder.is_package("pylib.tests"))


class TestMergingNamespaceAndDirectoryPackages(ArcadiaSourceFinderTestCase):
    """Merge parent package (top level in this test) dirs with namespace dirs (DEVTOOLS-8979)"""

    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               contrib:
                 python:
                   pylint:
                     ya.make: ""
                     pylint:
                       __init__.py: ""
                     patcher:
                       patch.py: ""
                       ya.make: ""
        """

    def _get_mock_resources(self):
        return {
            b"py/namespace/unique_prefix1/contrib/python/pylint": b".",
            b"py/namespace/unique_prefix1/contrib/python/pylint/patcher": b"pylint.",
        }

    @parameterized.expand(
        [
            ("pylint.__init__", b"contrib/python/pylint/pylint/__init__.py"),
            ("pylint.patch", b"contrib/python/pylint/patcher/patch.py"),
        ]
    )
    def test_get_module_path(self, module, path):
        assert path == self.arcadia_source_finder.get_module_path(module)


class TestEmptyResources(ArcadiaSourceFinderTestCase):
    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               project:
                 lib:
                   mod1.py: ''
        """

    def _get_mock_resources(self):
        return {}

    def test_get_module_path(self):
        assert self.arcadia_source_finder.get_module_path("project.lib.mod1") is None

    def test_is_package(self):
        self.assertRaises(ImportError, lambda: self.arcadia_source_finder.is_package("project"))

    def test_iter_modules(self):
        assert [] == list(self.arcadia_source_finder.iter_modules("", "PFX."))


class TestDictionaryChangedSizeDuringIteration(ArcadiaSourceFinderTestCase):
    def _get_mock_fs(self):
        return """
           home:
             arcadia:
               project:
                 lib1:
                   mod1.py: ''
                 lib2:
                   mod2.py: ''
        """

    def _get_mock_resources(self):
        return {
            b"py/namespace/unique_prefix1/project/lib1": b"project.lib1.",
            b"py/namespace/unique_prefix1/project/lib2": b"project.lib2.",
        }

    def test_no_crash_on_recursive_iter_modules(self):
        for package in self.arcadia_source_finder.iter_modules("project.", ""):
            for _ in self.arcadia_source_finder.iter_modules(package[0], ""):
                pass
