"""Tests for scandir.scandir()."""

from __future__ import unicode_literals

import os
import shutil
import sys
import time
import unittest

import yatest.common

try:
    import scandir
    has_scandir = True
except ImportError:
    has_scandir = False

FILE_ATTRIBUTE_DIRECTORY = 16

IS_PY3 = sys.version_info >= (3, 0)

if IS_PY3:
    int_types = int
else:
    int_types = (int, long)
    str = unicode


if hasattr(os, 'symlink'):
    try:
        #link_name = os.path.join(os.path.dirname(__file__), '_testlink')
        #os.symlink(__file__, link_name)
        #os.remove(link_name)
        symlinks_supported = True
    except NotImplementedError:
        # Windows versions before Vista don't support symbolic links
        symlinks_supported = False
else:
    symlinks_supported = False


def create_file(path, contents='1234'):
    with open(path, 'w') as f:
        f.write(contents)


def setup_main():
    join = os.path.join

    os.mkdir(TEST_PATH)
    os.mkdir(join(TEST_PATH, 'subdir'))
    create_file(join(TEST_PATH, 'file1.txt'))
    create_file(join(TEST_PATH, 'file2.txt'), contents='12345678')

    os.mkdir(join(TEST_PATH, 'subdir', 'unidir\u018F'))
    create_file(join(TEST_PATH, 'subdir', 'file1.txt'))
    create_file(join(TEST_PATH, 'subdir', 'unicod\u018F.txt'))

    create_file(join(TEST_PATH, 'subdir', 'unidir\u018F', 'file1.txt'))

    os.mkdir(join(TEST_PATH, 'linkdir'))


def setup_symlinks():
    join = os.path.join

    os.mkdir(join(TEST_PATH, 'linkdir', 'linksubdir'))
    create_file(join(TEST_PATH, 'linkdir', 'file1.txt'))

    os.symlink(os.path.abspath(join(TEST_PATH, 'linkdir', 'file1.txt')),
               join(TEST_PATH, 'linkdir', 'link_to_file'))

    dir_name = os.path.abspath(join(TEST_PATH, 'linkdir', 'linksubdir'))
    dir_link = join(TEST_PATH, 'linkdir', 'link_to_dir')
    if IS_PY3:
        os.symlink(dir_name, dir_link, target_is_directory=True)
    else:
        os.symlink(dir_name, dir_link)


def teardown():
    try:
        shutil.rmtree(TEST_PATH)
    except OSError:
        # why does the above fail sometimes?
        time.sleep(0.1)
        shutil.rmtree(TEST_PATH)


class TestMixin(unittest.TestCase):
    def setUp(self):
        global TEST_PATH
        TEST_PATH = yatest.common.test_output_path('../test')
        if not os.path.exists(TEST_PATH):
            setup_main()
        if symlinks_supported and not os.path.exists(
                os.path.join(TEST_PATH, 'linkdir', 'linksubdir')):
            setup_symlinks()

    if not hasattr(unittest.TestCase, 'skipTest'):
        def skipTest(self, reason):
            sys.stdout.write('skipped {0!r} '.format(reason))

    def test_basic(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        entries = sorted(self.scandir_func(TEST_PATH), key=lambda e: e.name)
        self.assertEqual([(e.name, e.is_dir()) for e in entries],
                         [('file1.txt', False), ('file2.txt', False),
                          ('linkdir', True), ('subdir', True)])
        self.assertEqual([e.path for e in entries],
                         [os.path.join(TEST_PATH, e.name) for e in entries])

    def test_dir_entry(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        entries = dict((e.name, e) for e in self.scandir_func(TEST_PATH))
        e = entries['file1.txt']
        self.assertEqual([e.is_dir(), e.is_file(), e.is_symlink()], [False, True, False])
        e = entries['file2.txt']
        self.assertEqual([e.is_dir(), e.is_file(), e.is_symlink()], [False, True, False])
        e = entries['subdir']
        self.assertEqual([e.is_dir(), e.is_file(), e.is_symlink()], [True, False, False])

        self.assertEqual(entries['file1.txt'].stat().st_size, 4)
        self.assertEqual(entries['file2.txt'].stat().st_size, 8)

    def test_stat(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        entries = list(self.scandir_func(TEST_PATH))
        for entry in entries:
            os_stat = os.stat(os.path.join(TEST_PATH, entry.name))
            scandir_stat = entry.stat()
            self.assertEqual(os_stat.st_mode, scandir_stat.st_mode)
            # TODO: be nice to figure out why these aren't identical on Windows and on PyPy
            # * Windows: they seem to be a few microseconds to tens of seconds out
            # * PyPy: for some reason os_stat's times are nanosecond, scandir's are not
            self.assertAlmostEqual(os_stat.st_mtime, scandir_stat.st_mtime, delta=1)
            self.assertAlmostEqual(os_stat.st_ctime, scandir_stat.st_ctime, delta=1)
            if entry.is_file():
                self.assertEqual(os_stat.st_size, scandir_stat.st_size)

    def test_returns_iter(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        it = self.scandir_func(TEST_PATH)
        entry = next(it)
        assert hasattr(entry, 'name')

    def check_file_attributes(self, result):
        self.assertTrue(hasattr(result, 'st_file_attributes'))
        self.assertTrue(isinstance(result.st_file_attributes, int_types))
        self.assertTrue(0 <= result.st_file_attributes <= 0xFFFFFFFF)

    def test_file_attributes(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        if sys.platform != 'win32' or not self.has_file_attributes:
            # st_file_attributes is Win32 specific
            return self.skipTest('st_file_attributes not supported')

        entries = dict((e.name, e) for e in self.scandir_func(TEST_PATH))

        # test st_file_attributes on a file (FILE_ATTRIBUTE_DIRECTORY not set)
        result = entries['file1.txt'].stat()
        self.check_file_attributes(result)
        self.assertEqual(result.st_file_attributes & FILE_ATTRIBUTE_DIRECTORY, 0)

        # test st_file_attributes on a directory (FILE_ATTRIBUTE_DIRECTORY set)
        result = entries['subdir'].stat()
        self.check_file_attributes(result)
        self.assertEqual(result.st_file_attributes & FILE_ATTRIBUTE_DIRECTORY,
                         FILE_ATTRIBUTE_DIRECTORY)

    def test_path(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        entries = sorted(self.scandir_func(TEST_PATH), key=lambda e: e.name)
        self.assertEqual([os.path.basename(e.name) for e in entries],
                         ['file1.txt', 'file2.txt', 'linkdir', 'subdir'])
        self.assertEqual([os.path.normpath(os.path.join(TEST_PATH, e.name)) for e in entries],
                         [os.path.normpath(e.path) for e in entries])

    def test_symlink(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        if not symlinks_supported:
            return self.skipTest('symbolic links not supported')

        entries = sorted(self.scandir_func(os.path.join(TEST_PATH, 'linkdir')),
                         key=lambda e: e.name)

        self.assertEqual([(e.name, e.is_symlink()) for e in entries],
                         [('file1.txt', False),
                          ('link_to_dir', True),
                          ('link_to_file', True),
                          ('linksubdir', False)])

        self.assertEqual([(e.name, e.is_file(), e.is_file(follow_symlinks=False))
                          for e in entries],
                         [('file1.txt', True, True),
                          ('link_to_dir', False, False),
                          ('link_to_file', True, False),
                          ('linksubdir', False, False)])

        self.assertEqual([(e.name, e.is_dir(), e.is_dir(follow_symlinks=False))
                          for e in entries],
                         [('file1.txt', False, False),
                          ('link_to_dir', True, False),
                          ('link_to_file', False, False),
                          ('linksubdir', True, True)])

    def test_bytes(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        # Check that unicode filenames are returned correctly as bytes in output
        path = os.path.join(TEST_PATH, 'subdir').encode(sys.getfilesystemencoding(), 'replace')
        self.assertTrue(isinstance(path, bytes))

        # Python 3.6 on Windows fixes the bytes filename thing by using UTF-8
        if IS_PY3 and sys.platform == 'win32':
            if not (sys.version_info >= (3, 6) and self.scandir_func == os.scandir):
                self.assertRaises(TypeError, self.scandir_func, path)
                return

        entries = [e for e in self.scandir_func(path) if e.name.startswith(b'unicod')]
        self.assertEqual(len(entries), 1)
        entry = entries[0]

        self.assertTrue(isinstance(entry.name, bytes))
        self.assertTrue(isinstance(entry.path, bytes))

        # b'unicod?.txt' on Windows, b'unicod\xc6\x8f.txt' (UTF-8) or similar on POSIX
        entry_name = 'unicod\u018f.txt'.encode(sys.getfilesystemencoding(), 'replace')
        self.assertEqual(entry.name, entry_name)
        self.assertEqual(entry.path, os.path.join(path, entry_name))

    def test_unicode(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        # Check that unicode filenames are returned correctly as (unicode) str in output
        path = os.path.join(TEST_PATH, 'subdir')
        if not IS_PY3:
            path = path.decode(sys.getfilesystemencoding(), 'replace')
        self.assertTrue(isinstance(path, str))
        entries = [e for e in self.scandir_func(path) if e.name.startswith('unicod')]
        self.assertEqual(len(entries), 1)
        entry = entries[0]

        self.assertTrue(isinstance(entry.name, str))
        self.assertTrue(isinstance(entry.path, str))

        entry_name = 'unicod\u018f.txt'
        self.assertEqual(entry.name, entry_name)
        self.assertEqual(entry.path, os.path.join(path, 'unicod\u018f.txt'))

        # Check that it handles unicode input properly
        path = os.path.join(TEST_PATH, 'subdir', 'unidir\u018f')
        self.assertTrue(isinstance(path, str))
        entries = list(self.scandir_func(path))
        self.assertEqual(len(entries), 1)
        entry = entries[0]

        self.assertTrue(isinstance(entry.name, str))
        self.assertTrue(isinstance(entry.path, str))
        self.assertEqual(entry.name, 'file1.txt')
        self.assertEqual(entry.path, os.path.join(path, 'file1.txt'))

    def test_walk_unicode_handling(self):
        if not hasattr(self, 'scandir_func'):
            self.skipTest('skip mixin')
        encoding = sys.getfilesystemencoding()
        dirname_unicode = u'test_unicode_dir'
        dirname_bytes = dirname_unicode.encode(encoding)
        dirpath = os.path.join(TEST_PATH.encode(encoding), dirname_bytes)
        try:
            os.makedirs(dirpath)

            if sys.platform != 'win32':
                # test bytes
                self.assertTrue(isinstance(dirpath, bytes))
                for (path, dirs, files) in scandir.walk(dirpath):
                    self.assertTrue(isinstance(path, bytes))

            # test unicode
            text_type = str if IS_PY3 else unicode
            dirpath_unicode = text_type(dirpath, encoding)
            self.assertTrue(isinstance(dirpath_unicode, text_type))
            for (path, dirs, files) in scandir.walk(dirpath_unicode):
                self.assertTrue(isinstance(path, text_type))
        finally:
            shutil.rmtree(dirpath)

if has_scandir:
    class TestScandirGeneric(TestMixin, unittest.TestCase):
        def setUp(self):
            self.scandir_func = scandir.scandir_generic
            self.has_file_attributes = False
            TestMixin.setUp(self)


    if getattr(scandir, 'scandir_python', None):
        class TestScandirPython(TestMixin, unittest.TestCase):
            def setUp(self):
                self.scandir_func = scandir.scandir_python
                self.has_file_attributes = True
                TestMixin.setUp(self)


    if getattr(scandir, 'scandir_c', None):
        class TestScandirC(TestMixin, unittest.TestCase):
            def setUp(self):
                self.scandir_func = scandir.scandir_c
                self.has_file_attributes = True
                TestMixin.setUp(self)


    class TestScandirDirEntry(unittest.TestCase):
        def setUp(self):
            if not os.path.exists(TEST_PATH):
                setup_main()

        def test_iter_returns_dir_entry(self):
            it = scandir.scandir(TEST_PATH)
            entry = next(it)
            assert isinstance(entry, scandir.DirEntry)


if hasattr(os, 'scandir'):
    class TestScandirOS(TestMixin, unittest.TestCase):
        def setUp(self):
            self.scandir_func = os.scandir
            self.has_file_attributes = True
            TestMixin.setUp(self)
