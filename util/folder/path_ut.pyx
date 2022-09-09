# cython: c_string_type=str, c_string_encoding=utf8

from util.folder.path cimport TFsPath
from util.generic.string cimport TString, TStringBuf
from util.generic.vector cimport TVector

import unittest
import yatest.common

import os.path


class TestPath(unittest.TestCase):
    def test_ctor1(self):
        cdef TFsPath path = TFsPath()
        self.assertEqual(path.IsDefined(), False)
        self.assertEquals(path.c_str(), "")

    def test_ctor2(self):
        cdef TString str_path = "/a/b/c"
        cdef TFsPath path = TFsPath(str_path)
        self.assertEqual(path.IsDefined(), True)
        self.assertEquals(path.c_str(), "/a/b/c")

    def test_ctor3(self):
        cdef TStringBuf buf_path = "/a/b/c"
        cdef TFsPath path = TFsPath(buf_path)
        self.assertEqual(path.IsDefined(), True)
        self.assertEquals(path.c_str(), "/a/b/c")

    def test_ctor4(self):
        cdef char* char_path = "/a/b/c"
        cdef TFsPath path = TFsPath(char_path)
        self.assertEqual(path.IsDefined(), True)
        self.assertEquals(path.c_str(), "/a/b/c")

    def test_assignment(self):
        cdef TFsPath path1 = TFsPath("/a/b")
        cdef TFsPath path2 = TFsPath("/a/c")

        self.assertEquals(path1.GetPath(), "/a/b")
        self.assertEquals(path2.GetPath(), "/a/c")

        path2 = path1

        self.assertEquals(path1.GetPath(), "/a/b")
        self.assertEquals(path2.GetPath(), "/a/b")

    def test_check_defined(self):
        cdef TFsPath path1 = TFsPath()
        with self.assertRaises(RuntimeError):
            path1.CheckDefined()
        self.assertEqual(path1.IsDefined(), False)
        if path1:
            assert False
        else:
            pass

        cdef TFsPath path2 = TFsPath("")
        with self.assertRaises(RuntimeError):
            path2.CheckDefined()
        self.assertEqual(path2.IsDefined(), False)
        if path2:
            assert False
        else:
            pass

        cdef TFsPath path3 = TFsPath("/")
        path3.CheckDefined()
        self.assertEqual(path3.IsDefined(), True)
        if path3:
            pass
        else:
            assert False

    def test_comparison(self):
        cdef TFsPath path1 = TFsPath("/a/b")
        cdef TFsPath path2 = TFsPath("/a/c")
        cdef TFsPath path3 = TFsPath("/a/b")

        self.assertEqual(path1 == path3, True)
        self.assertEqual(path1 != path2, True)
        self.assertEqual(path3 != path2, True)

    def test_concatenation(self):
        cdef TFsPath path1 = TFsPath("/a")
        cdef TFsPath path2 = TFsPath("b")
        cdef TFsPath path3 = path1 / path2
        cdef TFsPath path4 = TFsPath("/a/b")

        self.assertEqual(path3 == path4, True)

    def test_fix(self):
        cdef TFsPath path = TFsPath("test_fix/b/c/../d")
        cdef TFsPath fixed = path.Fix()
        self.assertEquals(fixed.GetPath(), "test_fix/b/d")

    def test_parts(self):
        cdef TFsPath path = TFsPath("/a/b/c")
        self.assertEquals(path.GetPath(), "/a/b/c")
        self.assertEquals(path.GetName(), "c")
        self.assertEquals(path.GetExtension(), "")
        self.assertEquals(path.Basename(), "c")
        self.assertEquals(path.Dirname(), "/a/b")

        cdef TFsPath path_ext = TFsPath("/a/b/c.ext")
        self.assertEquals(path_ext.GetPath(), "/a/b/c.ext")
        self.assertEquals(path_ext.GetName(), "c.ext")
        self.assertEquals(path_ext.GetExtension(), "ext")
        self.assertEquals(path_ext.Basename(), "c.ext")
        self.assertEquals(path_ext.Dirname(), "/a/b")

        cdef TFsPath path_only_ext = TFsPath("/a/b/.ext")
        self.assertEquals(path_only_ext.GetPath(), "/a/b/.ext")
        self.assertEquals(path_only_ext.GetName(), ".ext")
        self.assertEquals(path_only_ext.GetExtension(), "")
        self.assertEquals(path_only_ext.Basename(), ".ext")
        self.assertEquals(path_only_ext.Dirname(), "/a/b")

        cdef TFsPath path_dir = TFsPath("/a/b/")
        self.assertEquals(path_dir.GetPath(), "/a/b/")
        self.assertEquals(path_dir.GetName(), "b")
        self.assertEquals(path_dir.GetExtension(), "")
        self.assertEquals(path_dir.Basename(), "b")
        self.assertEquals(path_dir.Dirname(), "/a")

    def test_absolute(self):
        cdef TFsPath path_absolute = TFsPath("/a/b/c")
        self.assertEquals(path_absolute.IsAbsolute(), True)
        self.assertEquals(path_absolute.IsRelative(), False)

        self.assertEquals(path_absolute.IsSubpathOf(TFsPath("/a/b")), True)
        self.assertEquals(path_absolute.IsNonStrictSubpathOf(TFsPath("/a/b")), True)
        self.assertEquals(TFsPath("/a/b").IsContainerOf(path_absolute), True)

        self.assertEquals(path_absolute.IsSubpathOf(TFsPath("/a/b/c")), False)
        self.assertEquals(path_absolute.IsNonStrictSubpathOf(TFsPath("/a/b/c")), True)
        self.assertEquals(TFsPath("/a/b/c").IsContainerOf(path_absolute), False)

        self.assertEquals(path_absolute.IsSubpathOf(TFsPath("/a/c")), False)
        self.assertEquals(path_absolute.IsNonStrictSubpathOf(TFsPath("/a/c")), False)
        self.assertEquals(TFsPath("/a/c").IsContainerOf(path_absolute), False)

        with self.assertRaises(RuntimeError):
            path_absolute.RelativeTo(TFsPath("/a/c"))
        self.assertEquals(path_absolute.RelativePath(TFsPath("/a/с")).GetPath(), "../b/c")
        self.assertEquals(path_absolute.RelativeTo(TFsPath("/a")).GetPath(), "b/c")
        self.assertEquals(path_absolute.RelativePath(TFsPath("/a")).GetPath(), "b/c")
        self.assertEquals(path_absolute.RelativeTo(TFsPath("/")).GetPath(), "a/b/c")
        self.assertEquals(path_absolute.RelativePath(TFsPath("/")).GetPath(), "a/b/c")

        with self.assertRaises(RuntimeError):
            path_absolute.RelativeTo(TFsPath("./a"))
        with self.assertRaises(RuntimeError):
            path_absolute.RelativePath(TFsPath("d"))
        self.assertEquals(path_absolute.RelativePath(TFsPath("./a")).GetPath(), "b/c")

        self.assertEquals(path_absolute.Parent().GetPath(), "/a/b")
        self.assertEquals(path_absolute.Child("d").GetPath(), "/a/b/c/d")

    def test_relative(self):
        cdef TFsPath path_relative_1 = TFsPath("a/b/c")
        self.assertEquals(path_relative_1.IsAbsolute(), False)
        self.assertEquals(path_relative_1.IsRelative(), True)

        self.assertEquals(path_relative_1.IsSubpathOf(TFsPath("a/b")), True)
        self.assertEquals(path_relative_1.IsNonStrictSubpathOf(TFsPath("a/b")), True)
        self.assertEquals(TFsPath("a/b").IsContainerOf(path_relative_1), True)

        self.assertEquals(path_relative_1.IsSubpathOf(TFsPath("a/b/c")), False)
        self.assertEquals(path_relative_1.IsNonStrictSubpathOf(TFsPath("a/b/c")), True)
        self.assertEquals(TFsPath("a/b/c").IsContainerOf(path_relative_1), False)

        self.assertEquals(path_relative_1.IsSubpathOf(TFsPath("a/c")), False)
        self.assertEquals(path_relative_1.IsNonStrictSubpathOf(TFsPath("a/c")), False)
        self.assertEquals(TFsPath("a/c").IsContainerOf(path_relative_1), False)

        self.assertEquals(path_relative_1.Parent().GetPath(), "a/b")
        self.assertEquals(path_relative_1.Child("d").GetPath(), "a/b/c/d")

        cdef TFsPath path_relative_2 = TFsPath("./a/b/c")
        self.assertEquals(path_relative_2.IsAbsolute(), False)
        self.assertEquals(path_relative_2.IsRelative(), True)

        self.assertEquals(path_relative_2.IsSubpathOf(TFsPath("a/b")), True)
        self.assertEquals(path_relative_2.IsNonStrictSubpathOf(TFsPath("a/b")), True)
        self.assertEquals(TFsPath("a/b").IsContainerOf(path_relative_2), True)

        self.assertEquals(path_relative_2.IsSubpathOf(TFsPath("a/b/c")), False)
        self.assertEquals(path_relative_2.IsNonStrictSubpathOf(TFsPath("a/b/c")), True)
        self.assertEquals(TFsPath("a/b/c").IsContainerOf(path_relative_2), False)

        self.assertEquals(path_relative_2.IsSubpathOf(TFsPath("a/c")), False)
        self.assertEquals(path_relative_2.IsNonStrictSubpathOf(TFsPath("a/c")), False)
        self.assertEquals(TFsPath("a/c").IsContainerOf(path_relative_2), False)

        with self.assertRaises(RuntimeError):
            path_relative_2.RelativeTo(TFsPath("a/c"))
        self.assertEquals(path_relative_2.RelativePath(TFsPath("a/с")).GetPath(), "../b/c")
        self.assertEquals(path_relative_2.RelativeTo(TFsPath("a")).GetPath(), "b/c")
        self.assertEquals(path_relative_2.RelativePath(TFsPath("a")).GetPath(), "b/c")
        self.assertEquals(path_relative_2.RelativeTo(TFsPath("./")).GetPath(), "a/b/c")
        self.assertEquals(path_relative_2.RelativePath(TFsPath("/a")).GetPath(), "b/c")

        with self.assertRaises(RuntimeError):
            self.assertEquals(path_relative_2.RelativePath(TFsPath("./")).GetPath(), "a/b/c")

        with self.assertRaises(RuntimeError):
            path_relative_2.RelativeTo(TFsPath("/d"))
        with self.assertRaises(RuntimeError):
            path_relative_2.RelativePath(TFsPath("/d"))
        with self.assertRaises(RuntimeError):
            path_relative_2.RelativePath(TFsPath("/"))

        self.assertEquals(path_relative_2.Parent().GetPath(), "a/b")
        self.assertEquals(path_relative_2.Child("d").GetPath(), "a/b/c/d")

    def test_mkdir(self):
        cdef TFsPath directory = TFsPath("test_mkdir")
        cdef TFsPath full = directory / directory
        cdef TFsPath internal = full / directory
        with self.assertRaises(RuntimeError):
            full.MkDir()
        full.MkDirs()
        internal.MkDir()

    def test_list(self):
        cdef TFsPath dir = TFsPath("test_list")
        dir.MkDir()
        TFsPath("test_list/b").Touch()
        TFsPath("test_list/c").Touch()

        cdef TVector[TFsPath] files
        cdef TVector[TString] names

        dir.List(files)
        dir.ListNames(names)

        self.assertEquals(files.size(), 2)
        self.assertEquals(sorted([files[0].GetPath(), files[1].GetPath()]), ["test_list/b", "test_list/c"])
        self.assertEquals(names.size(), 2)
        self.assertEquals(sorted(list(names)), ["b", "c"])

    def test_contains(self):
        cdef TFsPath path = TFsPath("a/b/c")
        self.assertEquals(path.Contains("c"), True)
        self.assertEquals(path.Contains("b"), True)
        self.assertEquals(path.Contains("d"), False)

    def test_delete(self):
        cdef TFsPath root = TFsPath("/")
        with self.assertRaises(RuntimeError):
            root.DeleteIfExists()
        with self.assertRaises(RuntimeError):
            root.ForceDelete()

        cdef TFsPath directory = TFsPath("test_delete")
        cdef TFsPath full = directory / directory
        full.MkDirs()

        self.assertEquals(full.Exists(), True)
        with self.assertRaises(RuntimeError):
            directory.DeleteIfExists()
        self.assertEquals(directory.Exists(), True)
        directory.ForceDelete()
        self.assertEquals(directory.Exists(), False)

        cdef TFsPath local_file = TFsPath("test_delete_1")
        self.assertEquals(local_file.Exists(), False)
        local_file.DeleteIfExists()
        self.assertEquals(local_file.Exists(), False)
        local_file.ForceDelete()
        self.assertEquals(local_file.Exists(), False)

        local_file.Touch()
        self.assertEquals(local_file.Exists(), True)
        local_file.DeleteIfExists()
        self.assertEquals(local_file.Exists(), False)

        local_file.Touch()
        self.assertEquals(local_file.Exists(), True)
        local_file.ForceDelete()
        self.assertEquals(local_file.Exists(), False)

        full.MkDirs()
        self.assertEquals(full.Exists(), True)
        full.DeleteIfExists()
        self.assertEquals(full.Exists(), False)
        self.assertEquals(directory.Exists(), True)
        directory.DeleteIfExists()
        self.assertEquals(directory.Exists(), False)

    def test_checks(self):
        cdef TFsPath local_file = TFsPath("test_checks")
        with self.assertRaises(RuntimeError):
            local_file.CheckExists()
        local_file.Touch()
        self.assertEquals(local_file.Exists(), True)
        self.assertEquals(local_file.IsDirectory(), False)
        self.assertEquals(local_file.IsFile(), True)
        self.assertEquals(local_file.IsSymlink(), False)
        local_file.CheckExists()

        local_file.DeleteIfExists()
        local_file.MkDir()
        self.assertEquals(local_file.Exists(), True)
        self.assertEquals(local_file.IsDirectory(), True)
        self.assertEquals(local_file.IsFile(), False)
        self.assertEquals(local_file.IsSymlink(), False)
        local_file.CheckExists()

    def test_rename(self):
        cdef TFsPath path = TFsPath("test_rename_a")
        path.Touch()

        cdef TString path_str = "test_rename_b"
        cdef TFsPath path_from_str = TFsPath(path_str)
        self.assertEquals(path.Exists(), True)
        self.assertEquals(path_from_str.Exists(), False)
        path.RenameTo(path_str)
        self.assertEquals(path.Exists(), False)
        self.assertEquals(path_from_str.Exists(), True)

        cdef const char* path_char = "test_rename_c"
        cdef TFsPath path_from_char = TFsPath(path_char)
        self.assertEquals(path_from_str.Exists(), True)
        self.assertEquals(path_from_char.Exists(), False)
        path_from_str.RenameTo(path_char)
        self.assertEquals(path_from_str.Exists(), False)
        self.assertEquals(path_from_char.Exists(), True)

        path_from_char.RenameTo(path)

        self.assertEquals(path_from_char.Exists(), False)
        self.assertEquals(path.Exists(), True)

        path.ForceRenameTo(path_str)

        self.assertEquals(path_from_str.Exists(), True)
        self.assertEquals(path.Exists(), False)

        with self.assertRaises(RuntimeError):
            path_from_str.RenameTo("")

    def test_copy(self):
        cdef TString dst = "test_copy_dst"
        cdef TFsPath src_path = TFsPath("test_copy_src")
        cdef TFsPath dst_path = TFsPath(dst)
        self.assertEquals(src_path.Exists(), False)
        src_path.Touch()
        self.assertEquals(src_path.Exists(), True)
        src_path.CopyTo(dst, False)
        self.assertEquals(src_path.Exists(), True)
        self.assertEquals(dst_path.Exists(), True)

    def test_real_path(self):
        cdef TFsPath path = TFsPath("test_real_path_a")
        path.Touch()
        real_work_path = os.path.join(os.path.realpath(yatest.common.work_path()), "test_real_path_a")
        self.assertEquals(path.RealPath().GetPath(), real_work_path)
        self.assertEquals(path.RealLocation().GetPath(), real_work_path)
        with self.assertRaises(RuntimeError):
            path.ReadLink()

    def test_cwd(self):
        cdef TFsPath path = TFsPath.Cwd()
        self.assertEquals(path.GetPath(), os.path.realpath(yatest.common.work_path()))

    def test_swap(self):
        cdef TFsPath first = TFsPath("first")
        cdef TFsPath second = TFsPath("second")

        self.assertEquals(first.GetPath(), "first")
        self.assertEquals(second.GetPath(), "second")
        first.Swap(second)
        self.assertEquals(first.GetPath(), "second")
        self.assertEquals(second.GetPath(), "first")
        second.Swap(first)
        self.assertEquals(first.GetPath(), "first")
        self.assertEquals(second.GetPath(), "second")
