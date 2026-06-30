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
        self.assertEqual(path.c_str(), "")

    def test_ctor2(self):
        cdef TString str_path = "/a/b/c"
        cdef TFsPath path = TFsPath(str_path)
        self.assertEqual(path.IsDefined(), True)
        self.assertEqual(path.c_str(), "/a/b/c")

    def test_ctor3(self):
        cdef TStringBuf buf_path = "/a/b/c"
        cdef TFsPath path = TFsPath(buf_path)
        self.assertEqual(path.IsDefined(), True)
        self.assertEqual(path.c_str(), "/a/b/c")

    def test_ctor4(self):
        cdef char* char_path = "/a/b/c"
        cdef TFsPath path = TFsPath(char_path)
        self.assertEqual(path.IsDefined(), True)
        self.assertEqual(path.c_str(), "/a/b/c")

    def test_assignment(self):
        cdef TFsPath path1 = TFsPath("/a/b")
        cdef TFsPath path2 = TFsPath("/a/c")

        self.assertEqual(path1.GetPath(), "/a/b")
        self.assertEqual(path2.GetPath(), "/a/c")

        path2 = path1

        self.assertEqual(path1.GetPath(), "/a/b")
        self.assertEqual(path2.GetPath(), "/a/b")

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
        self.assertEqual(fixed.GetPath(), "test_fix/b/d")

    def test_parts(self):
        cdef TFsPath path = TFsPath("/a/b/c")
        self.assertEqual(path.GetPath(), "/a/b/c")
        self.assertEqual(path.GetName(), "c")
        self.assertEqual(path.GetExtension(), "")
        self.assertEqual(path.Basename(), "c")
        self.assertEqual(path.Dirname(), "/a/b")

        cdef TFsPath path_ext = TFsPath("/a/b/c.ext")
        self.assertEqual(path_ext.GetPath(), "/a/b/c.ext")
        self.assertEqual(path_ext.GetName(), "c.ext")
        self.assertEqual(path_ext.GetExtension(), "ext")
        self.assertEqual(path_ext.Basename(), "c.ext")
        self.assertEqual(path_ext.Dirname(), "/a/b")

        cdef TFsPath path_only_ext = TFsPath("/a/b/.ext")
        self.assertEqual(path_only_ext.GetPath(), "/a/b/.ext")
        self.assertEqual(path_only_ext.GetName(), ".ext")
        self.assertEqual(path_only_ext.GetExtension(), "")
        self.assertEqual(path_only_ext.Basename(), ".ext")
        self.assertEqual(path_only_ext.Dirname(), "/a/b")

        cdef TFsPath path_dir = TFsPath("/a/b/")
        self.assertEqual(path_dir.GetPath(), "/a/b/")
        self.assertEqual(path_dir.GetName(), "b")
        self.assertEqual(path_dir.GetExtension(), "")
        self.assertEqual(path_dir.Basename(), "b")
        self.assertEqual(path_dir.Dirname(), "/a")

    def test_absolute(self):
        cdef TFsPath path_absolute = TFsPath("/a/b/c")
        self.assertEqual(path_absolute.IsAbsolute(), True)
        self.assertEqual(path_absolute.IsRelative(), False)

        self.assertEqual(path_absolute.IsSubpathOf(TFsPath("/a/b")), True)
        self.assertEqual(path_absolute.IsNonStrictSubpathOf(TFsPath("/a/b")), True)
        self.assertEqual(TFsPath("/a/b").IsContainerOf(path_absolute), True)

        self.assertEqual(path_absolute.IsSubpathOf(TFsPath("/a/b/c")), False)
        self.assertEqual(path_absolute.IsNonStrictSubpathOf(TFsPath("/a/b/c")), True)
        self.assertEqual(TFsPath("/a/b/c").IsContainerOf(path_absolute), False)

        self.assertEqual(path_absolute.IsSubpathOf(TFsPath("/a/c")), False)
        self.assertEqual(path_absolute.IsNonStrictSubpathOf(TFsPath("/a/c")), False)
        self.assertEqual(TFsPath("/a/c").IsContainerOf(path_absolute), False)

        with self.assertRaises(RuntimeError):
            path_absolute.RelativeTo(TFsPath("/a/c"))
        self.assertEqual(path_absolute.RelativePath(TFsPath("/a/с")).GetPath(), "../b/c")
        self.assertEqual(path_absolute.RelativeTo(TFsPath("/a")).GetPath(), "b/c")
        self.assertEqual(path_absolute.RelativePath(TFsPath("/a")).GetPath(), "b/c")
        self.assertEqual(path_absolute.RelativeTo(TFsPath("/")).GetPath(), "a/b/c")
        self.assertEqual(path_absolute.RelativePath(TFsPath("/")).GetPath(), "a/b/c")

        with self.assertRaises(RuntimeError):
            path_absolute.RelativeTo(TFsPath("./a"))
        with self.assertRaises(RuntimeError):
            path_absolute.RelativePath(TFsPath("d"))
        self.assertEqual(path_absolute.RelativePath(TFsPath("./a")).GetPath(), "b/c")

        self.assertEqual(path_absolute.Parent().GetPath(), "/a/b")
        self.assertEqual(path_absolute.Child("d").GetPath(), "/a/b/c/d")

    def test_relative(self):
        cdef TFsPath path_relative_1 = TFsPath("a/b/c")
        self.assertEqual(path_relative_1.IsAbsolute(), False)
        self.assertEqual(path_relative_1.IsRelative(), True)

        self.assertEqual(path_relative_1.IsSubpathOf(TFsPath("a/b")), True)
        self.assertEqual(path_relative_1.IsNonStrictSubpathOf(TFsPath("a/b")), True)
        self.assertEqual(TFsPath("a/b").IsContainerOf(path_relative_1), True)

        self.assertEqual(path_relative_1.IsSubpathOf(TFsPath("a/b/c")), False)
        self.assertEqual(path_relative_1.IsNonStrictSubpathOf(TFsPath("a/b/c")), True)
        self.assertEqual(TFsPath("a/b/c").IsContainerOf(path_relative_1), False)

        self.assertEqual(path_relative_1.IsSubpathOf(TFsPath("a/c")), False)
        self.assertEqual(path_relative_1.IsNonStrictSubpathOf(TFsPath("a/c")), False)
        self.assertEqual(TFsPath("a/c").IsContainerOf(path_relative_1), False)

        self.assertEqual(path_relative_1.Parent().GetPath(), "a/b")
        self.assertEqual(path_relative_1.Child("d").GetPath(), "a/b/c/d")

        cdef TFsPath path_relative_2 = TFsPath("./a/b/c")
        self.assertEqual(path_relative_2.IsAbsolute(), False)
        self.assertEqual(path_relative_2.IsRelative(), True)

        self.assertEqual(path_relative_2.IsSubpathOf(TFsPath("a/b")), True)
        self.assertEqual(path_relative_2.IsNonStrictSubpathOf(TFsPath("a/b")), True)
        self.assertEqual(TFsPath("a/b").IsContainerOf(path_relative_2), True)

        self.assertEqual(path_relative_2.IsSubpathOf(TFsPath("a/b/c")), False)
        self.assertEqual(path_relative_2.IsNonStrictSubpathOf(TFsPath("a/b/c")), True)
        self.assertEqual(TFsPath("a/b/c").IsContainerOf(path_relative_2), False)

        self.assertEqual(path_relative_2.IsSubpathOf(TFsPath("a/c")), False)
        self.assertEqual(path_relative_2.IsNonStrictSubpathOf(TFsPath("a/c")), False)
        self.assertEqual(TFsPath("a/c").IsContainerOf(path_relative_2), False)

        with self.assertRaises(RuntimeError):
            path_relative_2.RelativeTo(TFsPath("a/c"))
        self.assertEqual(path_relative_2.RelativePath(TFsPath("a/с")).GetPath(), "../b/c")
        self.assertEqual(path_relative_2.RelativeTo(TFsPath("a")).GetPath(), "b/c")
        self.assertEqual(path_relative_2.RelativePath(TFsPath("a")).GetPath(), "b/c")
        self.assertEqual(path_relative_2.RelativeTo(TFsPath("./")).GetPath(), "a/b/c")
        self.assertEqual(path_relative_2.RelativePath(TFsPath("/a")).GetPath(), "b/c")

        with self.assertRaises(RuntimeError):
            self.assertEqual(path_relative_2.RelativePath(TFsPath("./")).GetPath(), "a/b/c")

        with self.assertRaises(RuntimeError):
            path_relative_2.RelativeTo(TFsPath("/d"))
        with self.assertRaises(RuntimeError):
            path_relative_2.RelativePath(TFsPath("/d"))
        with self.assertRaises(RuntimeError):
            path_relative_2.RelativePath(TFsPath("/"))

        self.assertEqual(path_relative_2.Parent().GetPath(), "a/b")
        self.assertEqual(path_relative_2.Child("d").GetPath(), "a/b/c/d")

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

        self.assertEqual(files.size(), 2)
        self.assertEqual(sorted([files[0].GetPath(), files[1].GetPath()]), ["test_list/b", "test_list/c"])
        self.assertEqual(names.size(), 2)
        self.assertEqual(sorted(list(names)), ["b", "c"])

    def test_contains(self):
        cdef TFsPath path = TFsPath("a/b/c")
        self.assertEqual(path.Contains("c"), True)
        self.assertEqual(path.Contains("b"), True)
        self.assertEqual(path.Contains("d"), False)

    def test_delete(self):
        cdef TFsPath root = TFsPath("/")
        with self.assertRaises(RuntimeError):
            root.DeleteIfExists()
        with self.assertRaises(RuntimeError):
            root.ForceDelete()

        cdef TFsPath directory = TFsPath("test_delete")
        cdef TFsPath full = directory / directory
        full.MkDirs()

        self.assertEqual(full.Exists(), True)
        with self.assertRaises(RuntimeError):
            directory.DeleteIfExists()
        self.assertEqual(directory.Exists(), True)
        directory.ForceDelete()
        self.assertEqual(directory.Exists(), False)

        cdef TFsPath local_file = TFsPath("test_delete_1")
        self.assertEqual(local_file.Exists(), False)
        local_file.DeleteIfExists()
        self.assertEqual(local_file.Exists(), False)
        local_file.ForceDelete()
        self.assertEqual(local_file.Exists(), False)

        local_file.Touch()
        self.assertEqual(local_file.Exists(), True)
        local_file.DeleteIfExists()
        self.assertEqual(local_file.Exists(), False)

        local_file.Touch()
        self.assertEqual(local_file.Exists(), True)
        local_file.ForceDelete()
        self.assertEqual(local_file.Exists(), False)

        full.MkDirs()
        self.assertEqual(full.Exists(), True)
        full.DeleteIfExists()
        self.assertEqual(full.Exists(), False)
        self.assertEqual(directory.Exists(), True)
        directory.DeleteIfExists()
        self.assertEqual(directory.Exists(), False)

    def test_checks(self):
        cdef TFsPath local_file = TFsPath("test_checks")
        with self.assertRaises(RuntimeError):
            local_file.CheckExists()
        local_file.Touch()
        self.assertEqual(local_file.Exists(), True)
        self.assertEqual(local_file.IsDirectory(), False)
        self.assertEqual(local_file.IsFile(), True)
        self.assertEqual(local_file.IsSymlink(), False)
        local_file.CheckExists()

        local_file.DeleteIfExists()
        local_file.MkDir()
        self.assertEqual(local_file.Exists(), True)
        self.assertEqual(local_file.IsDirectory(), True)
        self.assertEqual(local_file.IsFile(), False)
        self.assertEqual(local_file.IsSymlink(), False)
        local_file.CheckExists()

    def test_rename(self):
        cdef TFsPath path = TFsPath("test_rename_a")
        path.Touch()

        cdef TString path_str = "test_rename_b"
        cdef TFsPath path_from_str = TFsPath(path_str)
        self.assertEqual(path.Exists(), True)
        self.assertEqual(path_from_str.Exists(), False)
        path.RenameTo(path_str)
        self.assertEqual(path.Exists(), False)
        self.assertEqual(path_from_str.Exists(), True)

        cdef const char* path_char = "test_rename_c"
        cdef TFsPath path_from_char = TFsPath(path_char)
        self.assertEqual(path_from_str.Exists(), True)
        self.assertEqual(path_from_char.Exists(), False)
        path_from_str.RenameTo(path_char)
        self.assertEqual(path_from_str.Exists(), False)
        self.assertEqual(path_from_char.Exists(), True)

        path_from_char.RenameTo(path)

        self.assertEqual(path_from_char.Exists(), False)
        self.assertEqual(path.Exists(), True)

        path.ForceRenameTo(path_str)

        self.assertEqual(path_from_str.Exists(), True)
        self.assertEqual(path.Exists(), False)

        with self.assertRaises(RuntimeError):
            path_from_str.RenameTo("")

    def test_copy(self):
        cdef TString dst = "test_copy_dst"
        cdef TFsPath src_path = TFsPath("test_copy_src")
        cdef TFsPath dst_path = TFsPath(dst)
        self.assertEqual(src_path.Exists(), False)
        src_path.Touch()
        self.assertEqual(src_path.Exists(), True)
        src_path.CopyTo(dst, False)
        self.assertEqual(src_path.Exists(), True)
        self.assertEqual(dst_path.Exists(), True)

    def test_real_path(self):
        cdef TFsPath path = TFsPath("test_real_path_a")
        path.Touch()
        real_work_path = os.path.join(os.path.realpath(yatest.common.work_path()), "test_real_path_a")
        self.assertEqual(path.RealPath().GetPath(), real_work_path)
        self.assertEqual(path.RealLocation().GetPath(), real_work_path)
        with self.assertRaises(RuntimeError):
            path.ReadLink()

    def test_cwd(self):
        cdef TFsPath path = TFsPath.Cwd()
        self.assertEqual(path.GetPath(), os.path.realpath(yatest.common.work_path()))

    def test_swap(self):
        cdef TFsPath first = TFsPath("first")
        cdef TFsPath second = TFsPath("second")

        self.assertEqual(first.GetPath(), "first")
        self.assertEqual(second.GetPath(), "second")
        first.Swap(second)
        self.assertEqual(first.GetPath(), "second")
        self.assertEqual(second.GetPath(), "first")
        second.Swap(first)
        self.assertEqual(first.GetPath(), "first")
        self.assertEqual(second.GetPath(), "second")
