# cython: c_string_type=str, c_string_encoding=utf8

from libcpp.string cimport string as std_string
from util.generic.string cimport TString
from util.memory.blob cimport TBlob

import pytest
import unittest


class TestBlob(unittest.TestCase):
    def test_ctor(self):
        cdef TBlob tmp = TBlob()
        cdef TBlob tmp2 = TBlob(tmp)
        self.assertEqual(tmp.Size(), 0)
        self.assertEqual(tmp2.Size(), 0)

    def test_empty_data(self):
        cdef TBlob tmp = TBlob()
        self.assertEqual(tmp.Data() == NULL, True)
        self.assertEqual(tmp.AsCharPtr() == NULL, True)
        self.assertEqual(tmp.AsUnsignedCharPtr() == NULL, True)
        self.assertEqual(tmp.Empty(), True)
        self.assertEqual(tmp.IsNull(), True)

    def test_empty_is_null(self):
        cdef TBlob tmp = TBlob.NoCopy("", 0)
        self.assertEqual(tmp.Empty(), True)
        self.assertEqual(tmp.IsNull(), False)

    def test_data_types(self):
        cdef const char* char_data = TBlob().AsCharPtr()
        cdef const unsigned char* uchar_data = TBlob().AsUnsignedCharPtr()
        cdef const void* void_data = TBlob().Data()

    def test_no_copy(self):
        cdef const char* txt = "hello world"
        cdef TBlob tmp = TBlob.NoCopy(txt, len(txt))
        self.assertEqual(tmp.AsCharPtr() - txt, 0)
        self.assertEqual(tmp.Size(), 11)
        self.assertEqual(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEqual(tmp.Empty(), False)
        self.assertEqual(tmp.IsNull(), False)

    def test_copy(self):
        cdef const char* txt = "hello world"
        cdef TBlob tmp = TBlob.Copy(txt, len(txt))
        self.assertNotEqual(tmp.AsCharPtr() - txt, 0)
        self.assertEqual(tmp.Size(), 11)
        self.assertEqual(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEqual(tmp.Empty(), False)
        self.assertEqual(tmp.IsNull(), False)

    def test_from_string(self):
        cdef TBlob tmp = TBlob.FromString(TString("hello world"))
        self.assertEqual(tmp.Size(), 11)
        self.assertEqual(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEqual(tmp.Empty(), False)
        self.assertEqual(tmp.IsNull(), False)

    def test_from_file(self):
        with open("file", "w") as f:
            f.write("hello world")
        cdef TBlob tmp = TBlob.FromFile("file")
        self.assertEqual(tmp.Size(), 11)
        self.assertEqual(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEqual(tmp.Empty(), False)
        self.assertEqual(tmp.IsNull(), False)

    def test_precharged_from_file(self):
        with open("precharged", "w") as f:
            f.write("hello world")
        cdef TBlob tmp = TBlob.PrechargedFromFile("precharged")
        self.assertEqual(tmp.Size(), 11)
        self.assertEqual(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEqual(tmp.Empty(), False)
        self.assertEqual(tmp.IsNull(), False)

    def test_swap_drop(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        cdef TBlob tmp2
        tmp2.Swap(tmp)
        self.assertEqual(tmp2.Size(), 11)
        self.assertEqual(tmp.Size(), 0)
        self.assertEqual(tmp2.AsCharPtr()[:tmp2.Size()], "hello world")
        tmp2.Swap(tmp)
        self.assertEqual(tmp2.Size(), 0)
        self.assertEqual(tmp.Size(), 11)
        tmp.Drop()
        self.assertEqual(tmp.Size(), 0)
        
    def test_operator_brackets(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        self.assertEqual(tmp[0], ord('h'))
        self.assertEqual(tmp[1], ord('e'))
        self.assertEqual(tmp[2], ord('l'))
        self.assertEqual(tmp[3], ord('l'))
        self.assertEqual(tmp[4], ord('o'))
        self.assertEqual(tmp[5], ord(' '))
        self.assertEqual(tmp[6], ord('w'))
        self.assertEqual(tmp[7], ord('o'))
        self.assertEqual(tmp[8], ord('r'))
        self.assertEqual(tmp[9], ord('l'))
        self.assertEqual(tmp[10], ord('d'))

    def test_operator_equal(self):
        cdef TBlob foo = TBlob.NoCopy("foo", 3)
        cdef TBlob bar = TBlob.NoCopy("bar", 3)
        self.assertEqual(foo.AsCharPtr(), "foo")
        self.assertEqual(bar.AsCharPtr(), "bar")
        bar = foo
        self.assertEqual(foo.AsCharPtr(), "foo")
        self.assertEqual(bar.AsCharPtr(), "foo")

    def test_sub_blob(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        self.assertEqual(tmp.SubBlob(0).Size(), 0)
        self.assertEqual(tmp.SubBlob(1).Size(), 1)
        self.assertEqual(tmp.SubBlob(5).Size(), 5)
        self.assertEqual(tmp.AsCharPtr() - tmp.SubBlob(0).AsCharPtr(), 0)

        self.assertEqual(tmp.SubBlob(0, 0).Size(), 0)
        self.assertEqual(tmp.SubBlob(0, 1).Size(), 1)
        self.assertEqual(tmp.SubBlob(0, 5).Size(), 5)
        self.assertEqual(tmp.AsCharPtr() - tmp.SubBlob(0, 0).AsCharPtr(), 0)

        self.assertEqual(tmp.SubBlob(1, 1).Size(), 0)
        self.assertEqual(tmp.SubBlob(1, 2).Size(), 1)
        self.assertEqual(tmp.SubBlob(1, 6).Size(), 5)
        self.assertEqual(tmp.SubBlob(1, 1).AsCharPtr() - tmp.AsCharPtr(), 1)

        with self.assertRaises(Exception):
            tmp.SubBlob(2, 1)

    def test_deep_copy(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        cdef TBlob tmp2 = tmp.DeepCopy()
        self.assertEqual(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEqual(tmp2.AsCharPtr()[:tmp2.Size()], "hello world")
        self.assertNotEqual(tmp2.AsCharPtr() - tmp.AsCharPtr(), 0)

