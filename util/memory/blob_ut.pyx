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
        self.assertEquals(tmp.Size(), 0)
        self.assertEquals(tmp2.Size(), 0)

    def test_empty_data(self):
        cdef TBlob tmp = TBlob()
        self.assertEquals(tmp.Data() == NULL, True)
        self.assertEquals(tmp.AsCharPtr() == NULL, True)
        self.assertEquals(tmp.AsUnsignedCharPtr() == NULL, True)
        self.assertEquals(tmp.Empty(), True)
        self.assertEquals(tmp.IsNull(), True)

    def test_empty_is_null(self):
        cdef TBlob tmp = TBlob.NoCopy("", 0)
        self.assertEquals(tmp.Empty(), True)
        self.assertEquals(tmp.IsNull(), False)

    def test_data_types(self):
        cdef const char* char_data = TBlob().AsCharPtr()
        cdef const unsigned char* uchar_data = TBlob().AsUnsignedCharPtr()
        cdef const void* void_data = TBlob().Data()

    def test_no_copy(self):
        cdef const char* txt = "hello world"
        cdef TBlob tmp = TBlob.NoCopy(txt, len(txt))
        self.assertEquals(tmp.AsCharPtr() - txt, 0)
        self.assertEquals(tmp.Size(), 11)
        self.assertEquals(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEquals(tmp.Empty(), False)
        self.assertEquals(tmp.IsNull(), False)

    def test_copy(self):
        cdef const char* txt = "hello world"
        cdef TBlob tmp = TBlob.Copy(txt, len(txt))
        self.assertNotEquals(tmp.AsCharPtr() - txt, 0)
        self.assertEquals(tmp.Size(), 11)
        self.assertEquals(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEquals(tmp.Empty(), False)
        self.assertEquals(tmp.IsNull(), False)

    def test_from_string(self):
        cdef TBlob tmp = TBlob.FromString(TString("hello world"))
        self.assertEquals(tmp.Size(), 11)
        self.assertEquals(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEquals(tmp.Empty(), False)
        self.assertEquals(tmp.IsNull(), False)

    def test_from_file(self):
        with open("file", "w") as f:
            f.write("hello world")
        cdef TBlob tmp = TBlob.FromFile("file")
        self.assertEquals(tmp.Size(), 11)
        self.assertEquals(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEquals(tmp.Empty(), False)
        self.assertEquals(tmp.IsNull(), False)

    def test_precharged_from_file(self):
        with open("precharged", "w") as f:
            f.write("hello world")
        cdef TBlob tmp = TBlob.PrechargedFromFile("precharged")
        self.assertEquals(tmp.Size(), 11)
        self.assertEquals(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEquals(tmp.Empty(), False)
        self.assertEquals(tmp.IsNull(), False)

    def test_swap_drop(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        cdef TBlob tmp2
        tmp2.Swap(tmp)
        self.assertEquals(tmp2.Size(), 11)
        self.assertEquals(tmp.Size(), 0)
        self.assertEquals(tmp2.AsCharPtr()[:tmp2.Size()], "hello world")
        tmp2.Swap(tmp)
        self.assertEquals(tmp2.Size(), 0)
        self.assertEquals(tmp.Size(), 11)
        tmp.Drop()
        self.assertEquals(tmp.Size(), 0)
        
    def test_operator_brackets(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        self.assertEquals(tmp[0], ord('h'))
        self.assertEquals(tmp[1], ord('e'))
        self.assertEquals(tmp[2], ord('l'))
        self.assertEquals(tmp[3], ord('l'))
        self.assertEquals(tmp[4], ord('o'))
        self.assertEquals(tmp[5], ord(' '))
        self.assertEquals(tmp[6], ord('w'))
        self.assertEquals(tmp[7], ord('o'))
        self.assertEquals(tmp[8], ord('r'))
        self.assertEquals(tmp[9], ord('l'))
        self.assertEquals(tmp[10], ord('d'))

    def test_operator_equal(self):
        cdef TBlob foo = TBlob.NoCopy("foo", 3)
        cdef TBlob bar = TBlob.NoCopy("bar", 3)
        self.assertEquals(foo.AsCharPtr(), "foo")
        self.assertEquals(bar.AsCharPtr(), "bar")
        bar = foo
        self.assertEquals(foo.AsCharPtr(), "foo")
        self.assertEquals(bar.AsCharPtr(), "foo")

    def test_sub_blob(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        self.assertEquals(tmp.SubBlob(0).Size(), 0)
        self.assertEquals(tmp.SubBlob(1).Size(), 1)
        self.assertEquals(tmp.SubBlob(5).Size(), 5)
        self.assertEquals(tmp.AsCharPtr() - tmp.SubBlob(0).AsCharPtr(), 0)

        self.assertEquals(tmp.SubBlob(0, 0).Size(), 0)
        self.assertEquals(tmp.SubBlob(0, 1).Size(), 1)
        self.assertEquals(tmp.SubBlob(0, 5).Size(), 5)
        self.assertEquals(tmp.AsCharPtr() - tmp.SubBlob(0, 0).AsCharPtr(), 0)

        self.assertEquals(tmp.SubBlob(1, 1).Size(), 0)
        self.assertEquals(tmp.SubBlob(1, 2).Size(), 1)
        self.assertEquals(tmp.SubBlob(1, 6).Size(), 5)
        self.assertEquals(tmp.SubBlob(1, 1).AsCharPtr() - tmp.AsCharPtr(), 1)

        with self.assertRaises(Exception):
            tmp.SubBlob(2, 1)

    def test_deep_copy(self):
        cdef TBlob tmp = TBlob.NoCopy("hello world", 11)
        cdef TBlob tmp2 = tmp.DeepCopy()
        self.assertEquals(tmp.AsCharPtr()[:tmp.Size()], "hello world")
        self.assertEquals(tmp2.AsCharPtr()[:tmp2.Size()], "hello world")
        self.assertNotEquals(tmp2.AsCharPtr() - tmp.AsCharPtr(), 0)

