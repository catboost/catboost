# cython: c_string_type=str, c_string_encoding=utf8

from libcpp.string cimport string as std_string
from util.generic.string cimport TString, npos

import pytest
import unittest

import sys


class TestStroka(unittest.TestCase):
    def test_unicode(self):
        cdef TString x = "привет"
        self.assertEquals(x, "привет")


    def test_ctor1(self):
        cdef TString tmp = TString()
        cdef TString tmp2 = TString(tmp)
        self.assertEquals(tmp2, "")

    def test_ctor2(self):
        cdef std_string tmp = b"hello"
        cdef TString tmp2 = TString(tmp)
        self.assertEquals(tmp2, "hello")

    def test_ctor3(self):
        cdef TString tmp = b"hello"
        cdef TString tmp2 = TString(tmp, 0, 4)
        self.assertEquals(tmp2, "hell")

    def test_ctor4(self):
        cdef TString tmp = TString(<char*>b"hello")
        self.assertEquals(tmp, "hello")

    def test_ctor5(self):
        cdef TString tmp = TString(<char*>b"hello", 4)
        self.assertEquals(tmp, "hell")

    def test_ctor6(self):
        cdef TString tmp = TString(<char*>b"hello", 1, 3)
        self.assertEquals(tmp, "ell")

    def test_ctor7(self):
        cdef TString tmp = TString(3, <char>'x')
        self.assertEquals(tmp, "xxx")

    def test_ctor8(self):
        cdef bytes tmp = b"hello"
        cdef TString tmp2 = TString(<char*>tmp, <char*>tmp + 4)
        self.assertEquals(tmp2, "hell")

    def test_compare(self):
        cdef TString tmp1 = b"abacab"
        cdef TString tmp2 = b"abacab"
        cdef TString tmp3 = b"abacac"

        self.assertTrue(tmp1.compare(tmp2) == 0)
        self.assertTrue(tmp1.compare(tmp3) < 0)
        self.assertTrue(tmp3.compare(tmp1) > 0)

        self.assertTrue(tmp1 == tmp2)
        self.assertTrue(tmp1 != tmp3)

        self.assertTrue(tmp1 < tmp3)
        self.assertTrue(tmp1 <= tmp3)

        self.assertTrue(tmp3 > tmp1)
        self.assertTrue(tmp3 >= tmp1)

    def test_operator_assign(self):
        cdef TString tmp = b"hello"
        cdef TString tmp2 = tmp
        self.assertEquals(tmp2, "hello")

    def test_operator_plus(self):
        cdef TString tmp = TString(b"hello ") + TString(b"world")
        self.assertEquals(tmp, "hello world")

    def test_c_str(self):
        cdef TString tmp = b"hello"
        if sys.version_info.major == 2:
            self.assertEquals(bytes(tmp.c_str()), b"hello")
        else:
            self.assertEquals(bytes(tmp.c_str(), 'utf8'), b"hello")

    def test_length(self):
        cdef TString tmp = b"hello"
        self.assertEquals(tmp.size(), tmp.length())

    def test_index(self):
        cdef TString tmp = b"hello"

        self.assertEquals(<bytes>tmp[0], b'h')
        self.assertEquals(<bytes>tmp.at(0), b'h')

        self.assertEquals(<bytes>tmp[4], b'o')
        self.assertEquals(<bytes>tmp.at(4), b'o')

        # Actually, TString::at() is noexcept
        # with pytest.raises(IndexError):
        #     tmp.at(100)

    def test_append(self):
        cdef TString tmp
        cdef TString tmp2 = b"fuu"

        tmp.append(tmp2)
        self.assertEquals(tmp, "fuu")

        tmp.append(tmp2, 1, 2)
        self.assertEquals(tmp, "fuuuu")

        tmp.append(<char*>"ll ")
        self.assertEquals(tmp, "fuuuull ")

        tmp.append(<char*>"of greatness", 4)
        self.assertEquals(tmp, "fuuuull of g")

        tmp.append(2, <char>b'o')
        self.assertEquals(tmp, "fuuuull of goo")

        tmp.push_back(b'z')
        self.assertEquals(tmp, "fuuuull of gooz")

    def test_assign(self):
        cdef TString tmp

        tmp.assign(b"one")
        self.assertEquals(tmp, "one")

        tmp.assign(b"two hundred", 0, 3)
        self.assertEquals(tmp, "two")

        tmp.assign(<char*>b"three")
        self.assertEquals(tmp, "three")

        tmp.assign(<char*>b"three fiddy", 5)
        self.assertEquals(tmp, "three")

    def test_insert(self):
        cdef TString tmp

        tmp = b"xx"
        tmp.insert(1, b"foo")
        self.assertEquals(tmp, "xfoox")

        tmp = b"xx"
        tmp.insert(1, b"haxor", 1, 3)
        self.assertEquals(tmp, "xaxox")

        tmp = b"xx"
        tmp.insert(1, <char*>b"foo")
        self.assertEquals(tmp, "xfoox")

        tmp = b"xx"
        tmp.insert(1, <char*>b"foozzy", 3)
        self.assertEquals(tmp, "xfoox")

        tmp = b"xx"
        tmp.insert(1, 2, <char>b'u')
        self.assertEquals(tmp, "xuux")

    def test_copy(self):
        cdef char buf[16]
        cdef TString tmp = b"hello"
        tmp.copy(buf, 5, 0)
        self.assertEquals(buf[:5], "hello")

    def test_find(self):
        cdef TString haystack = b"whole lotta bytes"
        cdef TString needle = "hole"

        self.assertEquals(haystack.find(needle), 1)
        self.assertEquals(haystack.find(needle, 3), npos)

        self.assertEquals(haystack.find(<char>b'h'), 1)
        self.assertEquals(haystack.find(<char>b'h', 3), npos)

    def test_rfind(self):
        cdef TString haystack = b"whole lotta bytes"
        cdef TString needle = b"hole"

        self.assertEquals(haystack.rfind(needle), 1)
        self.assertEquals(haystack.rfind(needle, 0), npos)

        self.assertEquals(haystack.rfind(<char>b'h'), 1)
        self.assertEquals(haystack.rfind(<char>b'h', 0), npos)

    def test_find_first_of(self):
        cdef TString haystack = b"whole lotta bytes"
        cdef TString cset = b"hxz"

        self.assertEquals(haystack.find_first_of(<char>b'h'), 1)
        self.assertEquals(haystack.find_first_of(<char>b'h', 3), npos)

        self.assertEquals(haystack.find_first_of(cset), 1)
        self.assertEquals(haystack.find_first_of(cset, 3), npos)

    def test_first_not_of(self):
        cdef TString haystack = b"whole lotta bytes"
        cdef TString cset = b"wxz"

        self.assertEquals(haystack.find_first_not_of(<char>b'w'), 1)
        self.assertEquals(haystack.find_first_not_of(<char>b'w', 3), 3)

        self.assertEquals(haystack.find_first_not_of(cset), 1)
        self.assertEquals(haystack.find_first_not_of(cset, 3), 3)

    def test_find_last_of(self):
        cdef TString haystack = b"whole lotta bytes"
        cdef TString cset = b"hxz"

        self.assertEquals(haystack.find_last_of(<char>b'h'), 1)
        self.assertEquals(haystack.find_last_of(<char>b'h', 0), npos)

        self.assertEquals(haystack.find_last_of(cset), 1)
        self.assertEquals(haystack.find_last_of(cset, 0), npos)

    def test_substr(self):
        cdef TString tmp = b"foobar"

        self.assertEquals(tmp.substr(1), "oobar")
        self.assertEquals(tmp.substr(1, 4), "ooba")
