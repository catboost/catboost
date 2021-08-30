# cython: c_string_type=str, c_string_encoding=utf8

from util.generic.hash_set cimport THashSet
from util.generic.string cimport TString

import pytest
import unittest

from cython.operator cimport dereference as deref


class TestHashSet(unittest.TestCase):

    def test_simple_constructor_equality_operator(self):
        cdef THashSet[int] c1
        c1.insert(1)
        assert c1.size() == 1
        c1.insert(2)
        c1.insert(2)
        c1.insert(2)
        c1.insert(2)
        assert c1.size() == 2
        assert c1.contains(2)
        assert not c1.contains(5)
        cdef THashSet[int] c2 = c1
        assert c1 == c2
        c1.insert(3)
        assert c1 != c2
        c1.erase(3)
        assert c1 == c2

    def test_insert_erase(self):
        cdef THashSet[TString] tmp
        self.assertTrue(tmp.insert("one").second)
        self.assertFalse(tmp.insert("one").second)
        self.assertTrue(tmp.insert("two").second)
        cdef TString one = "one"
        cdef TString two = "two"
        self.assertEqual(tmp.erase(one), 1)
        self.assertEqual(tmp.erase(two), 1)
        self.assertEqual(tmp.size(), 0)
        self.assertTrue(tmp.empty())

    def test_iterators_and_find(self):
        cdef THashSet[TString] tmp
        self.assertTrue(tmp.begin() == tmp.end())
        self.assertTrue(tmp.find("1") == tmp.end())
        tmp.insert("1")
        self.assertTrue(tmp.begin() != tmp.end())
        cdef THashSet[TString].iterator it = tmp.find("1")
        self.assertTrue(it != tmp.end())
        self.assertEqual(deref(it), "1")

