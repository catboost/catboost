from util.generic.hash cimport yhash
from util.generic.string cimport TString

import pytest
import unittest

from libcpp.pair cimport pair
from cython.operator cimport dereference as deref


class TestHash(unittest.TestCase):

    def test_constructors_and_assignments(self):
        cdef yhash[TString, int] c1
        c1["one"] = 1
        c1["two"] = 2
        cdef yhash[TString, int] c2 = yhash[TString, int](c1)
        self.assertEqual(2, c1.size())
        self.assertEqual(2, c2.size())
        self.assertEqual(1, c1.at("one"))
        self.assertTrue(c1.has("two"))
        self.assertTrue(c2.has("one"))
        self.assertEqual(2, c2.at("two"))
        c2["three"] = 3
        c1 = c2
        self.assertEqual(3, c1.size())
        self.assertEqual(3, c2.size())
        self.assertEqual(3, c1.at("three"))

    def test_equality_operator(self):
        cdef yhash[TString, int] base
        base["one"] = 1
        base["two"] = 2

        cdef yhash[TString, int] c1 = yhash[TString, int](base)
        self.assertTrue(c1==base)

        cdef yhash[TString, int] c2
        c2["one"] = 1
        c2["two"] = 2
        self.assertTrue(c2 == base)

        c2["three"] = 3
        self.assertTrue(c2 != base)

        cdef yhash[TString, int] c3 = yhash[TString, int](base)
        c3["one"] = 0
        self.assertTrue(c3 != base)

    def test_insert_erase(self):
        cdef yhash[TString, int] tmp
        self.assertTrue(tmp.insert(pair[TString, int]("one", 0)).second)
        self.assertFalse(tmp.insert(pair[TString, int]("one", 1)).second)
        self.assertTrue(tmp.insert(pair[TString, int]("two", 2)).second)
        cdef TString one = "one"
        cdef TString two = "two"
        self.assertEqual(tmp.erase(one), 1)
        self.assertEqual(tmp.erase(two), 1)
        self.assertEqual(tmp.size(), 0)
        self.assertTrue(tmp.empty())

    def test_iterators_and_find(self):
        cdef yhash[TString, int] tmp
        self.assertTrue(tmp.begin() == tmp.end())
        self.assertTrue(tmp.find("1") == tmp.end())
        tmp["1"] = 1
        self.assertTrue(tmp.begin() != tmp.end())
        cdef yhash[TString, int].iterator it = tmp.find("1")
        self.assertTrue(it != tmp.end())
        self.assertEqual(deref(it).second, 1)

