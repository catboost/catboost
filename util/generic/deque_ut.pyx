from libcpp.deque cimport deque
from util.generic.deque cimport TDeque

import pytest
import unittest


class TestDeque(unittest.TestCase):
    def test_ctor1(self):
        cdef TDeque[int] tmp = TDeque[int]()
        self.assertEqual(tmp.size(), 0)

    def test_ctor2(self):
        cdef TDeque[int] tmp = TDeque[int](10)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp[0], 0)

    def test_ctor3(self):
        cdef TDeque[int] tmp = TDeque[int](10, 42)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp[0], 42)

    def test_ctor4(self):
        cdef TDeque[int] tmp = TDeque[int](10, 42)
        cdef TDeque[int] tmp2 = TDeque[int](tmp)
        self.assertEqual(tmp2.size(), 10)
        self.assertEqual(tmp2[0], 42)

    def test_operator_assign(self):
        cdef TDeque[int] tmp2
        tmp2.push_back(1)
        tmp2.push_back(2)

        cdef TDeque[int] tmp3
        tmp3.push_back(1)
        tmp3.push_back(3)

        self.assertEqual(tmp2[1], 2)
        self.assertEqual(tmp3[1], 3)

        tmp3 = tmp2

        self.assertEqual(tmp2[1], 2)
        self.assertEqual(tmp3[1], 2)

    def test_compare(self):
        cdef TDeque[int] tmp1
        tmp1.push_back(1)
        tmp1.push_back(2)

        cdef TDeque[int] tmp2
        tmp2.push_back(1)
        tmp2.push_back(2)

        cdef TDeque[int] tmp3
        tmp3.push_back(1)
        tmp3.push_back(3)

        self.assertTrue(tmp1 == tmp2)
        self.assertTrue(tmp1 != tmp3)

        self.assertTrue(tmp1 < tmp3)
        self.assertTrue(tmp1 <= tmp3)

        self.assertTrue(tmp3 > tmp1)
        self.assertTrue(tmp3 >= tmp1)