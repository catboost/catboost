from util.generic.list cimport TList

import unittest
from cython.operator cimport preincrement


class TestList(unittest.TestCase):

    def test_ctor1(self):
        cdef TList[int] tmp = TList[int]()
        self.assertEqual(tmp.size(), 0)

    def test_ctor2(self):
        cdef TList[int] tmp = TList[int](10, 42)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp.front(), 42)

    def test_ctor3(self):
        cdef TList[int] tmp = TList[int](10, 42)
        cdef TList[int] tmp2 = TList[int](tmp)
        self.assertEqual(tmp2.size(), 10)
        self.assertEqual(tmp2.front(), 42)

    def test_operator_assign(self):
        cdef TList[int] tmp2
        tmp2.push_back(1)
        tmp2.push_back(2)

        cdef TList[int] tmp3
        tmp3.push_back(1)
        tmp3.push_back(3)

        tmp3 = tmp2

    def test_compare(self):
        cdef TList[int] tmp1
        tmp1.push_back(1)
        tmp1.push_back(2)

        cdef TList[int] tmp2
        tmp2.push_back(1)
        tmp2.push_back(2)

        cdef TList[int] tmp3
        tmp3.push_back(1)
        tmp3.push_back(3)

        self.assertTrue(tmp1 == tmp2)
        self.assertTrue(tmp1 != tmp3)

        self.assertTrue(tmp1 < tmp3)
        self.assertTrue(tmp1 <= tmp3)

        self.assertTrue(tmp3 > tmp1)
        self.assertTrue(tmp3 >= tmp1)

    def test_push_pop_back(self):
        cdef TList[int] tmp
        self.assertEqual(tmp.size(), 0)

        tmp.push_back(42)
        self.assertEqual(tmp.size(), 1)
        self.assertEqual(tmp.back(), 42)

        tmp.push_back(77)
        self.assertEqual(tmp.size(), 2)
        self.assertEqual(tmp.back(), 77)

        tmp.pop_back()
        self.assertEqual(tmp.size(), 1)
        self.assertEqual(tmp.back(), 42)

        tmp.pop_back()
        self.assertEqual(tmp.size(), 0)

    def test_front(self):
        cdef TList[int] tmp
        tmp.push_back(42)
        self.assertEqual(tmp.front(), 42)

    def test_empty(self):
        cdef TList[int] tmp
        self.assertTrue(tmp.empty())
        tmp.push_back(42)
        self.assertFalse(tmp.empty())

    def test_max_size(self):
        cdef TList[int] tmp
        self.assertTrue(tmp.max_size() > 0)

    def test_resize(self):
        cdef TList[int] tmp

        tmp.resize(100, 42)
        self.assertEqual(tmp.size(), 100)
        self.assertEqual(tmp.front(), 42)
        self.assertEqual(tmp.back(), 42)

    def test_iter(self):
        cdef TList[int] tmp
        tmp.push_back(1)
        tmp.push_back(20)
        tmp.push_back(300)

        self.assertEqual([i for i in tmp], [1, 20, 300])

    def test_iterator(self):
        cdef TList[int] tmp

        self.assertTrue(tmp.begin() == tmp.end())
        self.assertTrue(tmp.rbegin() == tmp.rend())
        self.assertTrue(tmp.const_begin() == tmp.const_end())
        self.assertTrue(tmp.const_rbegin() == tmp.const_rend())

        tmp.push_back(1)

        self.assertTrue(tmp.begin() != tmp.end())
        self.assertTrue(tmp.rbegin() != tmp.rend())
        self.assertTrue(tmp.const_begin() != tmp.const_end())
        self.assertTrue(tmp.const_rbegin() != tmp.const_rend())

        self.assertTrue(preincrement(tmp.begin()) == tmp.end())
        self.assertTrue(preincrement(tmp.rbegin()) == tmp.rend())
        self.assertTrue(preincrement(tmp.const_begin()) == tmp.const_end())
        self.assertTrue(preincrement(tmp.const_rbegin()) == tmp.const_rend())

    def test_assign(self):
        cdef TList[int] tmp

        tmp.assign(10, 42)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp.front(), 42)
        self.assertEqual(tmp.back(), 42)

    def test_insert(self):
        cdef TList[int] tmp
        tmp.push_back(1)
        tmp.push_back(2)
        tmp.push_back(3)

        tmp.insert(tmp.begin(), 8)
        self.assertEqual([i for i in tmp], [8, 1, 2, 3])

        tmp.insert(tmp.begin(), 2, 6)
        self.assertEqual([i for i in tmp], [6, 6, 8, 1, 2, 3])

    def test_erase(self):
        cdef TList[int] tmp
        tmp.push_back(1)
        tmp.push_back(2)
        tmp.push_back(3)
        tmp.push_back(4)

        tmp.erase(preincrement(tmp.begin()))
        self.assertEqual([i for i in tmp], [1, 3, 4])

        tmp.erase(tmp.begin(), preincrement(preincrement(tmp.begin())))
        self.assertEqual([i for i in tmp], [4])
