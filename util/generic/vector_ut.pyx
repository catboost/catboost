from util.generic.vector cimport yvector

import pytest
import unittest


class TestVector(unittest.TestCase):

    def test_ctor1(self):
        cdef yvector[int] tmp = yvector[int]()
        self.assertEqual(tmp.size(), 0)

    def test_ctor2(self):
        cdef yvector[int] tmp = yvector[int](10)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp[0], 0)

    def test_ctor3(self):
        cdef yvector[int] tmp = yvector[int](10, 42)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp[0], 42)

    def test_ctor4(self):
        cdef yvector[int] tmp = yvector[int](10, 42)
        cdef yvector[int] tmp2 = yvector[int](tmp)
        self.assertEqual(tmp2.size(), 10)
        self.assertEqual(tmp2[0], 42)

    def test_operator_assign(self):
        cdef yvector[int] tmp2
        tmp2.push_back(1)
        tmp2.push_back(2)

        cdef yvector[int] tmp3
        tmp3.push_back(1)
        tmp3.push_back(3)

        tmp3 = tmp2

    def test_compare(self):
        cdef yvector[int] tmp1
        tmp1.push_back(1)
        tmp1.push_back(2)

        cdef yvector[int] tmp2
        tmp2.push_back(1)
        tmp2.push_back(2)

        cdef yvector[int] tmp3
        tmp3.push_back(1)
        tmp3.push_back(3)

        self.assertTrue(tmp1 == tmp2)
        self.assertTrue(tmp1 != tmp3)

        self.assertTrue(tmp1 < tmp3)
        self.assertTrue(tmp1 <= tmp3)

        self.assertTrue(tmp3 > tmp1)
        self.assertTrue(tmp3 >= tmp1)

    def test_index(self):
        cdef yvector[int] tmp = yvector[int](10, 42)

        self.assertEqual(tmp[0], 42)
        self.assertEqual(tmp[5], 42)

        self.assertEqual(tmp.data()[0], 42)
        self.assertEqual(tmp.data()[5], 42)

        self.assertEqual(tmp.at(0), 42)
        self.assertEqual(tmp.at(5), 42)

        with pytest.raises(IndexError):
            tmp.at(100)

    def test_push_pop_back(self):
        cdef yvector[int] tmp
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
        cdef yvector[int] tmp
        tmp.push_back(42)
        self.assertEqual(tmp.front(), 42)

    def test_empty(self):
        cdef yvector[int] tmp
        self.assertTrue(tmp.empty())
        tmp.push_back(42)
        self.assertFalse(tmp.empty())

    def test_max_size(self):
        cdef yvector[int] tmp
        self.assertTrue(tmp.max_size() > 0)

    def test_reserve_resize(self):
        cdef yvector[int] tmp
        tmp.reserve(1000)
        self.assertEqual(tmp.capacity(), 1000)

        tmp.resize(100)
        self.assertEqual(tmp.size(), 100)
        self.assertEqual(tmp.front(), 0)
        self.assertEqual(tmp.back(), 0)

        tmp.shrink_to_fit()
        tmp.clear()

        tmp.resize(100, 42)
        self.assertEqual(tmp.size(), 100)
        self.assertEqual(tmp.front(), 42)
        self.assertEqual(tmp.back(), 42)

    def test_iter(self):
        cdef yvector[int] tmp
        tmp.push_back(1)
        tmp.push_back(20)
        tmp.push_back(300)

        self.assertEqual([i for i in tmp], [1, 20, 300])

    def test_iterator(self):
        cdef yvector[int] tmp

        self.assertTrue(tmp.begin() == tmp.end())
        self.assertTrue(tmp.rbegin() == tmp.rend())
        self.assertTrue(tmp.const_begin() == tmp.const_end())
        self.assertTrue(tmp.const_rbegin() == tmp.const_rend())

        tmp.push_back(1)

        self.assertTrue(tmp.begin() != tmp.end())
        self.assertTrue(tmp.rbegin() != tmp.rend())
        self.assertTrue(tmp.const_begin() != tmp.const_end())
        self.assertTrue(tmp.const_rbegin() != tmp.const_rend())

        self.assertTrue(tmp.begin() < tmp.end())
        self.assertTrue(tmp.rbegin() < tmp.rend())
        self.assertTrue(tmp.const_begin() < tmp.const_end())
        self.assertTrue(tmp.const_rbegin() < tmp.const_rend())

        self.assertTrue(tmp.begin() + 1 == tmp.end())
        self.assertTrue(tmp.rbegin() + 1 == tmp.rend())
        self.assertTrue(tmp.const_begin() + 1 == tmp.const_end())
        self.assertTrue(tmp.const_rbegin() + 1 == tmp.const_rend())

    def test_assign(self):
        cdef yvector[int] tmp

        tmp.assign(10, 42)
        self.assertEqual(tmp.size(), 10)
        self.assertEqual(tmp.front(), 42)
        self.assertEqual(tmp.back(), 42)

    def test_insert(self):
        cdef yvector[int] tmp
        tmp.push_back(1)
        tmp.push_back(2)
        tmp.push_back(3)

        cdef yvector[int] tmp2
        tmp2.push_back(7)
        tmp2.push_back(9)

        tmp.insert(tmp.begin(), 8)
        self.assertEqual([i for i in tmp], [8, 1, 2, 3])

        tmp.insert(tmp.begin(), 2, 6)
        self.assertEqual([i for i in tmp], [6, 6, 8, 1, 2, 3])

        tmp.insert(tmp.begin(), tmp2.begin(), tmp2.end())
        self.assertEqual([i for i in tmp], [7, 9, 6, 6, 8, 1, 2, 3])

    def test_erase(self):
        cdef yvector[int] tmp
        tmp.push_back(1)
        tmp.push_back(2)
        tmp.push_back(3)
        tmp.push_back(4)

        tmp.erase(tmp.begin() + 1)
        self.assertEqual([i for i in tmp], [1, 3, 4])

        tmp.erase(tmp.begin(), tmp.begin() + 2)
        self.assertEqual([i for i in tmp], [4])
