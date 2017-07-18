from util.generic.maybe cimport TMaybe, Nothing

import pytest
import unittest


class TestMaybe(unittest.TestCase):

    def test_ctor1(self):
        cdef TMaybe[int] tmp = TMaybe[int]()
        self.assertFalse(tmp.Defined())

    def test_ctor2(self):
        cdef TMaybe[int] tmp = TMaybe[int](42)
        self.assertTrue(tmp.Defined())
        self.assertEquals(tmp.GetRef(), 42)

    def test_ctor3(self):
        cdef TMaybe[int] tmp = Nothing()
        self.assertFalse(tmp.Defined())

    def test_operator_assign(self):
        cdef TMaybe[int] tmp
        tmp = 42
        self.assertTrue(tmp.Defined())
        self.assertEquals(tmp.GetRef(), 42)

    def test_compare(self):
        cdef TMaybe[int] tmp1 = 17
        cdef TMaybe[int] tmp2 = 42
        cdef TMaybe[int] nothing

        # ==
        self.assertTrue(tmp1 == 17)
        self.assertTrue(tmp1 == tmp1)
        self.assertTrue(nothing == nothing)

        self.assertFalse(tmp1 == 16)
        self.assertFalse(tmp1 == tmp2)
        self.assertFalse(tmp1 == nothing)

        # !=
        self.assertTrue(tmp1 != 16)
        self.assertTrue(tmp1 != tmp2)
        self.assertTrue(tmp1 != nothing)

        self.assertFalse(tmp1 != 17)
        self.assertFalse(tmp1 != tmp1)
        self.assertFalse(nothing != nothing)

        # <
        self.assertTrue(nothing < tmp1)
        self.assertTrue(nothing < tmp2)
        self.assertTrue(tmp1 < tmp2)
        self.assertTrue(nothing < 0)
        self.assertTrue(tmp1 < 18)

        self.assertFalse(nothing < nothing)
        self.assertFalse(tmp1 < tmp1)
        self.assertFalse(tmp2 < tmp1)
        self.assertFalse(tmp1 < 16)

        # <=
        self.assertTrue(nothing <= nothing)
        self.assertTrue(nothing <= tmp1)
        self.assertTrue(nothing <= tmp2)
        self.assertTrue(tmp1 <= tmp1)
        self.assertTrue(tmp1 <= tmp2)
        self.assertTrue(nothing <= 0)
        self.assertTrue(tmp1 <= 18)

        self.assertFalse(tmp2 <= tmp1)
        self.assertFalse(tmp1 <= 16)

        # >
        self.assertTrue(tmp1 > nothing)
        self.assertTrue(tmp2 > nothing)
        self.assertTrue(tmp2 > tmp1)
        self.assertTrue(tmp1 > 16)

        self.assertFalse(nothing > nothing)
        self.assertFalse(nothing > 0)
        self.assertFalse(tmp1 > tmp1)
        self.assertFalse(tmp1 > tmp2)
        self.assertFalse(tmp1 > 18)

        # >=
        self.assertTrue(nothing >= nothing)
        self.assertTrue(tmp1 >= nothing)
        self.assertTrue(tmp2 >= nothing)
        self.assertTrue(tmp2 >= tmp1)
        self.assertTrue(tmp1 >= tmp1)
        self.assertTrue(tmp1 >= 16)

        self.assertFalse(nothing >= 0)
        self.assertFalse(tmp1 >= tmp2)
        self.assertFalse(tmp1 >= 18)

    def test_construct_in_place(self):
        cdef TMaybe[int] tmp
        tmp.ConstructInPlace(42)
        self.assertTrue(tmp.Defined())
        self.assertEquals(tmp.GetRef(), 42)

    def test_clear(self):
        cdef TMaybe[int] tmp = 42
        tmp.Clear()
        self.assertFalse(tmp.Defined())

    def test_defined(self):
        cdef TMaybe[int] tmp
        self.assertFalse(tmp.Defined())
        self.assertTrue(tmp.Empty())
        tmp = 42
        self.assertTrue(tmp.Defined())
        self.assertFalse(tmp.Empty())

    def test_check_defined(self):
        cdef TMaybe[int] tmp
        with pytest.raises(RuntimeError):
            tmp.CheckDefined()
        tmp = 42
        tmp.CheckDefined()

    def test_get(self):
        cdef TMaybe[int] tmp = 42
        cdef int* p = tmp.Get()
        self.assertTrue(p != NULL)
        self.assertEquals(p[0], 42)

    def test_get_ref(self):
        cdef TMaybe[int] tmp = 42
        self.assertTrue(tmp.Defined())
        self.assertEquals(tmp.GetRef(), 42)

    def test_get_or_else(self):
        cdef TMaybe[int] tmp = 42
        self.assertEquals(tmp.GetOrElse(13), 42)
        tmp.Clear()
        self.assertEquals(tmp.GetOrElse(13), 13)

    def test_or_else(self):
        cdef TMaybe[int] tmp = 42
        cdef TMaybe[int] nothing
        self.assertFalse(nothing.OrElse(nothing).Defined())
        self.assertEquals(tmp.OrElse(nothing).GetRef(), 42)
        self.assertEquals(nothing.OrElse(tmp).GetRef(), 42)
        self.assertEquals(tmp.OrElse(tmp).GetRef(), 42)

    def test_cast(self):
        cdef TMaybe[int] tmp = 42
        cdef TMaybe[char] tmp2 = tmp.Cast[char]()
        self.assertEquals(tmp2.GetRef(), 42)

    def test_swap(self):
        cdef TMaybe[int] tmp1 = 42
        cdef TMaybe[int] tmp2
        tmp2.Swap(tmp1)
        self.assertFalse(tmp1.Defined())
        self.assertEquals(tmp2.GetRef(), 42)
