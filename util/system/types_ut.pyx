from util.system.types cimport i8, i16, i32, i64
from util.system.types cimport ui8, ui16, ui32, ui64

import pytest
import unittest


class TestTypes(unittest.TestCase):
    def test_i8(self):
        cdef i8 value = 42
        self.assertEqual(sizeof(value), 1)

    def test_ui8(self):
        cdef ui8 value = 42
        self.assertEqual(sizeof(value), 1)

    def test_i16(self):
        cdef i16 value = 42
        self.assertEqual(sizeof(value), 2)

    def test_ui16(self):
        cdef ui16 value = 42
        self.assertEqual(sizeof(value), 2)

    def test_i32(self):
        cdef i32 value = 42
        self.assertEqual(sizeof(value), 4)

    def test_ui32(self):
        cdef ui32 value = 42
        self.assertEqual(sizeof(value), 4)

    def test_i64(self):
        cdef i64 value = 42
        self.assertEqual(sizeof(value), 8)

    def test_ui64(self):
        cdef ui64 value = 42
        self.assertEqual(sizeof(value), 8)

