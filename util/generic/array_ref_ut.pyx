import pytest
import unittest
from util.generic.array_ref cimport TArrayRef
from util.generic.vector cimport TVector


class TestArrayRef(unittest.TestCase):
    def test_array_data_reference(self):
        array_size = 30
        cdef TVector[int] vec
        for i in xrange(array_size):
            vec.push_back(i)
        cdef TArrayRef[int] array_ref = TArrayRef[int](vec.data(), vec.size())
        for i in xrange(array_size / 2):
            array_ref[array_size - 1 - i] = array_ref[i]
        for i in xrange(array_size):
            self.assertEqual(array_ref[i], array_size - 1 - i)

    def test_array_vec_reference(self):
        array_size = 30
        cdef TVector[int] vec
        for i in xrange(array_size):
            vec.push_back(i)
        cdef TArrayRef[int] array_ref = TArrayRef[int](vec)
        for i in xrange(array_size / 2):
            array_ref[array_size - 1 - i] = array_ref[i]
        for i in xrange(array_size):
            self.assertEqual(array_ref[i], array_size - 1 - i)