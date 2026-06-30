from util.digest.multi cimport MultiHash
from util.generic.string cimport TString

import pytest
import unittest


class TestMultiHash(unittest.TestCase):

    def test_str_int(self):
        value = MultiHash(TString(b"1234567"), 123)
        self.assertEqual(value, 17038203285960021630)

    def test_int_str(self):
        value = MultiHash(123, TString(b"1234567"))
        self.assertEqual(value, 9973288649881090712)

    def test_collision(self):
        self.assertNotEqual(MultiHash(1, 1, 0), MultiHash(2, 2, 0))
