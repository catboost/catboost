# cython: c_string_type=str, c_string_encoding=utf8

from util.string.cast cimport FromString, ToString

import unittest

class TestFromString(unittest.TestCase):
    def test_from_int(self):
        self.assertEqual(FromString[int]("42"), 42)

class TestToString(unittest.TestCase):
    def test_from_int(self):
        self.assertEqual(ToString(42), "42")
