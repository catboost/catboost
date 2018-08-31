from util.string.cast cimport FromString, ToString

import unittest

class TestFromString(unittest.TestCase):
    def test_from_int(self):
        self.assertEquals(FromString[int]("42"), 42)

class TestToString(unittest.TestCase):
    def test_from_int(self):
        self.assertEquals(ToString(42), "42")
