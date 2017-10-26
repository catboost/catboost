from util.string.cast cimport ToString

import unittest

class TestToString(unittest.TestCase):
    def test_from_int(self):
        self.assertEquals(ToString(42), "42")
