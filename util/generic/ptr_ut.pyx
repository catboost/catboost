from util.generic.ptr cimport THolder
from util.generic.string cimport TString

import pytest
import unittest


class TestHolder(unittest.TestCase):

    def test_basic(self):
        cdef THolder[TString] holder
        holder.Reset(new TString("aaa"))
        assert holder.Get()[0] == "aaa"
        holder.Destroy()
        assert holder.Get() == NULL
        holder.Reset(new TString("bbb"))
        assert holder.Get()[0] == "bbb"
        holder.Reset(new TString("ccc"))
        assert holder.Get()[0] == "ccc"
