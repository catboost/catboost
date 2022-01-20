from libcpp.utility cimport pair
from util.generic.ptr cimport MakeAtomicShared, TAtomicSharedPtr, THolder
from util.generic.string cimport TString
from util.system.types cimport ui64

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

    def test_make_atomic_shared(self):
        cdef TAtomicSharedPtr[pair[ui64, TString]] atomic_shared_ptr = MakeAtomicShared[pair[ui64, TString]](15, "Some string")
        assert atomic_shared_ptr.Get().first == 15
        assert atomic_shared_ptr.Get().second == "Some string"
