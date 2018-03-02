from build.plugins import pyx


def test_include_parser():
    pyx_file = """
include "arc_lang.pxi"
  include "new.pxi"

include 'parameters.pxi'

cdef extern from "util/generic/string.h":
    ctypedef string TString

cdef extern from "<urlutils.h>" nogil:
    TString NormalizeUrl(const TString& keyStr) nogil except+

#cdef extern from "commented.h" nogil:

cdef extern from "noop1.h #no quote et the end
include "noop2.h

cimport   cares #comment

from zmq.core.context cimport Context as _original_Context

cimport cython

from libcpp.string cimport *

from lxml.includes.etreepublic  cimport    elementFactory,  import_lxml__etree, textOf, pyunicode #comment

snippet cimport

cdef extern from 'factors.h':

cimport util.generic.maybe
from util.generic.vector cimport TVector

cimport some.name1, name2, other.path.name3
cimport xlibrary as xl, some.path.ylibrary as yl, other.zlibrary as zl
"""
    includes, induced, susp_includes = pyx.PyxParser.parse_includes(pyx_file.split('\n'))
    assert includes == pyx.PyxParser.get_perm_includes() + [
        'arc_lang.pxi',
        'new.pxi',
        'parameters.pxi',
        'cares.pxd',
        'util/generic/maybe.pxd',
        'some/name1.pxd',
        'name2.pxd',
        'other/path/name3.pxd',
        'xlibrary.pxd',
        'some/path/ylibrary.pxd',
        'other/zlibrary.pxd',
    ]
    assert induced == [
        'util/generic/string.h',
        'urlutils.h',
        'factors.h',
    ]
    assert susp_includes == [
        ('zmq/core/context', ['Context']),
        ('lxml/includes/etreepublic', ['elementFactory', 'import_lxml__etree', 'textOf', 'pyunicode']),
        ('util/generic/vector', ['TVector']),
    ]
