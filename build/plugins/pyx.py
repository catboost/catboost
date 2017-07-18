import os
import re

import _common
import _import_wrapper as iw


INCLUDE_PATTERN = re.compile('include *"([^"]+)"')
CIMPORT_PATTERN = re.compile('cimport +([^#]+)')
FROM_CIMPORT_PATTERN = re.compile('from +(\S+) +cimport +([^#]+)')
INDUCED_PATTERN = re.compile('cdef +extern +from +["\']<?([^">]+)>?["\']')


def find_init_files(files, unit):
    traversed = set()
    for include in files:
        path = os.path.dirname(include)
        while path and path not in traversed:
            traversed.add(path)
            init_file = _common.join_intl_paths(path, '__init__.py')
            path = os.path.dirname(path)
            if os.path.isfile(unit.resolve(_common.join_intl_paths('$S', init_file))):
                yield init_file


class PyxParser(object):
    def __init__(self, path, unit):
        self._path = path
        self._includes = []
        retargeted = os.path.join(unit.path(), os.path.basename(path))

        with open(path, 'rb') as f:
            includes, induced, susp_includes = self.parse_includes(f.readlines())

            for susp_incl in susp_includes:
                incl_path = unit.resolve(os.path.join(unit.path(), susp_incl[0]))
                if not os.path.isdir(incl_path):
                    includes.append(susp_incl[0] + '.pxd')
                else:
                    for f in susp_incl[1]:
                        if f != '*':
                            includes.append(susp_incl[0] + '/' + f + '.pxd')

        self._includes = unit.resolve_include([retargeted] + includes + list(find_init_files(includes, unit))) if includes else []
        self._induced = unit.resolve_include([retargeted] + induced + list(find_init_files(induced, unit))) if induced else []

    @staticmethod
    def get_perm_includes():
        where = 'contrib/tools/cython/Cython/Utility'

        includes = [
            'Buffer.c',
            'CommonTypes.c',
            'Exceptions.c',
            'Generator.c',
            'ModuleSetupCode.c',
            'Overflow.c',
            'StringTools.c',
            'Builtins.c',
            'CythonFunction.c',
            'ExtensionTypes.c',
            'ImportExport.c',
            'ObjectHandling.c',
            'Printing.c',
            'TestUtilityLoader.c',
            'Capsule.c',
            'Embed.c',
            'FunctionArguments.c',
            'MemoryView_C.c',
            'Optimize.c',
            'Profile.c',
            'TypeConversion.c',
        ]
        return [where + '/' + x for x in includes]

    @staticmethod
    def parse_includes(content):
        includes = PyxParser.get_perm_includes()
        induced = []
        susp_includes = []

        for line in content:
            line = line.lstrip()
            incl = INCLUDE_PATTERN.match(line)
            if incl:
                if incl.group(1):
                    includes.append(incl.group(1))
            else:
                ind = INDUCED_PATTERN.match(line)
                if ind and ind.group(1):
                    induced.append(ind.group(1))
                else:
                    def filter_known_inner_paths(p):
                        # XXX: cpython is here but need to be added in PEERDIRs|ADDINCLs of cython project and * supported somehow
                        return p and p.split('.')[0] not in ('libc', 'libcpp', 'cython', 'cpython')

                    cimport = CIMPORT_PATTERN.match(line)
                    if cimport:
                        cimport_files = cimport.group(1)
                        cimport_files = [x.strip() for x in cimport_files.split(',')]
                        cimport_files = [x.split(' ')[0] for x in cimport_files]
                        for cimport_file in cimport_files:
                            if filter_known_inner_paths(cimport_file):
                                includes.append(cimport_file.replace('.', '/') + '.pxd')
                    else:
                        from_cimport = FROM_CIMPORT_PATTERN.match(line)
                        if from_cimport:
                            cimport_source = from_cimport.group(1)
                            cimport_symbols = from_cimport.group(2) or ''
                            cimport_symbols = [x.strip() for x in cimport_symbols.split(',')]
                            cimport_symbols = [x.split(' ')[0] for x in cimport_symbols]
                            if filter_known_inner_paths(cimport_source):
                                susp_includes.append((cimport_source.replace('.', '/'), cimport_symbols))

        return includes, induced, susp_includes

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {
            'cpp': ['$S/contrib/tools/python/src/Include/Python.h'] + self._induced
        }


def init():
    iw.addparser('pyx', PyxParser, pass_induced_includes=True)
    iw.addparser('pxd', PyxParser, pass_induced_includes=True)
    iw.addparser('pxi', PyxParser, pass_induced_includes=True)


# ----------------Plugin test------------------ #
def test_include_parser():
    pyx_file = """
include "arc_lang.pxi"
  include "new.pxi"

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
from util.generic.vector cimport yvector

cimport some.name1, name2, other.path.name3
cimport xlibrary as xl, some.path.ylibrary as yl, other.zlibrary as zl
"""
    includes, induced, susp_includes = PyxParser.parse_includes(pyx_file.split('\n'))
    assert includes == PyxParser.get_perm_includes() + [
        'arc_lang.pxi',
        'new.pxi',
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
        ('util/generic/vector', ['yvector']),
    ]
