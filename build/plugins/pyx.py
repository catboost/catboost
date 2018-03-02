import os
import re

import _common
import _import_wrapper as iw


INCLUDE_PATTERN = re.compile('(include *"([^"]+)")|(include *\'([^\']+)\')')
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
        self._induced = []
        retargeted = os.path.join(unit.path(), os.path.basename(path))

        if path.endswith('.pyx'):
            pxd = path[:-4] + '.pxd'
            if os.path.exists(pxd):
                self._includes.append(unit.resolve_arc_path(pxd))

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

        if includes:
            self._includes += unit.resolve_include([retargeted] + includes + list(find_init_files(includes, unit)))

        if induced:
            self._induced += unit.resolve_include([retargeted] + induced + list(find_init_files(induced, unit)))

    @staticmethod
    def get_perm_includes():
        where = 'contrib/tools/cython/Cython/Utility'

        includes = [
            'Buffer.c',
            'Builtins.c',
            'CMath.c',
            'Capsule.c',
            'CommonTypes.c',
            'Complex.c',
            'Coroutine.c',
            'CythonFunction.c',
            'Embed.c',
            'Exceptions.c',
            'ExtensionTypes.c',
            'FunctionArguments.c',
            'ImportExport.c',
            'MemoryView_C.c',
            'ModuleSetupCode.c',
            'ObjectHandling.c',
            'Optimize.c',
            'Overflow.c',
            'Printing.c',
            'Profile.c',
            'StringTools.c',
            'TestUtilityLoader.c',
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
                incl_value = incl.group(2) or incl.group(4)
                if incl_value:
                    includes.append(incl_value)
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
