import re
import os

import _import_wrapper as iw
import _common as common


_SWIG_LIB_PATH = 'contrib/libs/swig/swig-2.0.9/Lib'

_PREDEFINED_INCLUDES = [
    os.path.join(_SWIG_LIB_PATH, 'python', 'python.swg'),
    os.path.join(_SWIG_LIB_PATH, 'swig.swg'),
]


class Swig(iw.CustomCommand):
    def __init__(self, path, unit):
        self._path = path
        self._flags = ['-cpperraswarn']

        self._bindir = common.tobuilddir(unit.path())
        self._input_name = common.stripext(os.path.basename(self._path))

        relpath = os.path.relpath(os.path.dirname(self._path), unit.path())
        self._main_out = os.path.join(self._bindir, '' if relpath == '.' else relpath.replace('..', '__'), self._input_name + '_wrap.c')

        if not path.endswith('.c.swg'):
            self._flags += ['-c++']
            self._main_out += 'pp'

        self._swig_lang = unit.get('SWIG_LANG')

        lang_specific_incl_dir = 'perl5' if self._swig_lang == 'perl' else self._swig_lang
        incl_dirs = ['bindings/swiglib', os.path.join(_SWIG_LIB_PATH, lang_specific_incl_dir), _SWIG_LIB_PATH, os.path.join(_SWIG_LIB_PATH, 'python')]
        self._incl_dirs = ['$S', '$B'] + ['$S/{}'.format(d) for d in incl_dirs]

        modname = unit.get('REALPRJNAME')
        self._flags.extend(['-module', modname])

        unit.onaddincl(incl_dirs)
        unit.onpython_addincl([])

        if self._swig_lang == 'python':
            self._out_name = modname + '.py'
            self._flags.extend(['-interface', unit.get('MODULE_PREFIX') + modname])

        if self._swig_lang == 'perl':
            self._out_name = modname + '.pm'
            self._flags.append('-shadow')
            unit.onpeerdir(['contrib/libs/perl-core'])

        if self._swig_lang == 'java':
            self._out_name = os.path.splitext(os.path.basename(self._path))[0] + '.jsrc'
            self._package = 'ru.yandex.' + os.path.dirname(self._path).replace('$S/', '').replace('$B/', '').replace('/', '.').replace('-', '_')
            unit.onpeerdir(['contrib/libs/jdk'])

        self._flags.append('-' + self._swig_lang)

    def descr(self):
        return 'SW', self._path, 'yellow'

    def flags(self):
        return self._flags

    def tools(self):
        return ['contrib/tools/swig']

    def input(self):
        return [
            (self._path, [])
        ]

    def output(self):
        return [
            (self._main_out, []),
            (common.join_intl_paths(self._bindir, self._out_name), (['noauto', 'add_to_outs'] if self._swig_lang != 'java' else [])),
        ]

    def run(self, binary):
        return self.do_run(binary, self._path) if self._swig_lang != 'java' else self.do_run_java(binary, self._path)

    def do_run(self, binary, path):
        self._incl_dirs = ['-I' + self.resolve_path(x) for x in self._incl_dirs]
        self.call([binary] + self._flags + ['-o', self.resolve_path(common.get(self.output, 0)), '-outdir', self.resolve_path(self._bindir)]
                  + self._incl_dirs + [self.resolve_path(path)])

    def do_run_java(self, binary, path):
        cmd = common.get_interpreter_path() + ['$S/build/scripts/run_swig_java.py', '--tool', binary, '--src', self.resolve_path(path)]
        for inc in self._incl_dirs:
            if inc:
                cmd += ['--flag', 'I' + self.resolve_path(inc)]
        cmd += ['--cpp-out', self._main_out, '--jsrc-out', '/'.join([self.resolve_path(self._bindir), self._out_name]), '--package', self._package]
        self.call(cmd)


class SwigParser(object):
    def __init__(self, path, unit):
        self._path = path
        retargeted = os.path.join(unit.path(), os.path.basename(path))
        with open(path, 'rb') as f:
            includes, induced = SwigParser.parse_includes(f.readlines())

        self._includes = unit.resolve_include([retargeted] + _PREDEFINED_INCLUDES + includes)
        self._induced = unit.resolve_include([retargeted] + induced) if induced else []

    @staticmethod
    def parse_includes(content):
        includes = []
        induced = []
        in_block = False

        include_pattern = re.compile('\%(include|import|insert *\([^\)]*\)) *("|<)([^">]*)')
        induced_pattern = re.compile('include *("|<)([^">]*)')

        for line in content:
            line = line.lstrip()
            if not in_block and line.startswith('%include') or line.startswith('%import') or line.startswith('%insert'):
                m = include_pattern.match(line)
                if m and m.group(3):
                    includes.append(m.group(3))

            elif line.startswith('%{'):
                in_block = True
            elif line.startswith('%begin') or line.startswith('%header'):
                if line.find('%{'):
                    in_block = True
            elif line.startswith('%}'):
                in_block = False
            elif in_block and line.startswith('#'):
                m = induced_pattern.search(line)
                if m and m.group(2):
                    induced.append(m.group(2))
        return (includes, induced)

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {
            'h+cpp': self._induced
        }


def init():
    iw.addrule('swg', Swig)
    iw.addparser('swg', SwigParser)


# ----------------Plugin test------------------ #
def test_include_parser():
    swg_file = """
%module lemmer

%include <typemaps.i>
%include "defaults.swg"

%header %{
#include <util/generic/string.h>
#include "util/generic/vector.h"
%}

    %import <contrib/swiglib/stroka.swg>

%{
    #include "py_api.h"
%}

%begin %{
    #pragma clang diagnostic ignored "-Wself-assign"
%}

%insert(init) "1.swg"
%insert(runtime) "2.swg";
%insert("init") "3.swg"
%insert("runtime") "4.swg";
%insert (klsdgjsdflksdf) "5.swg";
%insert   ("lkgjkdfg") "6.swg";

%insert   ("lkgjkdfg") "7.swg"; // )

%insert("header") {
}

%insert(runtime) %{
}%

"""

    includes, induced = SwigParser.parse_includes(swg_file.split('\n'))
    assert includes == ['typemaps.i', 'defaults.swg', 'contrib/swiglib/stroka.swg', '1.swg', '2.swg', '3.swg', '4.swg', '5.swg', '6.swg', '7.swg']
    assert induced == ['util/generic/string.h', 'util/generic/vector.h', 'py_api.h']
