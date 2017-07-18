import collections
import re
import os

import _import_wrapper as iw
import _common as common

INCLUDE_PATTERN = re.compile(' *%(include|INCLUDE) *"([^"]*)"')


def get_retargeted(path, unit):
    return os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))


class Yasm(iw.CustomCommand):
    def __init__(self, path, unit):
        self._path = path
        self._flags = []

        prefix = unit.get('ASM_PREFIX')

        if prefix:
            self._flags += ['--prefix=' + prefix]

        self._incl_dirs = ['$S', '$B'] + unit.includes()
        self._pre_include = []

        flags = unit.get('YASM_FLAGS')
        if flags:
            self.parse_flags(path, unit, collections.deque(flags.split(' ')))

        if unit.enabled('darwin') or unit.enabled('ios'):
            self._platform = ['DARWIN', 'UNIX']
            self._fmt = 'macho'
        elif unit.enabled('win64') or unit.enabled('cygwin'):
            self._platform = ['WIN64']
            self._fmt = 'win'
        elif unit.enabled('win32'):
            self._platform = ['WIN32']
            self._fmt = 'win'
        else:
            self._platform = ['UNIX']
            self._fmt = 'elf'

        if 'elf' in self._fmt:
            self._flags += ['-g', 'dwarf2']

        self._fmt += unit.get('hardware_arch')
        self._type = unit.get('hardware_type')

    def parse_flags(self, path, unit, flags):
        while flags:
            flag = flags.popleft()
            if flag.startswith('-I'):
                raise Exception('Use ADDINCL macro')

            if flag.startswith('-P'):
                preinclude = flag[2:] or flags.popleft()
                self._pre_include += unit.resolve_include([(get_retargeted(path, unit)), preinclude])
                self._flags += ['-P', preinclude]
                continue

            self._flags.append(flag)

    def descr(self):
        return 'AS', self._path, 'light-green'

    def flags(self):
        return self._flags + self._platform + [self._fmt, self._type]

    def tools(self):
        return ['contrib/tools/yasm']

    def input(self):
        return common.make_tuples(self._pre_include + [self._path])

    def output(self):
        return common.make_tuples([common.tobuilddir(common.stripext(self._path)) + '.o'])

    def run(self, binary):
        return self.do_run(binary, self._path)

    def do_run(self, binary, path):
        def plt():
            for x in self._platform:
                yield '-D'
                yield x

        def incls():
            for x in self._incl_dirs:
                yield '-I'
                yield self.resolve_path(x)

        cmd = [binary, '-f', self._fmt] + list(plt()) + ['-D', '_' + self._type + '_', '-D_YASM_'] + self._flags + list(incls()) + ['-o', common.get(self.output, 0), path]
        self.call(cmd)


class YasmParser(object):
    def __init__(self, path, unit):
        with open(path) as f:
            includes = self.parse_includes(f)

        self._includes = unit.resolve_include([(get_retargeted(path, unit))] + includes) if includes else []

    @classmethod
    def parse_includes(cls, lines):
        includes = []
        for line in lines:
            m = INCLUDE_PATTERN.match(line)
            if m:
                includes.append(m.group(2))
        return includes

    def includes(self):
        return self._includes


def init():
    iw.addrule('asm', Yasm)
    iw.addparser('asm', YasmParser)


# ----------------Plugin tests------------------ #

def test_include_parser_1():
    swg_file = """
%macro EXPORT 1
    %ifdef DARWIN
        global _%1
        _%1:
    %else
        global %1
        %1:
    %endif
%endmacro

%include "include1.asm"
%INCLUDE "include2.asm"

%ifdef _x86_64_
    %include "context_x86_64.asm"
%else
    %include "context_i686.asm"
%endif

; Copyright (c) 2008-2013 GNU General Public License www.gnu.org/licenses
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; structure definition and constants:
%INCLUDE "randomah.asi"

global _MersRandomInit, _MersRandomInitByArray
"""

    includes = YasmParser.parse_includes(swg_file.split('\n'))
    assert includes == ['include1.asm', 'include2.asm', 'context_x86_64.asm', 'context_i686.asm', 'randomah.asi']
