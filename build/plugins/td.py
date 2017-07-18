import re
import os

import _import_wrapper as iw

INCLUDE_PATTERN = re.compile('include *"([^"]*)"')


class TableGenParser(object):
    def __init__(self, path, unit):
        self._path = path

        with open(path, 'r') as f:
            includes = list(self.parse_includes(f))

        retargeted = os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))
        self._includes = unit.resolve_include([retargeted] + includes) if includes else []

    @staticmethod
    def parse_includes(lines):
        for line in lines:
            m = INCLUDE_PATTERN.match(line)
            if m:
                yield m.group(1)

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {}


def init():
    iw.addparser('td', TableGenParser)


# ----------------Plugin test------------------ #
def test_include_parser():
    text = '''
//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

include "X86InstrFormats.td"

include "X86InstrExtension.td"

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Pattern fragments.
//

// X86 specific condition code. These correspond to CondCode in
// X86InstrInfo.h. They must be kept in synch.
def X86_COND_A   : PatLeaf<(i8 0)>;  // alt. COND_NBE
def X86_COND_AE  : PatLeaf<(i8 1)>;  // alt. COND_NC
def X86_COND_B   : PatLeaf<(i8 2)>;  // alt. COND_C
'''
    includes = list(TableGenParser.parse_includes(text.split('\n')))
    assert includes == [
        "X86InstrFormats.td",
        "X86InstrExtension.td",
        "llvm/Target/Target.td",
    ]
