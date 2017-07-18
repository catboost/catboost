import re
import os

import _import_wrapper as iw
import _common as common

INCLUDE_PATTERN = re.compile('include *"([^"]*)";')


class Flatc(iw.CustomCommand):

    def __init__(self, path, unit):
        self._path = path
        self._incl_dirs = ['$S', '$B']

    def descr(self):
        return 'FL', self._path, 'light-green'

    def tools(self):
        return ['contrib/tools/flatc']

    def input(self):
        return common.make_tuples([self._path, '$S/build/scripts/stdout2stderr.py'])

    def output(self):
        return common.make_tuples([common.tobuilddir(common.stripext(self._path)) + '.fbs.h'])

    def run(self, binary):
        return self.do_run(binary, self._path)

    def do_run(self, binary, path):
        def incls():
            for x in self._incl_dirs:
                yield '-I'
                yield self.resolve_path(x)
        output_dir = os.path.dirname(self.resolve_path(common.get(self.output, 0)))
        cmd = common.get_interpreter_path() + ['$S/build/scripts/stdout2stderr.py', binary, '--cpp'] + list(incls()) + ['-o', output_dir, path]
        self.call(cmd)


class FlatcParser(object):

    def __init__(self, path, unit):
        self._path = path
        retargeted = os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))

        with open(path, 'r') as f:
            includes, induced = FlatcParser.parse_includes(f.readlines())

        induced.append('contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h')

        self._includes = unit.resolve_include([retargeted] + includes) if includes else []
        self._induced = unit.resolve_include([retargeted] + induced) if induced else []

    @staticmethod
    def parse_includes(lines):
        includes = []
        induced = []

        for line in lines:
            m = INCLUDE_PATTERN.match(line)

            if m:
                incl = m.group(1)
                includes.append(incl)
                induced.append(common.stripext(incl) + '.fbs.h')

        return includes, induced

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {'h': self._induced}


def init():
    iw.addrule('fbs', Flatc)
    iw.addparser('fbs', FlatcParser)


# ----------------Plugin test------------------ #
def test_include_parser():
    text = '''
aaaaa
include "incl1.fbs";
include   "incl2.fbs";
// include "none.fbs";
'''
    includes, induced = FlatcParser.parse_includes(text.split('\n'))
    assert includes == ['incl1.fbs', 'incl2.fbs', ]
    assert induced == ['incl1.fbs.h', 'incl2.fbs.h', ]
