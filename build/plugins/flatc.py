import re
import os

import _import_wrapper as iw
import _common as common

INCLUDE_PATTERN = re.compile('include *"([^"]*)";')


class FlatcBase(iw.CustomCommand):

    def __init__(self, path, unit):
        self._path = path
        self._incl_dirs = ['$S', '$B']

    def input(self):
        return common.make_tuples([self._path, '$S/build/scripts/stdout2stderr.py'])

    def output(self):
        base_path = common.tobuilddir(common.stripext(self._path))
        return [
            (base_path + self.extension(), []),
            (base_path + self.schema_extension(), ['noauto'])]

    def run(self, binary):
        return self.do_run(binary, self._path)

    def do_run(self, binary, path):
        def incls():
            for x in self._incl_dirs:
                yield '-I'
                yield self.resolve_path(x)
        output_dir = os.path.dirname(self.resolve_path(common.get(self.output, 0)))
        cmd = common.get_interpreter_path() + ['$S/build/scripts/stdout2stderr.py', binary, '--cpp', '--keep-prefix', '--schema', '-b'] + list(incls()) + ['-o', output_dir, path]
        self.call(cmd)


class Flatc(FlatcBase):

    def __init__(self, path, unit):
        super(Flatc, self).__init__(path, unit)

    def descr(self):
        return 'FL', self._path, 'light-green'

    def tools(self):
        return ['contrib/tools/flatc']

    def extension(self):
        return ".fbs.h"

    def schema_extension(self):
        return ".bfbs"


class Flatc64(FlatcBase):

    def __init__(self, path, unit):
        super(Flatc64, self).__init__(path, unit)

    def descr(self):
        return 'FL64', self._path, 'light-green'

    def tools(self):
        return ['contrib/tools/flatc64']

    def extension(self):
        return ".fbs64.h"

    def schema_extension(self):
        return ".bfbs64"


class FlatcParserBase(object):

    def __init__(self, path, unit):
        self._path = path
        retargeted = os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))

        with open(path, 'r') as f:
            includes, induced = self.parse_includes(f.readlines())

        induced.append(self.contrib_path() + '/include/flatbuffers/flatbuffers.h')

        self._includes = unit.resolve_include([retargeted] + includes) if includes else []
        self._induced = unit.resolve_include([retargeted] + induced) if induced else []

    @classmethod
    def parse_includes(cls, lines):
        includes = []
        induced = []

        for line in lines:
            m = INCLUDE_PATTERN.match(line)

            if m:
                incl = m.group(1)
                includes.append(incl)
                induced.append(common.stripext(incl) + cls.extension())

        return includes, induced

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {'h': self._induced}


class FlatcParser(FlatcParserBase):

    def __init__(self, path, unit):
        super(FlatcParser, self).__init__(path, unit)

    @classmethod
    def extension(cls):
        return ".fbs.h"

    def contrib_path(self):
        return "contrib/libs/flatbuffers"


class FlatcParser64(FlatcParserBase):

    def __init__(self, path, unit):
        super(FlatcParser64, self).__init__(path, unit)

    @classmethod
    def extension(self):
        return ".fbs64.h"

    def contrib_path(self):
        return "contrib/libs/flatbuffers64"


def init():
    iw.addrule('fbs', Flatc)
    iw.addparser('fbs', FlatcParser)

    iw.addrule('fbs64', Flatc64)
    iw.addparser('fbs64', FlatcParser64)
