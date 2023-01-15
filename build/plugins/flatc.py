import os

import _import_wrapper as iw
import _common as common


class FlatcBase(iw.CustomCommand):

    def __init__(self, path, unit):
        self._path = path
        self._incl_dirs = ['$S', '$B']

    def input(self):
        return common.make_tuples([self._path, '$S/build/scripts/stdout2stderr.py'])

    def output(self):
        base_path = common.tobuilddir(common.stripext(self._path))
        return [(base_path + extension, []) for extension in self.extensions()] +\
            [(base_path + self.schema_extension(), ['noauto'])]

    def run(self, extra_args, binary):
        return self.do_run(binary, self._path)

    def do_run(self, binary, path):
        def incls():
            for x in self._incl_dirs:
                yield '-I'
                yield self.resolve_path(x)
        output_dir = os.path.dirname(self.resolve_path(common.get(self.output, 0)))
        cmd = (common.get_interpreter_path() +
               ['$S/build/scripts/stdout2stderr.py', binary, '--cpp', '--keep-prefix', '--gen-mutable', '--schema', '-b'] +
               self.extra_arguments() +
               list(incls()) +
               ['-o', output_dir, path])
        self.call(cmd)


class Flatc(FlatcBase):

    def __init__(self, path, unit):
        super(Flatc, self).__init__(path, unit)

    def descr(self):
        return 'FL', self._path, 'light-green'

    def tools(self):
        return ['contrib/tools/flatc']

    def extensions(self):
        return [".fbs.h", ".iter.fbs.h"]

    def schema_extension(self):
        return ".bfbs"

    def extra_arguments(self):
        return ['--yandex-maps-iter']


class Flatc64(FlatcBase):

    def __init__(self, path, unit):
        super(Flatc64, self).__init__(path, unit)

    def descr(self):
        return 'FL64', self._path, 'light-green'

    def tools(self):
        return ['contrib/tools/flatc64']

    def extensions(self):
        return [".fbs64.h"]

    def schema_extension(self):
        return ".bfbs64"

    def extra_arguments(self):
        return []


def init():
    iw.addrule('fbs', Flatc)

    iw.addrule('fbs64', Flatc64)
