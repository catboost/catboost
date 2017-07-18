import os

from _xsyn_includes import get_all_includes

import _import_wrapper as iw
import _common as common


class Xsyn(iw.CustomCommand):

    def __init__(self, path, unit):
        self._path = path

    def descr(self):
        return 'XN', self._path, 'yellow'

    def tools(self):
        return []

    def input(self):
        return common.make_tuples([
            '$S/library/xml/parslib/xsyn2ragel.py',
            self._path,
            '$S/library/xml/parslib/xmlpars.xh'
        ])

    def output(self):
        return common.make_tuples([
            common.tobuilddir(self._path + '.h.rl5')
        ])

    def run(self, interpeter):
        self.call(interpeter + [self.resolve_path(common.get(self.input, 0)), self.resolve_path(common.get(self.input, 1)),
                                self.resolve_path(common.get(self.input, 2)), 'dontuse'], stdout=common.get(self.output, 0))


class XsynParser(object):
    def __init__(self, path, unit):
        self._path = path
        self._includes = []
        retargeted = os.path.join(unit.path(), os.path.basename(path))

        self._includes = get_all_includes(path)
        self._includes = unit.resolve_include([retargeted] + self._includes) if self._includes else []

    def includes(self):
        return self._includes


def init():
    iw.addrule('xsyn', Xsyn)
    iw.addparser('xsyn', XsynParser)
