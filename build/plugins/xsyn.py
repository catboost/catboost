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
            '$S/library/cpp/xml/parslib/xsyn2ragel.py',
            self._path,
            '$S/library/cpp/xml/parslib/xmlpars.xh'
        ])

    def output(self):
        return common.make_tuples([
            common.tobuilddir(self._path + '.h.rl5')
        ])

    def run(self, extra_args, interpeter):
        self.call(interpeter + [self.resolve_path(common.get(self.input, 0)), self.resolve_path(common.get(self.input, 1)),
                                self.resolve_path(common.get(self.input, 2)), 'dontuse'], stdout=common.get(self.output, 0))


def init():
    iw.addrule('xsyn', Xsyn)
