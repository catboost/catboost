import re
import os

import _import_wrapper as iw

INCLUDE_PATTERN = re.compile("\s*include *'([^']*)'")


class FortranParser(object):
    def __init__(self, path, unit):
        self._path = path

        with open(path, 'r') as f:
            induced = list(self.parse_includes(f))

        retargeted = os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))
        self._induced = unit.resolve_include([retargeted] + induced) if induced else []

    @staticmethod
    def parse_includes(lines):
        for line in lines:
            m = INCLUDE_PATTERN.match(line)
            if m:
                yield m.group(1)

    def includes(self):
        return self._induced

    def induced_deps(self):
        return {'c': self._induced}


def init():
    iw.addparser('f', FortranParser)
