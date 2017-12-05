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
