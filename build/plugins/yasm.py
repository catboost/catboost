import re
import os

import _import_wrapper as iw

INCLUDE_PATTERN = re.compile(' *%(include|INCLUDE) *"([^"]*)"')


def get_retargeted(path, unit):
    return os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))


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
    iw.addparser('asm', YasmParser)
