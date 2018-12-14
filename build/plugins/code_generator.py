import re
import os

import _import_wrapper as iw

pattern = re.compile("#include\s*[<\"](?P<INDUCED>[^>\"]+)[>\"]|(?:@|{@)\s*(?:import|include|from)\s*[\"'](?P<INCLUDE>[^\"']+)[\"']")


class CodeGeneratorTemplateParser(object):
    def __init__(self, path, unit):
        self._path = path
        retargeted = os.path.join(unit.path(), os.path.relpath(path, unit.resolve(unit.path())))
        with open(path, 'rb') as f:
            includes, induced = CodeGeneratorTemplateParser.parse_includes(f.readlines())
        self._includes = unit.resolve_include([retargeted] + includes) if includes else []
        self._induced = unit.resolve_include([retargeted] + induced) if induced else []

    @staticmethod
    def parse_includes(lines):
        includes = []
        induced = []

        for line in lines:
            for match in pattern.finditer(line):
                type = match.lastgroup
                if type == 'INCLUDE':
                    includes.append(match.group(type))
                elif type == 'INDUCED':
                    induced.append(match.group(type))
                else:
                    raise Exception("Unexpected match! Perhaps it is a result of an error in pattern.")
        return (includes, induced)

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {
            'h+cpp': self._induced
        }


def init():
    iw.addparser('template', CodeGeneratorTemplateParser)
    iw.addparser('macro', CodeGeneratorTemplateParser)
