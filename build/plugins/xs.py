import os
import re

import _import_wrapper as iw


class XsParser(object):

    def __init__(self, path, unit):
        self._path = path
        retargeted = os.path.join(unit.path(), os.path.basename(path))
        with open(path, 'rb') as f:
            includes, induced = XsParser.parse_includes(f.readlines())

        self._includes = unit.resolve_include([retargeted] + includes) if includes else []
        self._induced = unit.resolve_include([retargeted] + induced) if induced else []

    @staticmethod
    def parse_includes(lines):
        includes = []
        induced = []

        include_pattern = re.compile(r'INCLUDE\s*:\s*(?P<path>\S*)')
        induced_pattern = re.compile(r'\#include\s*["<](?P<path>[^">]*)')

        for line in lines:
            line = line.lstrip()

            comment_pos = line.find('//')

            if comment_pos != -1:
                line = line[:comment_pos]  # assumes there are no cases like #include "a//b/c.h"

            if line.startswith('#include'):
                m = induced_pattern.match(line)

                if m:
                    induced.append(m.group('path'))

            elif line.startswith('INCLUDE'):
                m = include_pattern.match(line)

                if m:
                    includes.append(m.group('path'))

        return includes, induced

    def includes(self):
        return self._includes

    def induced_deps(self):
        return {'cpp': self._induced}


def init():
    if 1:
        iw.addparser('xs', XsParser, {'xs': 'use', 'xscpp': 'pass'})


# ----------------Plugin test------------------ #
def test_include_parser():
    text = '''
aaaaa
#include <induced1>
# include <not_induced>
#include<induced2>
INCLUDE: included1
INCLUDE : included2
INCLUDE: included3  // asdasd
//INCLUDE : not_included
'''
    includes, induced = XsParser.parse_includes(text.split('\n'))
    assert includes == ['included1', 'included2', 'included3', ]
    assert induced == ['induced1', 'induced2']
