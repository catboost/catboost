#!/usr/bin/env python2.7

from __future__ import print_function
import sys
import os.path
import re

cmakeDef01 = "#cmakedefine01"
cmakeDef = "#cmakedefine"


def replaceLine(line, varDict, define):
    words = line.split()
    if words:
        if words[0] == cmakeDef:
            sPos = line.find(cmakeDef)
            ePos = sPos + len(cmakeDef)
            line = line[:sPos] + define + line[ePos:] + '\n'
        if words[0] == cmakeDef01:
            var = words[1]
            cmakeValue = varDict.get(var)
            if cmakeValue == 'yes':
                val = '1'
            else:
                val = '0'
            sPos = line.find(cmakeDef01)
            ePos = line.find(var) + len(var)
            line = line[:sPos] + define + ' ' + var + ' ' + val + line[ePos + 1 :] + '\n'

    finder = re.compile(".*?(@[a-zA-Z0-9_]+@).*")
    while True:
        re_result = finder.match(line)
        if not re_result:
            return line
        key = re_result.group(1)[1:-1]
        line = line[: re_result.start(1)] + varDict.get(key, '') + line[re_result.end(1) :]


def main(inputPath, outputPath, varDict):
    define = '#define' if os.path.splitext(outputPath)[1] != '.asm' else '%define'
    with open(outputPath, 'w') as output:
        with open(inputPath, 'r') as input:
            for line in input:
                output.write(replaceLine(line, varDict, define))


def usage():
    print("usage: configure_file.py inputPath outputPath key1=value1 ...")
    exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()
    varDict = {}
    for x in sys.argv[3:]:
        try:
            key, value = str(x).split('=', 1)
            value = value.replace("#BACKSLASH#", "\\\\")
            value = value.replace("#DOUBLE_QUOTE#", '"')
        except Exception:
            continue
        varDict[key] = value

    main(sys.argv[1], sys.argv[2], varDict)
