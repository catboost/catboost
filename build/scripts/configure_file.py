#!/usr/bin/env python2.7

import sys
import os.path
import re

cmakeDef01 = "#cmakedefine01"
cmakeDef = "#cmakedefine"


def replaceLine(l, varDict, define):
    words = l.split()
    if words:
        if words[0] == cmakeDef:
            sPos = l.find(cmakeDef)
            ePos = sPos + len(cmakeDef)
            l = l[:sPos] + define + l[ePos:] + '\n'
        if words[0] == cmakeDef01:
            var = words[1]
            cmakeValue = varDict.get(var)
            if cmakeValue == 'yes':
                val = '1'
            else:
                val = '0'
            sPos = l.find(cmakeDef01)
            ePos = l.find(var) + len(var)
            l = l[:sPos] + define + ' ' + var + ' ' + val + l[ePos + 1 :] + '\n'

    finder = re.compile(".*?(@[a-zA-Z0-9_]+@).*")
    while True:
        re_result = finder.match(l)
        if not re_result:
            return l
        key = re_result.group(1)[1:-1]
        l = l[: re_result.start(1)] + varDict.get(key, '') + l[re_result.end(1) :]


def main(inputPath, outputPath, varDict):
    define = '#define' if os.path.splitext(outputPath)[1] != '.asm' else '%define'
    with open(outputPath, 'w') as output:
        with open(inputPath, 'r') as input:
            for l in input:
                output.write(replaceLine(l, varDict, define))


def usage():
    print("usage: configure_file.py inputPath outputPath key1=value1 ...")
    exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()
    varDict = {}
    for x in sys.argv[3:]:
        key, value = str(x).split('=', 1)
        varDict[key] = value

    main(sys.argv[1], sys.argv[2], varDict)
