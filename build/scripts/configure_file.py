#!/usr/bin/env python2.7

import sys
import os.path

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
            l = l[:sPos] + define + ' ' + var + ' ' + val + l[ePos + 1:] + '\n'

    while True:
        p1 = l.find('@')
        if p1 == -1:
            return l

        p2 = l.find('@', p1 + 1)
        if p2 == -1:
            return l

        l = l[:p1] + varDict.get(l[p1 + 1:p2], '') + l[p2 + 1:]


def main(inputPath, outputPath, varDict):
    define = '#define' if os.path.splitext(outputPath)[1] != '.asm' else '%define'
    with open(outputPath, 'w') as output:
        with open(inputPath, 'r') as input:
            for l in input:
                output.write(replaceLine(l, varDict, define))


def usage():
    print "usage: configure_file.py inputPath outputPath key1=value1 ..."
    exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()
    varDict = {}
    for x in sys.argv[3:]:
        key, value = str(x).split('=')
        varDict[key] = value

    main(sys.argv[1], sys.argv[2], varDict)
