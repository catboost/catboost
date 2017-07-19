import sys
import os

ERROR_PRONE_FLAGS = [
    '-Xep:FunctionalInterfaceMethodChanged:WARN',
    '-Xep:ReturnValueIgnored:WARN',
]


def just_do_it(argv):
    java, error_prone_tool, javac_cmd = argv[0], argv[1], argv[2:]
    os.execv(java, [java, '-Xbootclasspath/p:' + error_prone_tool, 'com.google.errorprone.ErrorProneCompiler'] + ERROR_PRONE_FLAGS + javac_cmd)


if __name__ == '__main__':
    just_do_it(sys.argv[1:])
