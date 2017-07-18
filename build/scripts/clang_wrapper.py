import os
import sys


def fix(s):
    # We use Clang 3.7 for generating bytecode but take CFLAGS from current compiler
    # We should filter new flags unknown to the old compiler
    if s == '-Wno-undefined-var-template':
        return None

    # disable dbg DEVTOOLS-2744
    if s == '-g':
        return None

    return s


if __name__ == '__main__':
    path = sys.argv[1]
    args = filter(None, [fix(s) for s in [path] + sys.argv[2:]])

    os.execv(path, args)
