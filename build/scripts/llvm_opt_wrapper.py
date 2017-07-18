import os
import sys


def fix(s):
    # we use '#' instead of ',' because ymake always splits args by comma
    if 'internalize' in s:
        return s.replace('#', ',')

    return s


if __name__ == '__main__':
    path = sys.argv[1]
    args = [fix(s) for s in [path] + sys.argv[2:]]

    os.execv(path, args)
