import os
import sys


if __name__ == '__main__':
    path = sys.argv[1]

    if path[0] != '/':
        path = os.path.join(os.path.dirname(__file__), path)

    os.execv(path, [path] + sys.argv[2:])
