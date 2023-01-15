import os
import sys


def make_cp_file(args):
    source = args[0]
    destination = args[1]
    with open(source) as src:
        lines = [l.strip() for l in src if l.strip()]
    with open(destination, 'w') as dst:
        dst.write(os.pathsep.join(lines))


if __name__ == '__main__':
    make_cp_file(sys.argv[1:])
