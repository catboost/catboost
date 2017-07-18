import os
import sys
import tarfile


def main(args):
    coverage_path = os.path.abspath(args[0])
    with tarfile.open(coverage_path, 'w:'):
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
