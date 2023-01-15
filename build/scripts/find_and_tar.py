import os
import sys
import tarfile


def find_gcno(dirname, tail):
    for cur, _dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(tail):
                yield os.path.relpath(os.path.join(cur, f))


def main(args):
    output = args[0]
    tail = args[1] if len(args) > 1 else ''
    with tarfile.open(output, 'w:') as tf:
        for f in find_gcno(os.getcwd(), tail):
            tf.add(f)


if __name__ == '__main__':
    main(sys.argv[1:])
