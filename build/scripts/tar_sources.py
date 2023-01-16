import argparse
import os
import tarfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exts', nargs='*', default=None)
    parser.add_argument('--flat', action='store_true')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    py_srcs = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if not args.exts or f.endswith(tuple(args.exts)):
                py_srcs.append(os.path.join(root, f))

    compression_mode = ''
    if args.output.endswith(('.tar.gz', '.tgz')):
        compression_mode = 'gz'
    elif args.output.endswith('.bzip2'):
        compression_mode = 'bz2'

    with tarfile.open(args.output, 'w:{}'.format(compression_mode)) as out:
        for f in py_srcs:
            arcname = os.path.basename(f) if args.flat else os.path.relpath(f, args.input)
            if args.prefix:
                arcname = os.path.join(args.prefix, arcname)
            out.add(f, arcname=arcname)


if __name__ == '__main__':
    main()
