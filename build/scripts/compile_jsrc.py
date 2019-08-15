import argparse
import os
import tarfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='*', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefix', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    with tarfile.open(args.output, 'w') as out:
        for f in args.input:
            out.add(f, arcname=os.path.relpath(f, args.prefix))


if __name__ == '__main__':
    main()
