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

    files = []
    for asrc in filter(lambda x: x.endswith('.asrc') and os.path.exists(x), args.input):
        with tarfile.open(asrc, 'r') as tar:
            files.extend(tar.getnames())
            tar.extractall(path=args.prefix)

    with tarfile.open(args.output, 'w') as out:
        for f in files:
            out.add(os.path.join(args.prefix, f), arcname=f)


if __name__ == '__main__':
    main()
