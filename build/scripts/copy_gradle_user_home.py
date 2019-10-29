import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    shutil.copytree(args.input, args.output, ignore=lambda _, x: filter(lambda y: y.endswith('.lock'), x))
    for root, _, files in os.walk(args.output):
        os.chmod(root, 0o775)
        for f in files:
            os.chmod(os.path.join(root, f), 0o775)


if __name__ == '__main__':
    main()
