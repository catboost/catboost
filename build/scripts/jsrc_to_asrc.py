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

    src_dir = os.path.join(args.prefix, 'src')
    os.makedirs(src_dir)

    files = []
    for jsrc in filter(lambda x: x.endswith('.jsrc'), args.input):
        with tarfile.open(jsrc, 'r') as tar:
            files.extend(tar.getnames())
            tar.extractall(path=src_dir)

    with tarfile.open(args.output, 'w') as out:
        for f in files:
            out.add(os.path.join(src_dir, f), arcname=os.path.join('src', f))


if __name__ == '__main__':
    main()
