import argparse
import os
import process_command_files as pcf
import tarfile
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest-dir', required=True)
    parser.add_argument('--skip-prefix', dest='skip_prefixes', action='append', default=[])
    parser.add_argument('docs', nargs='*')
    return parser.parse_args(pcf.get_args(sys.argv[1:]))


def main():
    args = parse_args()

    prefixes = ['{}{}'.format(os.path.normpath(p), os.path.sep) for p in args.skip_prefixes]

    for src in filter(lambda(p): os.path.basename(p) == 'preprocessed.tar.gz', args.docs):
        rel_dst = os.path.dirname(os.path.normpath(src))
        for prefix in prefixes:
            if src.startswith(prefix):
                rel_dst = src[len(prefix):]
        dest_dir = os.path.join(args.dest_dir, rel_dst)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with tarfile.open(src, 'r') as tar_file:
            tar_file.extractall(dest_dir)


if __name__ == '__main__':
    main()
