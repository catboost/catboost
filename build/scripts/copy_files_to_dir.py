import argparse
import errno
import os
import process_command_files as pcf
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest-dir', required=True)
    parser.add_argument('--existing', choices=('skip', 'overwrite'), default='overwrite')
    parser.add_argument('--flat', action='store_true')
    parser.add_argument('--skip-prefix', dest='skip_prefixes', action='append', default=[])
    parser.add_argument('files', nargs='*')
    return parser.parse_args(pcf.get_args(sys.argv[1:]))


def makedirs(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(dirname):
            pass
        else:
            raise


def main():
    args = parse_args()

    dest_dir = os.path.normpath(args.dest_dir) + os.pathsep
    makedirs(dest_dir)

    prefixes = ['{}{}'.format(os.path.normpath(p), os.path.sep) for p in args.skip_prefixes]

    for src in args.files:
        src = os.path.normpath(src)
        assert os.path.isfile(src)
        if args.flat:
            rel_dst = os.path.basename(src)
        else:
            rel_dst = src
            for prefix in prefixes:
                if src.startswith(prefix):
                    rel_dst = src[len(prefix):]
                    break
        assert not os.path.isabs(rel_dst)
        dst = os.path.join(args.dest_dir, rel_dst)
        if os.path.isfile(dst) and args.existing == 'skip':
            break

        makedirs(os.path.dirname(dst))

        shutil.copyfile(src, dst)


if __name__ == '__main__':
    main()
