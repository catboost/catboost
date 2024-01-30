import argparse
import os
import stat
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

    compression_mode = ''
    if args.output.endswith(('.tar.gz', '.tgz')):
        compression_mode = 'gz'
    elif args.output.endswith('.bzip2'):
        compression_mode = 'bz2'

    with tarfile.open(args.output, 'w:{}'.format(compression_mode)) as out:
        for root, dirs, files in os.walk(args.input, topdown=True):
            dirs.sort()
            for name in sorted(files):
                fname = os.path.join(root, name)
                if args.exts and not fname.endswith(tuple(args.exts)):
                    continue
                arcname = os.path.basename(fname) if args.flat else os.path.relpath(fname, args.input)
                if args.prefix:
                    arcname = os.path.join(args.prefix, arcname)
                with open(fname, 'rb') as fin:
                    tarinfo = out.gettarinfo(fname, arcname)
                    tarinfo.mode = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if tarinfo.mode | stat.S_IXUSR else 0
                    tarinfo.mode = (
                        tarinfo.mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH
                    )
                    tarinfo.mtime = 0
                    tarinfo.uid = 0
                    tarinfo.gid = 0
                    tarinfo.uname = 'dummy'
                    tarinfo.gname = 'dummy'
                    out.addfile(tarinfo, fin)


if __name__ == '__main__':
    main()
