import argparse
import os
import tarfile
import stat
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('args', nargs='*')
    return parser.parse_args()


def main(args):
    rawprotos = args.args
    with tarfile.open(args.output, 'w') as fout:
        for rawproto in sorted(os.path.normpath(r).replace('\\', '/') for r in rawprotos):
            assert rawproto.endswith('.rawproto')
            arcname = rawproto[:-len('.rawproto')]
            with open(rawproto, 'rb') as fin:
                tarinfo = fout.gettarinfo(rawproto, arcname)
                tarinfo.mode = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if tarinfo.mode | stat.S_IXUSR else 0
                tarinfo.mode = (
                    tarinfo.mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH
                )
                tarinfo.mtime = 0
                tarinfo.uid = 0
                tarinfo.gid = 0
                tarinfo.uname = 'dummy'
                tarinfo.gname = 'dummy'
                fout.addfile(tarinfo, fin)
    return 0


if __name__ == '__main__':
    sys.exit(main(parse_args()))
