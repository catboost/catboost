from __future__ import print_function

import os
import sys
import argparse
import tarfile
import subprocess


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def pack_dir(dir_path, dest_path):
    dir_path = os.path.abspath(dir_path)
    for tar_exe in ('/usr/bin/tar', '/bin/tar'):
        if is_exe(tar_exe):
            subprocess.check_call([tar_exe, '-cf', dest_path, '-C', os.path.dirname(dir_path), os.path.basename(dir_path)])
            break
    else:
        with tarfile.open(dest_path, 'w') as out:
            out.add(dir_path, arcname=os.path.basename(dir_path))


def unpack_dir(tared_dir, dest_path):
    tared_dir = os.path.abspath(tared_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for tar_exe in ('/usr/bin/tar', '/bin/tar'):
        if is_exe(tar_exe):
            subprocess.check_call([tar_exe, '-xf', tared_dir, '-C', dest_path])
            break
    else:
        with tarfile.open(tared_dir, 'r') as tar_file:
            tar_file.extractall(dest_path)


# Must only be used to pack directories in build root
# Must silently accept empty list of dirs and do nothing in such case (workaround for ymake.core.conf limitations)
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', action='store_true', default=False)
    parser.add_argument('--unpack', action='store_true', default=False)
    parser.add_argument('--ext')
    parser.add_argument('--outs', nargs='*', default=[])
    parser.add_argument('dirs', nargs='*')
    args = parser.parse_args(args)

    if args.pack:
        if len(args.dirs) != len(args.outs):
            print("Number and oder of dirs to pack must match to the number and order of outs", file=sys.stderr)
            return 1
        for dir, dest in zip(args.dirs, args.outs):
            pack_dir(dir, dest)
    elif args.unpack:
        for tared_dir in args.dirs:
            if not tared_dir.endswith(args.ext):
                print("Requested to unpack '{}' which do not have required extension '{}'".format(tared_dir, args.ext), file=sys.stderr)
                return 1
            dest = os.path.dirname(tared_dir)
            unpack_dir(tared_dir, dest)
    else:
        print("Neither --pack nor --unpack specified. Don't know what to do.", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
