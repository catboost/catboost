#!/usr/bin/env python

import argparse
import os
import shutil
import sys
import zipfile

if sys.version_info[0] == 2:
    import distutils.dir_util


def _main():
    parser = argparse.ArgumentParser(
        description='Prepare source code from multiple sources for processing by java/scaladoc'
    )
    parser.add_argument('--src-dirs', nargs='*', help='dirs with source files')
    parser.add_argument('--src-jars', nargs='*', help='jars with source files')
    parser.add_argument(
        '--dst-dir',
        help='destination directory (if directory already exists it will be overwritten)'
    )

    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        shutil.rmtree(args.dst_dir)
    os.makedirs(args.dst_dir)

    if args.src_dirs is not None:
        for src_dir in args.src_dirs:
            if sys.version_info[0] == 2:
                distutils.dir_util.copy_tree(src_dir, args.dst_dir)
            else:
                shutil.copytree(src_dir, args.dst_dir, dirs_exist_ok=True)

    if args.src_jars is not None:
        for src_jar in args.src_jars:
            with zipfile.ZipFile(src_jar, 'r') as zf:
                zf.extractall(args.dst_dir)


if '__main__' == __name__:
    _main()
